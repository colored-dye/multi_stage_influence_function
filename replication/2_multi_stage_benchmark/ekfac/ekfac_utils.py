import os
import types
from functools import partial
from typing import Union
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers.pytorch_utils import Conv1D


def vectorize(x: torch.Tensor):
    return x.t().reshape(-1)


def unvectorize(m: torch.tensor, h: int, w: int):
    """
    `m` is a 1D tensor. Reshape into (h,w).
    """
    return m.reshape(w, h).t()


def extend(model: nn.Module, white_list: dict[str], quiet=True):
    """
    Insert methods for gathering intermediate variables.
    """
    model.modules_of_interest = {}
    for name, module in model.named_modules():
        if not any([white in name for white in white_list]):
            continue

        model.modules_of_interest[name] = white_list[name]
        if not quiet:
            logger.info(f"{name}: {white_list[name]}")

    def intermediate_inputs(self):
        intermediate_inputs = {}
        for module_name in self.modules_of_interest:
            x = self.x_ins[module_name]
            if model.get_submodule(module_name).bias is not None:
                x = torch.cat(
                    [x, torch.ones(x.shape[:-1]+(1,), device=x.device)], dim=-1)
            intermediate_inputs[module_name] = x
        return intermediate_inputs

    # def intermediate_outputs(self):
    #     intermediate_outputs = {}
    #     for module_name in self.modules_of_interest:
    #         x = self.x_outs[module_name]
    #         intermediate_outputs[module_name] = x
    #     return intermediate_outputs

    def parameter_gradients(self):
        grads = {}
        for module_name in self.modules_of_interest:
            module = model.get_submodule(module_name)
            w_grad = module.weight.grad
            if isinstance(module, Conv1D):
                w_grad = w_grad.t()
            assert w_grad is not None
            if module.bias is not None:
                b_grad = module.bias.grad
                grad = torch.cat([w_grad, b_grad.unsqueeze(-1)], dim=1)
            else:
                grad = w_grad
            grads[module_name] = grad
        return grads

    def output_gradients(self):
        """
        Gradients of intermediate outputs.
        ```
        net = nn.Linear()
        y = net(x)
        loss = loss_fct(y, lbl)
        loss.backward()
        ```
        Returns:
            `y.grad`
        """
        jac_outputs_to_logits = {}
        for module_name in self.modules_of_interest:
            dz_dy = self.gradients[module_name]
            jac_outputs_to_logits[module_name] = dz_dy
        return jac_outputs_to_logits

    model.intermediate_inputs = types.MethodType(intermediate_inputs, model)
    # model.intermediate_outputs = types.MethodType(intermediate_outputs, model)
    model.parameter_gradients = types.MethodType(parameter_gradients, model)
    model.output_gradients = types.MethodType(output_gradients, model)


def retract(model: nn.Module):
    """Remove methods.
    """
    del model.modules_of_interest

    del model.intermediate_inputs
    # del model.intermediate_outputs
    del model.parameter_gradients
    del model.output_gradients


class JacobianMode:
    def __init__(self, model: nn.Module):
        self.model = model
        if not isinstance(model, nn.Module):
            raise TypeError("model should be a nn.Module")

    def __enter__(self):
        model = self.model
        self.model.x_ins = {}
        self.model.x_outs = {}
        self.model.gradients = {}
        self.forward_hooks = []
        self.backward_hooks = []

        def record_forward(module, input, output, layer):
            model.x_ins[layer] = input[0].detach()
            # model.x_outs[layer] = output.detach()

        def record_backward(module, grad_input, grad_output, layer):
            model.gradients[layer] = grad_output[0]

        for i, module_name in enumerate(self.model.modules_of_interest):
            module = self.model.get_submodule(module_name)
            self.model.x_ins[i] = None
            self.model.x_outs[i] = None
            self.model.gradients[i] = None
            self.forward_hooks.append(
                module.register_forward_hook(
                    partial(record_forward, layer=module_name)
                )
            )
            self.backward_hooks.append(
                module.register_full_backward_hook(
                    partial(record_backward, layer=module_name))
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook in self.forward_hooks:
            hook.remove()
        for hook in self.backward_hooks:
            hook.remove()

        del self.model.x_ins
        del self.model.x_outs
        del self.model.gradients


def sample_labels(logits: torch.Tensor):
    """Feed a logit tensor, and output labels via a single Monte-Carlo sampling.
    """
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    with torch.no_grad():
        targets = dist.sample()
    return targets


def cycle_through_dataloader(iterator, loader: DataLoader):
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator._reset(loader)
            yield next(iterator)


def get_log(x: torch.Tensor, cur_iter: int, batch_size: int, max_length: int):
    return torch.norm(x.double()/((cur_iter+1)*max_length*batch_size)).detach().cpu().numpy()


@torch.jit.script
def make_cov(x: torch.Tensor):
    return x.T @ x


def ekfac_fit_covariance(model: nn.Module,
                         device: str,
                         device_backup: str,
                         modules_of_interest: dict[str],
                         dataloader: DataLoader,
                         n_iters: int,
                         batch_size: int,
                         max_length: int,
                         hidden_size: int,
                         n_heads: int,
                         do_log: bool,
                         figures_dir: str,
                         dtype=torch.bfloat16,
                         do_finetune: bool = False):
    extend(model, modules_of_interest, quiet=False)

    assert hidden_size % n_heads == 0
    head_size = hidden_size // n_heads

    A = {}
    G = {}
    if do_log:
        logs = {"A": {}, "G": {}}
    for module_name, mlp_or_attention in model.modules_of_interest.items():
        module = model.get_submodule(module_name)
        if isinstance(module, nn.Linear):
            in_dim = module.in_features
            out_dim = module.out_features
        elif isinstance(module, Conv1D):
            in_dim = module.weight.shape[0]
            out_dim = module.weight.shape[1]
        if module.bias is not None:
            A[module_name] = torch.zeros(
                in_dim+1, in_dim+1, device=device_backup, dtype=dtype)
        else:
            A[module_name] = torch.zeros(
                in_dim, in_dim, device=device_backup, dtype=dtype)
        if do_log:
            logs["A"][module_name] = []

        if mlp_or_attention == "mlp":
            G[module_name] = torch.zeros(
                out_dim, out_dim, device=device_backup, dtype=dtype)
            if do_log:
                logs["G"][module_name] = []
        elif mlp_or_attention == "mha":
            G[module_name] = {}
            if do_log:
                logs["G"][module_name] = {}
            for key in ("q", "k", "v"):
                G[module_name][key] = []
                if do_log:
                    logs["G"][module_name][key] = []
                for nh in range(n_heads):
                    G[module_name][key].append(torch.zeros(
                        head_size, head_size, device=device_backup, dtype=dtype))
                    if do_log:
                        logs["G"][module_name][key].append([])
        else:
            raise ValueError(
                f"Unsupported linear module spec: {module_name}:{mlp_or_attention}")

    data_iterator = iter(dataloader)
    for cur_iter in tqdm(range(n_iters), desc="Fitting covariance"):
        with JacobianMode(model):
            if not do_finetune:
                X = next(cycle_through_dataloader(data_iterator, dataloader))
                X = X.to(device)
                logits = model(X)[0]
                logits = logits.view(-1, logits.size(-1))
                Y = sample_labels(logits.double())
            else:
                input_ids, attention_mask, Y = next(cycle_through_dataloader(data_iterator, dataloader))
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                Y = Y.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                label_logits = outputs.logits[Y != -100, :]
                sampled_labels = sample_labels(label_logits.double())
                Y[Y != -100] = sampled_labels
                logits = outputs.logits[:, :-1, :]
                logits = logits.reshape(-1, logits.size(-1))
            loss = F.cross_entropy(logits, Y[:, 1:].flatten(), ignore_index=-100, reduction='sum')
            model.zero_grad()
            loss.backward()

            a_l_minus_one = model.intermediate_inputs()
            d_s_l = model.output_gradients()

        for module_name, mlp_or_attention in model.modules_of_interest.items():
            a = a_l_minus_one[module_name]
            if do_finetune:
                a = a[Y!=-100]
            a = a.reshape(-1, a.size(-1))
            A[module_name] += make_cov(a).to(device_backup)
            if do_log:
                logs["A"][module_name].append(
                    get_log(A[module_name], cur_iter, batch_size, max_length))

            g = d_s_l[module_name]
            if do_finetune:
                g = g[Y!=-100]
            g = g.reshape(-1, g.size(-1))
            if mlp_or_attention == "mlp":
                G[module_name] += make_cov(g).to(device_backup)
                if do_log:
                    logs["G"][module_name].append(
                        get_log(G[module_name], cur_iter, batch_size, max_length))
            elif mlp_or_attention == "mha":
                gq, gk, gv = g.split(hidden_size, dim=-1)
                for key, value in zip(("q", "k", "v"), (gq, gk, gv)):
                    value = value.split(head_size, dim=-1)
                    for i in range(len(value)):
                        G[module_name][key][i] += make_cov(
                            value[i]).to(device_backup)
                        if do_log:
                            logs["G"][module_name][key][i].append(
                                get_log(G[module_name][key][i], cur_iter, batch_size, max_length))

    QA = {}
    QG = {}
    logger.info("Eigendecomposition...")
    for module_name, mlp_or_attention in model.modules_of_interest.items():
        A[module_name] /= n_iters*batch_size*max_length
        try:
            _, qa = torch.linalg.eigh(A[module_name].double())
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM. Switch to CPU.")
            _, qa = torch.linalg.eigh(A[module_name].cpu().double())
        QA[module_name] = qa.to(dtype)
        if mlp_or_attention == "mlp":
            G[module_name] /= n_iters*batch_size*max_length
            try:
                _, qg = torch.linalg.eigh(G[module_name].double())
            except torch.cuda.OutOfMemoryError:
                logger.warning("CUDA OOM. Switch to CPU.")
                _, qg = torch.linalg.eigh(G[module_name].cpu().double())
            QG[module_name] = qg.to(dtype)
        elif mlp_or_attention == "mha":
            QG[module_name] = {}
            for key in ("q", "k", "v"):
                QG[module_name][key] = []
                for nh in range(n_heads):
                    G[module_name][key][nh] /= n_iters*batch_size*max_length
                    try:
                        _, qg = torch.linalg.eigh(
                            G[module_name][key][nh].double())
                    except torch.cuda.OutOfMemoryError:
                        _, qg = torch.linalg.eigh(
                            G[module_name][key][nh].cpu().double())
                    QG[module_name][key].append(qg.to(dtype))
    logger.info("Eigendecomposition finished.")

    if do_log:
        logger.info("Plotting trace of fitting covariances...")
        os.makedirs(figures_dir, exist_ok=True)
        fig_width = 6
        sns.reset_orig()
        sns.set_theme()
        for module_name, mlp_or_attention in model.modules_of_interest.items():
            if mlp_or_attention == "mlp":
                fig, axes = plt.subplots(
                    1, 2, figsize=(2*fig_width, fig_width))
                axes[0].plot(range(len(logs["A"][module_name])),
                             logs["A"][module_name])
                axes[0].set_title("A")
                axes[1].plot(range(len(logs["G"][module_name])),
                             logs["G"][module_name])
                axes[1].set_title("G")

            elif mlp_or_attention == "mha":
                fig = plt.figure(figsize=(n_heads*fig_width, 4*fig_width))
                gs = fig.add_gridspec(4, n_heads)
                ax0 = fig.add_subplot(gs[0, :])
                ax0.plot(range(len(logs["A"][module_name])),
                         logs["A"][module_name])
                ax0.set_title("A")
                for i, key in enumerate(("q", "k", "v")):
                    for nh in range(n_heads):
                        ax = fig.add_subplot(gs[i+1, nh])
                        ax.plot(
                            range(len(logs["G"][module_name][key][nh])), logs["G"][module_name][key][nh])
                        ax.set_title(f"{key}_{nh}")
            figname = os.path.join(
                figures_dir, f"{module_name}.covariance.pdf")
            if os.path.exists(figname):
                os.remove(figname)
            fig.savefig(figname)
            plt.title(module_name)
            plt.close()
        logger.info("Figures saved.")

    retract(model)

    return QA, QG


def ekfac_fit_diagonal(model: nn.Module,
                       device: str,
                       QA: dict[str, Union[torch.Tensor, dict[str, list[torch.Tensor]]]],
                       QG: dict[str, Union[torch.Tensor, dict[str, list[torch.Tensor]]]],
                       modules_of_interest: dict[str],
                       dataloader: DataLoader,
                       n_iters: int,
                       batch_size: int,
                       max_length: int,
                       hidden_size: int,
                       n_heads: int,
                       do_log: bool,
                       figures_dir: str,
                       dtype=torch.bfloat16,
                       do_finetune: bool = False):
    extend(model, modules_of_interest, quiet=False)

    assert hidden_size % n_heads == 0
    head_size = hidden_size // n_heads

    Lambda = {}
    if do_log:
        logs = {}
    for module_name, mlp_or_attention in model.modules_of_interest.items():
        module = model.get_submodule(module_name)
        if isinstance(module, nn.Linear):
            in_dim = module.in_features
            out_dim = module.out_features
        elif isinstance(module, Conv1D):
            in_dim = module.weight.shape[0]
            out_dim = module.weight.shape[1]
        if mlp_or_attention == "mlp":
            if module.bias is not None:
                Lambda[module_name] = torch.zeros(
                    (in_dim+1)*out_dim, device=device, dtype=dtype)
            else:
                Lambda[module_name] = torch.zeros(
                    in_dim*out_dim, device=device, dtype=dtype)
            if do_log:
                logs[module_name] = []
        elif mlp_or_attention == "mha":
            Lambda[module_name] = {}
            if do_log:
                logs[module_name] = {}
            for key in ("q", "k", "v"):
                Lambda[module_name][key] = []
                if do_log:
                    logs[module_name][key] = []
                for nh in range(n_heads):
                    if module.bias is not None:
                        Lambda[module_name][key].append(torch.zeros(
                            (hidden_size+1)*head_size, device=device, dtype=dtype))
                    else:
                        Lambda[module_name][key].append(torch.zeros(
                            hidden_size*head_size, device=device, dtype=dtype))
                    if do_log:
                        logs[module_name][key].append([])
        else:
            raise ValueError(
                f"Unsupported linear module spec: {module_name}:{mlp_or_attention}")

    data_iterator = iter(dataloader)
    for cur_iter in tqdm(range(n_iters), desc="Fitting diagonal"):
        with JacobianMode(model):
            if not do_finetune:
                X = next(cycle_through_dataloader(data_iterator, dataloader))
                X = X.to(device)
                logits = model(X)[0]
                logits = logits.view(-1, logits.size(-1))
                Y = sample_labels(logits.double())
            else:
                input_ids, attention_mask, Y = next(cycle_through_dataloader(data_iterator, dataloader))
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                Y = Y.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                label_logits = outputs.logits[Y != -100, :]
                sampled_labels = sample_labels(label_logits.double())
                Y[Y != -100] = sampled_labels
                logits = outputs.logits[:, :-1, :]
                logits = logits.reshape(-1, logits.size(-1))
            loss = F.cross_entropy(logits, Y[:, 1:].flatten(), ignore_index=-100, reduction='sum')
            model.zero_grad()
            loss.backward()

            gradients = model.parameter_gradients()
        for module_name, mlp_or_attention in model.modules_of_interest.items():
            grads = gradients[module_name]
            if mlp_or_attention == "mlp":
                res = torch.linalg.multi_dot(
                    [QG[module_name].T.to(device), grads, QA[module_name].to(device)])
                res = vectorize(res)
                res = res.pow(2)
                Lambda[module_name] += res
                if do_log:
                    logs[module_name].append(
                        get_log(Lambda[module_name], cur_iter, batch_size, max_length))
            elif mlp_or_attention == "mha":
                grads_q, grads_k, grads_v = grads.split(hidden_size, dim=0)
                grads = {"q": grads_q, "k": grads_k, "v": grads_v}
                for key in grads:
                    grads[key] = grads[key].split(head_size, dim=0)
                    for nh in range(n_heads):
                        res = torch.linalg.multi_dot([QG[module_name][key][nh].T.to(
                            device), grads[key][nh], QA[module_name].to(device)])
                        res = vectorize(res)
                        res = res.pow(2)
                        Lambda[module_name][key][nh] += res
                        if do_log:
                            logs[module_name][key][nh].append(
                                get_log(Lambda[module_name][key][nh], cur_iter, batch_size, max_length))

    for module_name, mlp_or_attention in model.modules_of_interest.items():
        if mlp_or_attention == "mlp":
            Lambda[module_name] /= n_iters*batch_size*max_length
        elif mlp_or_attention == "mha":
            for key in ("q", "k", "v"):
                for nh in range(n_heads):
                    Lambda[module_name][key][nh] /= n_iters * \
                        batch_size*max_length

    if do_log:
        logger.info("Plotting trace of fitting diagonal...")
        os.makedirs(figures_dir, exist_ok=True)
        fig_width = 6
        sns.reset_orig()
        sns.set_theme()
        for module_name, mlp_or_attention in model.modules_of_interest.items():
            if mlp_or_attention == "mlp":
                fig = plt.figure(figsize=(2*fig_width, fig_width))
                plt.plot(range(len(logs[module_name])), logs[module_name])
            elif mlp_or_attention == "mha":
                fig = plt.figure(figsize=(2*fig_width, n_heads*fig_width))
                gs = fig.add_gridspec(n_heads, 3)
                for i, key in enumerate(("q", "k", "v")):
                    for nh in range(n_heads):
                        ax = fig.add_subplot(gs[nh, i])
                        ax.plot(
                            range(len(logs[module_name][key][nh])), logs[module_name][key][nh])
                        ax.set_title(f"{key}_{nh}")
            figname = os.path.join(figures_dir, f"{module_name}.diagonal.pdf")
            if os.path.exists(figname):
                os.remove(figname)
            fig.savefig(figname)
            plt.title(module_name)
            plt.close()
        logger.info("Figures saved.")

    retract(model)

    return Lambda


def example_grads(model: nn.Module,
                  modules_of_interest: dict[str, str],
                  hidden_size: int,
                  n_heads: int,
                  inputs: torch.Tensor,
                  targets: torch.Tensor):
    extend(model, modules_of_interest, quiet=True)
    with JacobianMode(model):
        logits = model(inputs)[0]
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        loss = F.cross_entropy(logits, targets, reduction='sum')
        model.zero_grad()
        loss.backward()

        gradients = model.parameter_gradients()
    retract(model)

    head_size = hidden_size // n_heads
    for module_name, mlp_or_attention in modules_of_interest.items():
        if mlp_or_attention == "mha":
            grads_q, grads_k, grads_v = gradients[module_name].split(
                hidden_size, dim=0)
            gradients[module_name] = {}
            grads = {"q": grads_q, "k": grads_k, "v": grads_v}
            for key in grads:
                gradients[module_name][key] = grads[key].split(
                    head_size, dim=0)

    return gradients


def ekfac_ihvp_single_block(qa: torch.Tensor,
                            qg: torch.Tensor,
                            diagonal: torch.Tensor,
                            damping: float,
                            v: torch.Tensor):
    qg_v_qa = torch.linalg.multi_dot([qg.T, v, qa])
    diagonal += damping
    diagonal = unvectorize(diagonal, v.shape[0], v.shape[1])
    result = qg_v_qa / diagonal
    ihvp = torch.linalg.multi_dot([qg, result, qa.T])
    return ihvp


def ekfac_ihvp(modules_of_interest: dict[str, str],
               QA: dict[str, Union[torch.Tensor, dict[str, list[torch.Tensor]]]],
               QG: dict[str, Union[torch.Tensor, dict[str, list[torch.Tensor]]]],
               Lambda: list[torch.Tensor],
               damping: float,
               vec: dict[str, torch.Tensor]):
    ihvps = {}
    for module_name, mlp_or_attention in modules_of_interest.items():
        if mlp_or_attention == "mlp":
            ihvps[module_name] = ekfac_ihvp_single_block(qa=QA[module_name],
                                                         qg=QG[module_name],
                                                         diagonal=Lambda[module_name],
                                                         damping=damping,
                                                         v=vec[module_name])
        elif mlp_or_attention == "mha":
            ihvps[module_name] = {}
            for key in ("q", "k", "v"):
                ihvps[module_name][key] = []
                for nh in range(len(QG[module_name][key])):
                    ihvp = ekfac_ihvp_single_block(qa=QA[module_name],
                                                   qg=QG[module_name][key][nh],
                                                   diagonal=Lambda[module_name][key][nh],
                                                   damping=damping,
                                                   v=vec[module_name][key][nh])
                    ihvps[module_name][key].append(ihvp)
    return ihvps


def ekfac_influence(modules_of_interest: dict[str, str],
                    ihvps: dict[str, Union[torch.Tensor, dict[str, list[torch.Tensor]]]],
                    grads: dict[str, Union[torch.Tensor, dict[str, list[torch.Tensor]]]]):
    infl_sum = 0
    infls_by_block = {}
    for module_name, mlp_or_attention in modules_of_interest.items():
        if mlp_or_attention == "mlp":
            infl = torch.dot(vectorize(ihvps[module_name]),
                              vectorize(grads[module_name])).detach().cpu().numpy()
            infls_by_block[module_name] = infl
            infl_sum += infl
        elif mlp_or_attention == "mha":
            infls_by_block[module_name] = {}
            for key in ("q", "k", "v"):
                infls_by_block[module_name][key] = []
                for nh in range(len(ihvps[module_name][key])):
                    infl = torch.dot(vectorize(ihvps[module_name][key][nh]), vectorize(
                        grads[module_name][key][nh])).detach().cpu().numpy()
                    infls_by_block[module_name][key].append(infl)
                    infl_sum += infl
    return infl_sum, infls_by_block
