import copy
import os
import pickle
import time
from tqdm import tqdm
from typing import Callable

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, jvp, vjp
from torch.utils.data import DataLoader
import einops

from modeling_gpt_neox import GPTNeoXForCausalLM
from transformers import (
    AutoTokenizer, AutoConfig
)
from datasets import load_dataset

import utils


pretrained_model_path = "/home/Model/Pythia/pythia-70m"
dataset_path = "/home/Dataset/ptb_text_only"

seed = 42
device = "cuda:3"
max_length = 512
batch_size = 32
n_iters = 200
top_k = 20

do_mlp = False
if do_mlp:
    save_dir = "infls/arnoldi-mlp/"
else:
    save_dir = "infls/arnoldi-all/"

do_arnoldi_iter = True

silence_hint = True
use_hvp_estimator = False


def prepare_normal_start_vector(shape_like: dict[str, torch.Tensor]):
    start_vector = {n: torch.empty_like(p) for n, p in shape_like.items()}
    for n in start_vector.keys():
        torch.nn.init.normal_(start_vector[n])
    return start_vector


def ravel_named_params(d: dict[str, torch.Tensor]):
    names = []
    sizes = []
    tmp = []
    for n, p in d.items():
        names.append(n)
        sizes.append(p.shape)
        tmp.append(p.t().reshape(-1))
    def unravel(params_flattened: torch.Tensor):
        tmp = {}
        pointer = 0
        for n, s in zip(names, sizes):
            np = s[0]*s[1] if len(s) == 2 else s[0]
            s = (s[1], s[0]) if len(s) == 2 else s
            tmp[n] = params_flattened[pointer:pointer+np].view(s).t().detach()
            pointer += np
        return tmp
    return torch.cat(tmp).detach(), unravel


def inner_product_named_params(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]):
    a_ = ravel_named_params(a)[0]
    b_ = ravel_named_params(b)[0]
    return torch.dot(a_, b_)


def multiply_by_scalar_named_params(a: dict[str, torch.Tensor], b: float):
    res = {n: p.clone() for n, p in a.items()}
    for n in res.keys():
        res[n] *= b
    return res


def to_device_named_params(a: dict[str, torch.Tensor], device: str):
    return {n: p.detach().to(device) for n, p in a.items()}


def add_scalar_multiple(t: dict[str, torch.Tensor], scalar: float, to_add: dict[str, torch.Tensor]):
    res = {n: p.clone() for n, p in t.items()}
    for n in res.keys():
        res[n] += scalar * to_add[n]
    return res


def arnoldi_iter(hvp_est_fn: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
                 start_vector: dict[str, torch.Tensor],
                 n_iters: int,
                 device: str,
                 stop_tol: float = 1e-6):
    proj = []
    appr_mat = torch.zeros((n_iters, n_iters-1))
    v0_norm = torch.sqrt(inner_product_named_params(
        start_vector, start_vector))
    vec0 = multiply_by_scalar_named_params(start_vector, 1.0 / v0_norm)
    proj.append(to_device_named_params(vec0, "cpu"))

    for n in tqdm(range(n_iters-1), desc="Arnolti iters"):
        vec = hvp_est_fn(to_device_named_params(proj[n], device))
        vec = to_device_named_params(vec, "cpu")
        for j, proj_vec in enumerate(proj):
            appr_mat[j, n] = inner_product_named_params(vec, proj_vec)
            vec = add_scalar_multiple(vec, -appr_mat[j, n], proj_vec)

        new_norm = torch.sqrt(inner_product_named_params(vec, vec))

        if new_norm < stop_tol:
            appr_mat[n+1, n] = 0
            vec = {n: torch.zeros_like(p) for n, p in vec.items()}
            proj.append(vec)
            break

        appr_mat[n+1, n] = new_norm
        vec = multiply_by_scalar_named_params(vec, 1.0/appr_mat[n+1, n])
        proj.append(vec)

    return appr_mat, proj


def change_basis_of_projections(matrix: torch.Tensor,
                                proj: list[dict[str, torch.Tensor]]):
    out = []
    for j in tqdm(range(matrix.shape[1]), desc="Change basis of projections"):
        element = {n: torch.zeros_like(p) for n, p in proj[0].items()}
        for i in range(matrix.shape[0]):
            element = add_scalar_multiple(element, matrix[i, j], proj[i])
        out.append(element)
    return out


def distill(appr_mat: torch.Tensor,
            proj: list[dict[str, torch.Tensor]],
            top_k: int):
    appr_mat = appr_mat[:-1, :]
    n = appr_mat.shape[0]

    for i in range(n):
        for j in range(n):
            if i-j > 1 or j-i > 1:
                appr_mat[i, j] = 0

    appr_mat = 0.5 * (appr_mat + appr_mat.T)
    eigvals, eigvecs = torch.linalg.eigh(appr_mat)
    idx = torch.argsort(torch.abs(eigvals))
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    reduced_projections = change_basis_of_projections(
        eigvecs[:, -top_k:], proj[:-1])
    return eigvals[-top_k:], reduced_projections


def get_projections(params: dict[str, torch.Tensor],
                    inputs: torch.Tensor,
                    targets: torch.Tensor,
                    eigvecs: list[dict[str, torch.Tensor]],
                    device: str,
                    do_mlp: bool):
    if do_mlp:
        def loss_fn(p): return utils.loss_f_blocks(model, p, inputs, targets)
    else:
        def loss_fn(p): return utils.loss_f_all(model, p, inputs, targets)
    def jvp_fn(v): return jvp(loss_fn, (params,), (v,))[1]
    projections = torch.empty((len(eigvecs),))
    for i, eigvec in enumerate(eigvecs):
        prj = jvp_fn(to_device_named_params(eigvec, device))
        projections[i] = prj.detach()
    return projections


def arnoldi_influence(params: dict[str, torch.Tensor],
                      eigvals: torch.Tensor,
                      eigvecs: list[dict[str, torch.Tensor]],
                      #   query_pair: tuple[torch.Tensor, torch.Tensor],
                      projections_test: torch.Tensor,
                      cand_pair: tuple[torch.Tensor, torch.Tensor],
                      device: str,
                      do_mlp: bool):
    """Influence of a candidate example on a query example."""
    projections_train = get_projections(params, cand_pair[0], cand_pair[1], eigvecs, device, do_mlp)
    projections_train_norm = torch.sqrt(torch.dot(projections_train, projections_train)) + 1e-10
    projections_train = projections_train / projections_train_norm / eigvals
    return torch.dot(projections_train, projections_test)


def example_grads_mlp(params, inputs, targets):
    loss_f_blocks = lambda params, inputs, targets: utils.loss_f_blocks(model, params, inputs, targets)
    return grad(loss_f_blocks)(params, inputs, targets)


def example_grads_all(params, inputs, targets):
    loss_f_all = lambda params, inputs, targets: utils.loss_f_all(model, params, inputs, targets)
    return grad(loss_f_all)(params, inputs, targets)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    config = AutoConfig.from_pretrained("config.json")
    config.use_hook = False
    model = GPTNeoXForCausalLM(config=config)
    model.load_state_dict(torch.load("model.ckpt", map_location=device))
    model = model.to(device)
    params = dict(model.named_parameters())
    mlp_params = {}
    for n, p in params.items():
        if "dense_h_to_4h" in n:
            mlp_params[n] = p

    print(sum([p.numel() for p in model.parameters()]))
    def count_mlp_params(model):
        n = 0
        for layer in model.gpt_neox.layers:
            n += sum([p.numel() for p in layer.mlp.dense_4h_to_h.parameters()])
        return n
    print(count_mlp_params(model))

    tokenized_ds = torch.load("tokenized_ds.pt")
    ds = load_dataset(dataset_path, trust_remote_code=True)

    out_f_blocks = lambda mlp_params, i: utils.out_f_blocks(model, params, mlp_params, i)
    out_f_all = lambda params, i: utils.out_f_all(model, params, i)

    if do_arnoldi_iter:
        train_loader = utils.get_dataloader(tokenized_ds['train'], batch_size, True)

        if use_hvp_estimator:
            hvp_step_fn = utils.hvp_on_sample(utils.loss_f_blocks)
            hvp_estimator = utils.create_hvp_estimator(hvp_step_fn, mlp_params, train_loader, device)
        else:
            if do_mlp:
                gnhvp_step_fn = utils.gnhvp_on_sample(out_f_blocks, utils.loss_with_logits)
                gnhvp_estimator = utils.create_gnhvp_estimator(gnhvp_step_fn, mlp_params, train_loader, device)
            else:
                gnhvp_step_fn = utils.gnhvp_on_sample(out_f_all, utils.loss_with_logits)
                gnhvp_estimator = utils.create_gnhvp_estimator(gnhvp_step_fn, params, train_loader, device)

        if do_mlp:
            start_vector = prepare_normal_start_vector(mlp_params)
        else:
            start_vector = prepare_normal_start_vector(params)

        start = time.time()
        if use_hvp_estimator:
            appr_mat, proj = arnoldi_iter(
                hvp_estimator, start_vector, n_iters, device)
        else:
            appr_mat, proj = arnoldi_iter(
                gnhvp_estimator, start_vector, n_iters, device)
        end = time.time()
        print(f"Time on arnoldi iterations: {end-start:.3f} s")

        start = time.time()
        eigvals, eigvecs = distill(appr_mat, proj, top_k)
        end = time.time()
        print(f"Time on distill: {end-start:.3f} s")

        torch.save(eigvals, "eigvals.pt")
        torch.save(eigvecs, "eigvecs.pt")
    else:
        eigvals = torch.load("eigvals.pt")
        eigvecs = torch.load("eigvecs.pt")

    with open("choices_candidates.pkl", "rb") as fp:
        choices_candidates = pickle.load(fp)

    with open("choices_queries.pkl", "rb") as fp:
        choices_queries = pickle.load(fp)

    os.makedirs(save_dir, exist_ok=True)

    start = time.time()
    for i, qi in enumerate(tqdm(choices_queries, desc="Influences")):
        text = ds['test'][qi]['sentence']
        ids = tokenizer(text, max_length=max_length, truncation=True, return_tensors='pt', return_attention_mask=False).input_ids.to(device)
        if do_mlp:
            projections_test = get_projections(mlp_params, ids[:, :-1], ids[:, 1:], eigvecs, device, do_mlp)
        else:
            projections_test = get_projections(params, ids[:, :-1], ids[:, 1:], eigvecs, device, do_mlp)

        infls = []
        for j, cj in enumerate(tqdm(choices_candidates[i], desc=f"Infl [{i}]", disable=silence_hint)):
            text = ds['train'][cj]['sentence']
            ids = tokenizer(text, max_length=max_length, truncation=True, return_tensors='pt', return_attention_mask=False).input_ids.to(device)
            if do_mlp:
                infl = arnoldi_influence(mlp_params, eigvals, eigvecs, projections_test, (ids[:, :-1], ids[:, 1:]), device, do_mlp)
            else:
                infl = arnoldi_influence(params, eigvals, eigvecs, projections_test, (ids[:, :-1], ids[:, 1:]), device, do_mlp)
            if torch.isnan(infl):
                infl = 0.0
            else:
                infl = infl.item()
            infls.append(infl)

        with open(os.path.join(save_dir, f"{i}.pkl"), "wb") as fp:
            pickle.dump(infls, fp)
            if not silence_hint:
                print(f"Saved to {fp.name}")
    end = time.time()
    print(f"Time on influences: {end-start:.3f} s")
