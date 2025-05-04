import os
import pickle
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn.functional as F
import einops

from modeling_gpt_neox import GPTNeoXForCausalLM
from transformers import (
    AutoTokenizer, AutoConfig
)
from datasets import load_dataset

from utils import get_dataloader


pretrained_model_path = "/home/Model/Pythia/pythia-70m"
dataset_path = "/home/Dataset/ptb_text_only"

device = "cuda:3"
seed = 42
batch_size = 1
n_iters = 10000
damping = 1e-4
max_length = 512

save_dir = "infls/ekfac/"
figures_dir = "figures/ekfac/"

do_cov = True
do_diag = True

silence_hint = True


def sample_labels(logits: torch.Tensor):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    if len(probs.shape) > 1:
        return probs.multinomial(num_samples=1, replacement=True)[:, 0]
    else:
        return probs.multinomial(num_samples=1, replacement=True)[0]


def vectorize(x: torch.Tensor):
    return x.t().reshape(-1)


def unvectorize(x: torch.Tensor, rows, cols):
    return x.reshape(cols, rows).t()


def ekfac_fit_covariance(model: GPTNeoXForCausalLM,
                         device: str,
                         dataloader: DataLoader,
                         n_iters: int,
                         figures_dir: str):
    n_blocks = len(model.gpt_neox.layers)
    batch_size = dataloader.batch_size
    
    A = [] # Covariance of inputs
    G = [] # Covariance of preactivation pseudo-gradients
    for block in model.gpt_neox.layers:
        mlp = block.mlp.dense_h_to_4h
        out_dim, in_dim = mlp.weight.shape
        A.append(torch.zeros((in_dim+1, in_dim+1), device=device))
        G.append(torch.zeros((out_dim, out_dim), device=device))

    logs = [
        {
            "A": [],
            "G": [],
        } for _ in range(n_blocks)
    ]
    for i in tqdm(range(n_iters), desc="Fitting covariance matrices A&G"):
        ids = next(iter(dataloader))
        ids = ids.to(device)

        model.zero_grad()
        logits = model(ids).logits
        logits = logits.reshape(-1, logits.size(-1))
        sampled_labels = sample_labels(logits)
        loss = F.cross_entropy(logits, sampled_labels, reduction="sum")
        loss.backward()

        for block_idx, block in enumerate(model.gpt_neox.layers):
            mlp = block.mlp
            a_l_minus_1 = mlp.get_a_l_minus_1()
            d_s_l = mlp.get_d_s_l()

            a_l_minus_1 = a_l_minus_1.reshape(-1, a_l_minus_1.size(-1))
            d_s_l = d_s_l.reshape(-1, d_s_l.size(-1))
            
            A[block_idx] += a_l_minus_1.T @ a_l_minus_1
            G[block_idx] += d_s_l.T @ d_s_l
            
            logs[block_idx]["A"].append(torch.linalg.norm(A[block_idx]/((i+1)*batch_size)).detach().cpu())
            logs[block_idx]["G"].append(torch.linalg.norm(G[block_idx]/((i+1)*batch_size)).detach().cpu())
    
    plt.figure()
    fig, axes = plt.subplots(2, n_blocks, figsize=(12,8))
    for i in range(n_blocks):
        axes[0, i].plot(range(n_iters), logs[i]["A"])
        axes[0, i].set_title(f"A[{i}]")
        axes[1, i].plot(range(n_iters), logs[i]["G"])
        axes[1, i].set_title(f"G[{i}]")
    os.makedirs(figures_dir, exist_ok=True)
    figname = os.path.join(figures_dir, "ekfac-cov.png")
    if os.path.exists(figname):
        os.remove(figname)
    plt.savefig(figname)

    QA = []
    QG = []
    for block_idx in range(n_blocks):
        A[block_idx] /= n_iters*batch_size
        G[block_idx] /= n_iters*batch_size

        _, qa = torch.linalg.eigh(A[block_idx])
        _, qg = torch.linalg.eigh(G[block_idx])
        QA.append(qa.detach())
        QG.append(qg.detach())
    return QA, QG


def ekfac_fit_diagonal(model: GPTNeoXForCausalLM,
                       device: str,
                       dataloader: DataLoader,
                       n_iters: int,
                       QA: list[torch.Tensor],
                       QG: list[torch.Tensor]):
    n_blocks = len(model.gpt_neox.layers)
    batch_size = dataloader.batch_size

    Lambda = []
    for block in model.gpt_neox.layers:
        mlp = block.mlp.dense_h_to_4h
        out_dim, in_dim = mlp.weight.shape
        Lambda.append(torch.zeros(((in_dim+1)*out_dim), device=device))
    
    for i in tqdm(range(n_iters), desc="Fitting diagonal"):
        ids = next(iter(dataloader))
        ids = ids.to(device)

        model.zero_grad()
        logits = model(ids).logits
        logits = logits.reshape(-1, logits.size(-1))
        sampled_labels = sample_labels(logits)
        loss = F.cross_entropy(logits, sampled_labels, reduction="sum")
        loss.backward()

        for block_idx, block in enumerate(model.gpt_neox.layers):
            mlp = block.mlp
            dw = mlp.get_d_w_l()
            result = QG[block_idx].T @ dw @ QA[block_idx]
            result = vectorize(result)
            Lambda[block_idx] += result.pow(2)

    for i in range(n_blocks):
        Lambda[i] /= n_iters*batch_size
    return Lambda


def example_grads(net: GPTNeoXForCausalLM,
                  inputs: torch.Tensor,
                  targets: torch.Tensor):
    net.zero_grad()
    logits = net(inputs).logits
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    grads = []
    for block in net.gpt_neox.layers:
        mlp = block.mlp
        grads.append(mlp.get_d_w_l())
    return grads


def ekfac_ihvp_single_block(qa: torch.Tensor,
                            qg: torch.Tensor,
                            diagonal: torch.Tensor,
                            damping: float,
                            v: torch.Tensor):
    qg_v_qa = qg.T @ v @ qa
    diagonal += damping
    diagonal = unvectorize(diagonal, v.shape[0], v.shape[1])
    result = qg_v_qa / diagonal
    ihvp = qg @ result @ qa.T
    return ihvp


def ekfac_ihvp(QA: list[torch.Tensor],
               QG: list[torch.Tensor],
               Lambda: list[torch.Tensor],
               damping: float,
               vec: list[torch.Tensor]):
    ihvps = []
    for qa, qg, diagonal, v in zip(QA, QG, Lambda, vec):
        ihvp = ekfac_ihvp_single_block(qa, qg, diagonal, damping, v)
        ihvp = vectorize(ihvp)
        ihvps.append(ihvp)
    return torch.cat(ihvps)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    config = AutoConfig.from_pretrained("config.json")
    config.use_hook = True
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

    if do_cov:
        train_loader = get_dataloader(tokenized_ds['train'], batch_size, True)
        start = time.time()
        QA, QG = ekfac_fit_covariance(model, device, train_loader, n_iters, figures_dir)
        end = time.time()
        print(f"Time on fitting covariance matrices: {end-start:.3f} s")
        torch.save(QA, "QA.pt")
        torch.save(QG, "QG.pt")
    else:
        QA = torch.load("QA.pt", map_location=device)
        QG = torch.load("QG.pt", map_location=device)

    if do_diag:
        train_loader = get_dataloader(tokenized_ds['train'], batch_size, True)
        start = time.time()
        Lambda = ekfac_fit_diagonal(model, device, train_loader, n_iters, QA, QG)
        end = time.time()
        print(f"Time on fitting diagonal matrix: {end-start:.3f} s")
        torch.save(Lambda, "Lambda.pt")
    else:
        Lambda = torch.load("Lambda.pt", map_location=device)
    
    with open("choices_candidates.pkl", "rb") as fp:
        choices_candidates = pickle.load(fp)
    
    with open("choices_queries.pkl", "rb") as fp:
        choices_queries = pickle.load(fp)
    
    os.makedirs(save_dir, exist_ok=True)
    start = time.time()
    for i, qi in enumerate(tqdm(choices_queries, desc="Influences")):
        text = ds['test'][qi]['sentence']
        ids = tokenizer(text, max_length=max_length, truncation=True, return_tensors='pt', return_attention_mask=False).input_ids.to(device)
        query_grads = example_grads(model, ids[:, :-1], ids[:, 1:])
        ihvp = ekfac_ihvp(QA, QG, Lambda, damping, query_grads)

        infls = []
        for j, cj in enumerate(tqdm(choices_candidates[i], desc=f"Infl [{i}]", disable=silence_hint)):
            text = ds['train'][cj]['sentence']
            ids = tokenizer(text, max_length=max_length, truncation=True, return_tensors='pt', return_attention_mask=False).input_ids.to(device)
            cand_grads = example_grads(model, ids[:, :-1], ids[:, 1:])
            cand_grads_flattened = []
            for cg in cand_grads:
                cand_grads_flattened.append(vectorize(cg))
            cand_grads_flattened = torch.cat(cand_grads_flattened)

            infl = torch.dot(ihvp, cand_grads_flattened)
            infls.append(infl.item())
        
        with open(os.path.join(save_dir, f"{i}.pkl"), "wb") as fp:
            pickle.dump(infls, fp)
            if not silence_hint:
                print(f"Saved to {fp.name}")
    end = time.time()
    print(f"Time on influence: {end-start:.3f} s")
