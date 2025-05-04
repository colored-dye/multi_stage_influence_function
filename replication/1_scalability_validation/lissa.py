import os
from typing import Callable
from tqdm import tqdm
import time
import pickle
import matplotlib.pyplot as plt

import torch
from torch.func import functional_call, jvp, grad, vjp
import torch.nn.functional as F
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

device = "cuda:3"
seed = 42
n_iters = 500
damping = 1e-4
alpha = 1/1e3
batch_size = 32
max_length = 512

do_mlp = False

if do_mlp:
    save_dir = f"infls/lissa-mlp-iters={n_iters}"
    figures_dir = f"figures/lissa-mlp-iters={n_iters}"
else:
    save_dir = f"infls/lissa-all-iters={n_iters}"
    figures_dir = f"figures/lissa-all-iters={n_iters}"


def lissa_ignhvp(mvp: Callable,
          vec,
          n_iters: int,
          damping: float,
          alpha: float):
    
    vec_flattened, unravel_fn = utils.ravel_named_params(vec)
    ihvp = vec_flattened.clone().detach()

    logs = []
    for i in tqdm(range(n_iters), desc="LiSSA iterations"):
        Ap = mvp(unravel_fn(ihvp))
        Ap = utils.ravel_named_params(Ap)[0].detach()
        ihvp_new = vec_flattened + (1-damping*alpha)*ihvp - alpha*Ap
        ihvp_update = torch.linalg.vector_norm(ihvp_new-ihvp)
        logs.append(ihvp_update.item())

        ihvp = ihvp_new

    return ihvp, logs


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

    train_loader = utils.get_dataloader(tokenized_ds['train'], batch_size, True)

    if do_mlp:
        gnhvp_step_fn = utils.gnhvp_on_sample(out_f_blocks, utils.loss_with_logits)
        gnhvp_estimator = utils.create_gnhvp_estimator(gnhvp_step_fn, mlp_params, train_loader, device)
    else:
        gnhvp_step_fn = utils.gnhvp_on_sample(out_f_all, utils.loss_with_logits)
        gnhvp_estimator = utils.create_gnhvp_estimator(gnhvp_step_fn, params, train_loader, device)

    with open("choices_candidates.pkl", "rb") as fp:
        choices_candidates = pickle.load(fp)
    
    with open("choices_queries.pkl", "rb") as fp:
        choices_queries = pickle.load(fp)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    first = True
    start = time.time()
    for i, qi in enumerate(tqdm(choices_queries, desc="Influences")):
        text = ds['test'][qi]['sentence']
        ids = tokenizer(text, max_length=max_length, truncation=True, return_tensors='pt', return_attention_mask=False).input_ids.to(device)
        if do_mlp:
            query_grads = example_grads_mlp(mlp_params, ids[:, :-1], ids[:, 1:])
        else:
            query_grads = example_grads_all(params, ids[:, :-1], ids[:, 1:])
        ignhvp, logs = lissa_ignhvp(gnhvp_estimator, query_grads, n_iters, damping, alpha)
        if first:
            plt.figure()
            plt.plot(range(len(logs)), logs)
            plt.title(f"ignhvp alpha={alpha}, damping={damping}, iteration={n_iters}")
            filename = f"{figures_dir}/ignhvp-alpha={alpha}-damping={damping}-iteration={n_iters}.png"
            if os.path.exists(filename):
                os.remove(filename)
            plt.savefig(filename)
            plt.close()
            first = False

        infls = []
        for j, cj in enumerate(tqdm(choices_candidates[i], desc=f"Infl [{i}]")):
            text = ds['train'][cj]['sentence']
            ids = tokenizer(text, max_length=max_length, truncation=True, return_tensors='pt', return_attention_mask=False).input_ids.to(device)
            if do_mlp:
                cand_grads = example_grads_mlp(mlp_params, ids[:, :-1], ids[:, 1:])
            else:
                cand_grads = example_grads_all(params, ids[:, :-1], ids[:, 1:])
            infl = torch.dot(utils.ravel_named_params(cand_grads)[0], ignhvp)
            infls.append(infl.item())
        
        with open(os.path.join(save_dir, f"{i}.pkl"), "wb") as fp:
            pickle.dump(infls, fp)
            print(f"Saved to {fp.name}")
    end = time.time()
    print(f"Time used: {end-start:.3f} s")

