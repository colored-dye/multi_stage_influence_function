import os
import pickle
import time
import copy
import functools
from tqdm import tqdm

import torch
from torch.func import grad, functional_call
import torch.nn.functional as F
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
max_length = 512
do_mlp = False

if do_mlp:
    save_dir = "infls/gdp-mlp"
else:
    save_dir = "infls/gdp-all"


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
            query_grads = example_grads_mlp(mlp_params, ids[:, :-1], ids[:, 1:])
        else:
            query_grads = example_grads_all(params, ids[:, :-1], ids[:, 1:])

        infls = []
        for j, cj in enumerate(tqdm(choices_candidates[i], desc=f"Infl [{i}]", disable=True)):
            text = ds['train'][cj]['sentence']
            ids = tokenizer(text, max_length=max_length, truncation=True, return_tensors='pt', return_attention_mask=False).input_ids.to(device)
            if do_mlp:
                cand_grads = example_grads_mlp(mlp_params, ids[:, :-1], ids[:, 1:])
            else:
                cand_grads = example_grads_all(params, ids[:, :-1], ids[:, 1:])

            infl = torch.dot(utils.ravel_named_params(query_grads)[0], utils.ravel_named_params(cand_grads)[0])
            infls.append(infl.item())
        
        with open(os.path.join(save_dir, f"{i}.pkl"), "wb") as fp:
            pickle.dump(infls, fp)
            # print(f"Saved to {fp.name}")
    end = time.time()
    print(f"Time used: {end-start:.3f} s")
