import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import string
import numpy as np
from tqdm import tqdm
from functools import partial


def tokenize_function(examples, tokenizer):
    return {"input_ids": tokenizer.encode(examples["sentence"]).ids}

def group_texts(examples, max_length):
    # Concatenate all texts.
    concatenated_examples = {
        k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + max_length]
            for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }

    input_ids = torch.tensor(result["input_ids"])
    result["input_ids"] = input_ids
    return result

def tokenize_and_shard(dataset, tokenizer, max_length):
    tokenized_dataset = dataset.map(
        partial(tokenize_function, tokenizer=tokenizer),
        remove_columns=dataset.column_names,
    )
    tokenized_dataset = tokenized_dataset.map(
        partial(group_texts, max_length=max_length),
        batched=True,
        batch_size=200,
    )
    return tokenized_dataset


def get_tokenized_dataset(ds, tokenizer, max_length):
    return tokenize_and_shard(ds, tokenizer, max_length)

def collate_fn(batch):
    return torch.stack([x[0] for x in batch])

def get_dataloader(tokenized_split, batch_size, shuffle):
    ds = TensorDataset(tokenized_split['input_ids'])
    loader = DataLoader(ds, batch_size, shuffle, collate_fn=collate_fn)
    return loader


"""
"""


from torch.func import functional_call, jvp, grad, vjp
import torch.nn.functional as F
import einops


def loss_with_logits(logits, targets):
    targets = targets.reshape(-1)
    loss = F.cross_entropy(logits, targets)
    return loss


def out_f_blocks(model, params, mlp_params, i):
    for n, p in mlp_params.items():
        params[n] = p
    
    logits = functional_call(model, params, (i,)).logits
    logits = logits.reshape(-1, logits.size(-1))
    return logits


def loss_f_blocks(model, mlp_params, i, t):
    logits = out_f_blocks(model, mlp_params, i)
    loss = loss_with_logits(logits, t)
    return loss


def out_f_all(model, params, i):
    out = functional_call(model, params, (i,)).logits
    out = einops.rearrange(out, "b s v -> (b s) v")
    return out


def loss_f_all(model, params, i, t):
    logits = out_f_all(model, params, i)
    loss = loss_with_logits(logits, t)
    return loss


def hvp_fn(loss_fn, params, v):
    return jvp(grad(loss_fn), (params,), (v,))[1]


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


def sample_labels(logits: torch.Tensor):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    if len(probs.shape) > 1:
        return probs.multinomial(num_samples=1, replacement=True)[:, 0]
    else:
        return probs.multinomial(num_samples=1, replacement=True)[0]


def gnhvp_on_sample(f, L):
    def gnhvp_step(primals, tangents, inputs):
        f_ = lambda p: f(p, inputs)
        z, R_z = jvp(f_, (primals,), (tangents,))

        sampled_labels = sample_labels(z)
        L_ = lambda y: L(y, sampled_labels)

        R_gz = hvp_fn(L_, z, R_z)
        _, f_vjp = vjp(f_, primals)
        return f_vjp(R_gz)[0]
    return gnhvp_step


def create_gnhvp_estimator(gnhvp_step_fn, parameters, data_loader, device):
    ids = next(iter(data_loader))
    ids = ids.to(device)
    
    def compute_fn(vec):
        return gnhvp_step_fn(parameters, vec, ids)
    return compute_fn


def hvp_on_sample(L):
    def hvp_step(primals, tangents, inputs):
        def L_(p): return L(p, inputs[:, :-1], inputs[:, 1:])
        return hvp_fn(L_, primals, tangents)
    return hvp_step


def create_hvp_estimator(hvp_step_fn, parameters, data_loader, device):
    ids = next(iter(data_loader))
    ids = ids.to(device)

    def compute_fn(vec):
        return hvp_step_fn(parameters, vec, ids)
    return compute_fn

