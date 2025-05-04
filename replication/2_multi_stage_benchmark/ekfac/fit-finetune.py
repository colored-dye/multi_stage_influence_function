from functools import partial
import os
from loguru import logger

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig, AutoTokenizer, BloomForCausalLM,
)
from datasets import load_dataset

import ekfac_utils


device = "cuda:1"
device_backup = "cuda:2"
model_path = "/home/Model/BLOOM/bloomz-560m"

BATCH_SIZE = 1
MAX_LENGTH = 1024
N_ITERS = 1000
DAMPING = 1e-4

figures_dir = "figures_finetune/"
factors_dir = "factors_finetune/"


def collate_fn(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    labels = torch.stack([x['labels'] for x in batch])
    return input_ids, attention_mask, labels


def preprocess(examples, tokenizer, max_length):
    prompt_ids = tokenizer(examples['inputs'],
                           max_length=max_length,
                           truncation=True,
                           padding=False,
                           return_attention_mask=False)['input_ids']
    targets = [" " + t for t in examples['targets']]
    target_ids = tokenizer(targets,
                           max_length=max_length,
                           truncation=True,
                           padding=False,
                           return_attention_mask=False)['input_ids']
    attention_mask = [[1] * len(p) + [0] * (max_length-len(p)) for p in prompt_ids]
    ids = [p + t + [tokenizer.pad_token_id]*(max_length-len(p)-len(t))
       for p, t in zip(prompt_ids, target_ids)]
    ids = torch.tensor(ids)
    labels = ids.clone()
    for i in range(labels.shape[0]):
        labels[i, :len(prompt_ids[i])] = -100
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        "input_ids": ids.tolist(),
        "attention_mask": attention_mask,
        "labels": labels.tolist(),
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    ds = load_dataset("/home/Dataset/xP3-en", split=f'train[:{BATCH_SIZE*N_ITERS}]', data_files=["merged_en.jsonl"])
    ds_tokenized = ds.map(partial(preprocess, tokenizer=tokenizer, max_length=MAX_LENGTH),
                          batched=True,
                          num_proc=32)
    ds_tokenized = ds_tokenized.with_format("torch")
    
    model = BloomForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    modules_of_interest = {}
    for i in range(model.config.num_hidden_layers):
        modules_of_interest[f'transformer.h.{i}.self_attention.query_key_value'] = 'mha'
        modules_of_interest[f'transformer.h.{i}.self_attention.dense'] = 'mlp'
        modules_of_interest[f'transformer.h.{i}.mlp.dense_h_to_4h'] = 'mlp'
        modules_of_interest[f'transformer.h.{i}.mlp.dense_4h_to_h'] = 'mlp'

    # train_loader = DataLoader(ds_tokenized,
    #                           batch_size=BATCH_SIZE,
    #                           collate_fn=collate_fn)
    # QA, QG = ekfac_utils.ekfac_fit_covariance(model=model,
    #                                           device=device,
    #                                           device_backup=device_backup,
    #                                           modules_of_interest=modules_of_interest,
    #                                           dataloader=train_loader,
    #                                           n_iters=N_ITERS,
    #                                           batch_size=BATCH_SIZE,
    #                                           max_length=MAX_LENGTH,
    #                                           hidden_size=model.config.hidden_size,
    #                                           n_heads=model.config.num_attention_heads,
    #                                           do_log=True,
    #                                           figures_dir=figures_dir,
    #                                           dtype=torch.float32,
    #                                           do_finetune=True)
    # os.makedirs(factors_dir, exist_ok=True)
    # torch.save(QA, os.path.join(factors_dir, "QA.pt"))
    # torch.save(QG, os.path.join(factors_dir, "QG.pt"))

    QA = torch.load(os.path.join(factors_dir, "QA.pt"), map_location=device)
    QG = torch.load(os.path.join(factors_dir, "QG.pt"), map_location=device)

    train_loader = DataLoader(ds_tokenized,
                              batch_size=BATCH_SIZE,
                              collate_fn=collate_fn)
    Lambda = ekfac_utils.ekfac_fit_diagonal(model=model,
                                            device=device,
                                            QA=QA,
                                            QG=QG,
                                            modules_of_interest=modules_of_interest,
                                            dataloader=train_loader,
                                            n_iters=N_ITERS,
                                            batch_size=BATCH_SIZE,
                                            max_length=MAX_LENGTH,
                                            hidden_size=model.config.hidden_size,
                                            n_heads=model.config.num_attention_heads,
                                            do_log=True,
                                            figures_dir=figures_dir,
                                            dtype=torch.float32,
                                            do_finetune=True)
    torch.save(Lambda, os.path.join(factors_dir, "Lambda.pt"))
    logger.info("All done.")


if __name__ == "__main__":
    main()
