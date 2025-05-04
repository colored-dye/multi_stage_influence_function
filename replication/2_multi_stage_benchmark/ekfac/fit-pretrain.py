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
device_backup = "cuda:3"
model_path = "/home/Model/BLOOM/bloom-560m"

BATCH_SIZE = 2
MAX_LENGTH = 1024
N_ITERS = 1000
DAMPING = 1e-4

figures_dir = "figures_pretrain/"
factors_dir = "factors_pretrain/"


def collate_fn(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch])
    return input_ids


def tokenize(examples, tokenizer):
    return tokenizer(examples['text'], return_attention_mask=False)


def group_texts(examples, max_length, pad_token_id):
    # Concatenate all texts.
    concatenated_examples = {
        k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    n_batches = total_length / max_length
    n_batches = n_batches.__ceil__()
    result = {
        k: [t[i*max_length: (i+1)*max_length]
            for i in range(n_batches)]
        for k, t in concatenated_examples.items()
    }
    if len(result['input_ids'][-1]) < max_length:
        result['input_ids'][-1] += [pad_token_id]*(max_length-len(result['input_ids'][-1]))

    input_ids = torch.tensor(result["input_ids"])
    result["input_ids"] = input_ids
    return result


def process_state_dict(state_dict):
    res = {}
    for n, p in state_dict.items():
        n = n[len("model."):]
        res[n] = p
    return res


def main():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("/home/Dataset/bigscience-data/roots_en_wikipedia", split='train')
    preprocess_fn = partial(tokenize, tokenizer=tokenizer)
    ds_tokenized = ds.map(preprocess_fn,
                          desc="Tokenize",
                          batched=True,
                          batch_size=1000,
                          num_proc=32,
                          remove_columns=ds.column_names)
    group_text_fn = partial(group_texts,
                            max_length=MAX_LENGTH,
                            pad_token_id=tokenizer.pad_token_id)
    ds_tokenized = ds_tokenized.map(group_text_fn,
                                    desc="Group texts",
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
    #                                           dtype=torch.float32)
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
                                            dtype=torch.float32)
    torch.save(Lambda, os.path.join(factors_dir, "Lambda.pt"))
    logger.info("All done.")


if __name__ == "__main__":
    main()
