"""Fit EK-FAC parameters for pretrained model and fine-tuned model."""

import os
import pickle
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import (
    AutoTokenizer, AutoConfig, GPTNeoXForCausalLM
)
from datasets import load_from_disk, load_dataset, Dataset

from ekfac import ekfac_utils
import utils

from loguru import logger


device = "cuda:3"
device_backup = "cuda:1"
seed = 42
batch_size = 6
n_iters = 20_000
pretrain_max_length = 512
finetune_max_length = 512

small_trainset_path = "../small-trainset/"
small_trainset_tokenized_path = f"../small-trainset_tokenized_length={pretrain_max_length}/"
pile_path = "/home/Dataset/pile-uncopyrighted/"
dolly_path = "/home/Dataset/databricks-dolly-15k/"
dolly_tokenized_path = f"./dolly-tokenized-length={finetune_max_length}/"

param_count = ("2.8b", "3b")
# param_count = ("6.9b", "7b")
# param_count = ("12b", "12b")
pretrained_model_path = f"Model/Pythia/pythia-{param_count[0]}/"
finetuned_model_path = f"Model/Pythia/dolly-v2-{param_count[1]}/"

figures_dir = f"figures/dolly-{param_count[1]}/"
save_dir = f"factors/dolly-{param_count[1]}/"

do_pretrain = False
do_full_ft = True


def fit_pretrain(model: GPTNeoXForCausalLM,
                 modules_of_interest: dict[str, str],
                 tokenizer: AutoTokenizer,
                 data_loader: DataLoader,
                 n_iters: int,
                 max_length: int,
                 batch_size: int,
                 figures_dir: str):
    logger.info("Start fitting EK-FAC parameters for pretrained model on pretraining dataset.")
    figures_dir = os.path.join(figures_dir, "pretrain")
    QA, QG = ekfac_utils.ekfac_fit_covariance(model=model,
                                              device=device,
                                              device_backup=device_backup,
                                              modules_of_interest=modules_of_interest,
                                              dataloader=data_loader,
                                              n_iters=n_iters,
                                              batch_size=batch_size,
                                              max_length=max_length,
                                              hidden_size=model.config.hidden_size,
                                              n_heads=model.config.num_attention_heads,
                                              do_log=True,
                                              figures_dir=figures_dir)
    Lambda = ekfac_utils.ekfac_fit_diagonal(model=model,
                                            device=device,
                                            QA=QA,
                                            QG=QG,
                                            modules_of_interest=modules_of_interest,
                                            dataloader=data_loader,
                                            n_iters=n_iters,
                                            batch_size=batch_size,
                                            max_length=max_length,
                                            hidden_size=model.config.hidden_size,
                                            n_heads=model.config.num_attention_heads,
                                            do_log=True,
                                            figures_dir=figures_dir)
    return QA, QG, Lambda


def fit_full_ft(model: GPTNeoXForCausalLM,
                modules_of_interest: dict[str, str],
                tokenizer: AutoTokenizer,
                data_loader: DataLoader,
                n_iters: int,
                max_length: int,
                batch_size: int,
                figures_dir: str):
    logger.info("Start fitting EK-FAC parameters for finetuned model on finetuning dataset.")
    figures_dir = os.path.join(figures_dir, "finetune-full")
    QA, QG = ekfac_utils.ekfac_fit_covariance(model=model,
                                              device=device,
                                              device_backup=device_backup,
                                              modules_of_interest=modules_of_interest,
                                              dataloader=data_loader,
                                              n_iters=n_iters,
                                              batch_size=batch_size,
                                              max_length=max_length,
                                              hidden_size=model.config.hidden_size,
                                              n_heads=model.config.num_attention_heads,
                                              do_log=True,
                                              figures_dir=figures_dir)
    Lambda = ekfac_utils.ekfac_fit_diagonal(model=model,
                                            device=device,
                                            QA=QA,
                                            QG=QG,
                                            modules_of_interest=modules_of_interest,
                                            dataloader=data_loader,
                                            n_iters=n_iters,
                                            batch_size=batch_size,
                                            max_length=max_length,
                                            hidden_size=model.config.hidden_size,
                                            n_heads=model.config.num_attention_heads,
                                            do_log=True,
                                            figures_dir=figures_dir)
    return QA, QG, Lambda


def pretrain_collator(batch):
    input_ids = torch.stack([torch.tensor(x['input_ids']) for x in batch])
    # labels = input_ids.clone()
    # return input_ids, labels
    return input_ids


def full_ft_collator(batch):
    input_ids = torch.stack([torch.tensor(x['input_ids']) for x in batch])
    # labels = utils.dolly_get_labels(input_ids, tokenizer)
    # return input_ids, labels
    return input_ids


if __name__ == "__main__":
    if do_pretrain:
        pretrain_dataset_tokenized = load_from_disk(small_trainset_tokenized_path)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        config = AutoConfig.from_pretrained(pretrained_model_path)
        config.use_hook = True
        model = GPTNeoXForCausalLM.from_pretrained(pretrained_model_path,
                                                   config=config,
                                                   device_map=device,
                                                   torch_dtype=torch.bfloat16)
        
        pretrain_loader = DataLoader(dataset=pretrain_dataset_tokenized,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     collate_fn=pretrain_collator)
        
        modules_of_interest = {}
        for i in range(config.num_hidden_layers):
            modules_of_interest[f"gpt_neox.layers.{i}.attention.query_key_value"] = "attention"
            modules_of_interest[f"gpt_neox.layers.{i}.attention.dense"] = "mlp"
            modules_of_interest[f"gpt_neox.layers.{i}.mlp.dense_h_to_4h"] = "mlp"
            modules_of_interest[f"gpt_neox.layers.{i}.mlp.dense_4h_to_h"] = "mlp"

        QA, QG, Lambda = fit_pretrain(model=model,
                                      modules_of_interest=modules_of_interest,
                                      tokenizer=tokenizer,
                                      data_loader=pretrain_loader,
                                      n_iters=n_iters,
                                      max_length=pretrain_max_length,
                                      batch_size=batch_size,
                                      figures_dir=figures_dir)

        pretrain_save_dir = os.path.join(save_dir, "pretrain")
        os.makedirs(pretrain_save_dir, exist_ok=True)
        torch.save(QA, os.path.join(pretrain_save_dir, "QA.pt"))
        torch.save(QG, os.path.join(pretrain_save_dir, "QG.pt"))
        torch.save(Lambda, os.path.join(pretrain_save_dir, "Lambda.pt"))


    if do_full_ft:
        tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
        full_ft_dataset_tokenized = load_from_disk(dolly_tokenized_path)
        finetune_loader = DataLoader(dataset=full_ft_dataset_tokenized,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     collate_fn=full_ft_collator,
                                     drop_last=True,)
        config = AutoConfig.from_pretrained(finetuned_model_path)
        config.use_hook = True
        model = GPTNeoXForCausalLM.from_pretrained(finetuned_model_path,
                                                   config=config,
                                                   device_map=device,
                                                   torch_dtype=torch.bfloat16)
        
        modules_of_interest = {}
        for i in range(config.num_hidden_layers):
            modules_of_interest[f"gpt_neox.layers.{i}.attention.query_key_value"] = "attention"
            modules_of_interest[f"gpt_neox.layers.{i}.attention.dense"] = "mlp"
            modules_of_interest[f"gpt_neox.layers.{i}.mlp.dense_h_to_4h"] = "mlp"
            modules_of_interest[f"gpt_neox.layers.{i}.mlp.dense_4h_to_h"] = "mlp"

        QA, QG, Lambda = fit_full_ft(model=model,
                                     modules_of_interest=modules_of_interest,
                                     tokenizer=tokenizer,
                                     data_loader=finetune_loader,
                                     n_iters=n_iters,
                                     max_length=finetune_max_length,
                                     batch_size=batch_size,
                                     figures_dir=figures_dir)
        finetune_save_dir = os.path.join(save_dir, "finetune-full")
        os.makedirs(finetune_save_dir, exist_ok=True)
        torch.save(QA, os.path.join(finetune_save_dir, "QA.pt"))
        torch.save(QG, os.path.join(finetune_save_dir, "QG.pt"))
        torch.save(Lambda, os.path.join(finetune_save_dir, "Lambda.pt"))
