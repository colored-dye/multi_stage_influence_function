import os
import pickle
import time
import json
import joblib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import (
    AutoTokenizer, AutoConfig
)
from datasets import load_from_disk

from sklearn.neighbors import NearestNeighbors

from modeling_gpt_neox import GPTNeoXForCausalLM
import utils

from sentence_transformers import SentenceTransformer

from loguru import logger


device = "cuda:2"
seed = 42
max_length = 2048
pretrain_damping = 1e-4
finetune_damping = 2e-4

pile_path = "/home/Dataset/pile-wikipedia-20200301-en.pkl"

n_candidates = 100

param_count = ("2.8b", "3b")
# param_count = ("6.9b", "7b")
# param_count = ("12b", "12b")
pretrained_model_path = f"Model/Pythia/pythia-{param_count[0]}/"
finetuned_model_path = f"Model/Pythia/dolly-v2-{param_count[1]}/"

# queries_path = "toxic.json"
# knn_candidates_path = "candidates-toxic.json"

queries_path = "qa.json"
knn_candidates_path = "candidates-qa.json"

# knn_path = "knn.joblib"
knn_path = "knn-wikipedia.joblib"

ekfac_factors_dir = f"factors/dolly-{param_count[1]}/"
infls_dir = f"infls/dolly-{param_count[1]}/full-qa/"


def load_ekfac_factors(subdir):
    ekfac_factors = {}
    ekfac_factors_subdir = os.path.join(ekfac_factors_dir, subdir)
    factors = ["QA", "QG", "Lambda"]
    for factor in factors:
        ekfac_factors[factor] = torch.load(os.path.join(ekfac_factors_subdir, f"{factor}.pt"), map_location=device)
    return ekfac_factors


def get_candidates_with_knn(knn: NearestNeighbors, queries: list[str], encoder: SentenceTransformer):
    query_embeddings = encoder.encode(queries)
    logger.info("Finding candidates...")
    candidate_indices = knn.kneighbors(query_embeddings, n_neighbors=n_candidates, return_distance=False)
    candidate_indices = [idx.tolist() for idx in candidate_indices]
    return candidate_indices


def main():
    with open(queries_path, "r") as fp:
        queries = json.load(fp)
    
    queries = [utils.dolly_format_with_response(rec) for rec in queries]

    logger.info("Loading KNN...")
    knn = joblib.load(knn_path)
    
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device, cache_folder="sentence_transformers-cache/")
    candidate_indices = get_candidates_with_knn(knn, queries, encoder)
    del knn, encoder
    
    with open(knn_candidates_path, "w") as fp:
        json.dump(candidate_indices, fp)
        logger.info(f"Candidates saved to {fp.name}")

    logger.info("Candidates ready. Load from disk.")
    with open(knn_candidates_path, "r") as fp:
        candidate_indices = json.load(fp)
    
    logger.info("Loading The Pile...")
    # pile_dataset = utils.load_partial_pile(pile_path, 0, 0, 100)
    pile_dataset = joblib.load(pile_path)

    logger.info("Loading EK-FAC factors...")
    ekfac_factors_subdirs = ["pretrain", "finetune-full"]
    ekfac_factors = {}
    for subdir in ekfac_factors_subdirs:
        ekfac_factors[subdir] = load_ekfac_factors(subdir)
        for n, p in ekfac_factors[subdir].items():
            ekfac_factors[subdir][n] = [pi.bfloat16() for pi in p]

    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)

    finetuned_config = AutoConfig.from_pretrained(finetuned_model_path)
    finetuned_config.use_hook = False
    finetuned_model = GPTNeoXForCausalLM.from_pretrained(finetuned_model_path,
                                                         config=finetuned_config,
                                                         device_map=device,
                                                         torch_dtype=torch.bfloat16)

    logger.info("Preparing query gradients...")
    tox_grads = []
    for tox_index, tox_text in enumerate(queries):
        tox_ids = tokenizer.encode(tox_text, max_length=max_length, truncation=True, padding=False, return_tensors='pt')
        tox_labels = utils.dolly_get_labels(tox_ids, tokenizer)
        tox_labels = tox_ids.clone()
        tox_ids = tox_ids[:, :-1].to(device)
        tox_labels = tox_labels[:, 1:].to(device)
        tox_grad = utils.example_grads(finetuned_model, tox_ids, tox_labels)
        tox_grads.append(tox_grad)
    del finetuned_model

    pretrained_config = AutoConfig.from_pretrained(pretrained_model_path)
    pretrained_config.use_hook = False
    pretrained_model = GPTNeoXForCausalLM.from_pretrained(pretrained_model_path,
                                                          config=pretrained_config,
                                                          device_map=device,
                                                          torch_dtype=torch.bfloat16)

    logger.info("Computing influences...")
    os.makedirs(infls_dir, exist_ok=True)
    for tox_index, tox_grad in enumerate(tqdm(tox_grads, desc="Queries")):
        ihvp_tox = utils.ekfac_ihvp(QA=ekfac_factors["finetune-full"]["QA"],
                                    QG=ekfac_factors["finetune-full"]["QG"],
                                    Lambda=ekfac_factors["finetune-full"]["Lambda"],
                                    damping=finetune_damping,
                                    vec=tox_grad)
        infls = []
        for candidate_index in tqdm(candidate_indices[tox_index], desc=f"Query [{tox_index}]"):
            # candidate_text = pile_dataset[candidate_index]["text"]
            candidate_text = pile_dataset[candidate_index]["title"] + "\n\n" + pile_dataset[candidate_index]["text"]
            cand_ids = tokenizer.encode(candidate_text, max_length=512, truncation=True, padding=False, return_tensors='pt')
            cand_labels = cand_ids[:, 1:].to(device)
            cand_ids = cand_ids[:, :-1].to(device)
            cand_grad = utils.example_grads(pretrained_model, cand_ids, cand_labels)
            ihvp_cand = utils.ekfac_ihvp(QA=ekfac_factors["pretrain"]["QA"],
                                         QG=ekfac_factors["pretrain"]["QG"],
                                         Lambda=ekfac_factors["pretrain"]["Lambda"],
                                         damping=pretrain_damping,
                                         vec=cand_grad)
            infls.append(utils.ekfac_influence(ihvp_cand, ihvp_tox)[0])
        infls = np.array(infls)
        infl_file = os.path.join(infls_dir, f"{tox_index}.pkl")
        with open(infl_file, "wb") as fp:
            pickle.dump(infls, fp)
    
    logger.info("All finished.")


if __name__ == "__main__":
    main()
