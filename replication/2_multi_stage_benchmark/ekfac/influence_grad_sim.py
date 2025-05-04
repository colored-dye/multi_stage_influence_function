"""∇_θ L(z_test, θ_ft)^T ∇_θ L(z_pre, θ_pt)"""

import json
import numpy as np
from datasets import load_from_disk
from loguru import logger
import os
import torch
from tqdm import tqdm
from transformers import (
    AutoConfig, AutoTokenizer, BloomForCausalLM,
)

import ekfac_utils
import sys
sys.path.append("../trace/")
import utils


device = "cuda:2"
ft_model_path = "/storage/Model/bigscience/bloomz-560m"
pt_model_path = "/storage/Model/bigscience/bloom-560m"
factors_dir = "factors_finetune/"
DAMPING = 1e-4
topk = 100

results_dir = "../trace/bloom_grad_sim_candidates/"


def process_state_dict(state_dict):
    res = {}
    for n, p in state_dict.items():
        n = n[len("model."):]
        res[n] = p
    return res


def parse_template(template, subject_label):
    SUBJ_SYMBOL = "[X]"
    template = template.replace(SUBJ_SYMBOL, subject_label)
    return template


def get_query_gradients(model, modules_of_interest, tokenizer, sample, relation_templates):
    prompt = parse_template(relation_templates[sample['predicate_id']], sample['sub_label'])
    prompt_ids = tokenizer.encode(prompt,
                           max_length=2048,
                           truncation=True,
                           padding=False)
    targets = " " + sample['obj_label']
    target_ids = tokenizer.encode(targets,
                           max_length=2048,
                           truncation=True,
                           padding=False)
    ids = prompt_ids + target_ids
    ids = torch.tensor([ids])
    labels = ids.clone()
    labels[0, :len(prompt_ids)] = -100
    labels[labels == tokenizer.pad_token_id] = -100

    input_ids = ids[:, :-1].to(device)
    labels = labels[:, 1:].to(device)

    grads = ekfac_utils.example_grads(model=model,
                                      modules_of_interest=modules_of_interest,
                                      hidden_size=model.config.hidden_size,
                                      n_heads=model.config.num_attention_heads,
                                      inputs=input_ids,
                                      targets=labels)
    return grads


def get_candidate_gradients(model, modules_of_interest, tokenizer, text):
    ids = tokenizer.encode(text,
                           max_length=2048,
                           truncation=True,
                           padding=False,
                           return_tensors='pt').to(device)
    input_ids = ids[:, :-1]
    labels = ids[:, 1:]
    grads = ekfac_utils.example_grads(model=model,
                                      modules_of_interest=modules_of_interest,
                                      hidden_size=model.config.hidden_size,
                                      n_heads=model.config.num_attention_heads,
                                      inputs=input_ids,
                                      targets=labels)
    return grads


def load_factors(factors_dir):
    QA = torch.load(os.path.join(factors_dir, "QA.pt"), map_location=device)
    QG = torch.load(os.path.join(factors_dir, "QG.pt"), map_location=device)
    Lambda = torch.load(os.path.join(factors_dir, "Lambda.pt"), map_location=device)
    return QA, QG, Lambda


def load_relation_template(lama_relations_path):
    relations = {}
    with open(lama_relations_path, "r") as fp:
        for line in fp.readlines():
            relation = json.loads(line)
            rel = relation['relation']
            template = relation['question']
            relations[rel] = template
    return relations


def main():
    train_data = load_from_disk("../data/HF_TREX_sentences")
    test_data = utils.load_lama_raw("../data/LAMA/TREx")
    relation_templates = load_relation_template("../data/LAMA/bloom_questions.jsonl")

    with open("../trace/test_data_selections.json", "r") as fp:
        selection_indices = json.load(fp)
    
    with open("../trace/reranker_candidates.json", "r") as fp:
        reranker_candidates = json.load(fp)
    
    n_experiments = len(selection_indices)
    os.makedirs(results_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(ft_model_path)

    ft_model = BloomForCausalLM.from_pretrained(ft_model_path, device_map=device)
    pt_model = BloomForCausalLM.from_pretrained(pt_model_path, device_map=device)

    modules_of_interest = {}
    for i in range(ft_model.config.num_hidden_layers):
        modules_of_interest[f'transformer.h.{i}.self_attention.query_key_value'] = 'mha'
        modules_of_interest[f'transformer.h.{i}.self_attention.dense'] = 'mlp'
        modules_of_interest[f'transformer.h.{i}.mlp.dense_h_to_4h'] = 'mlp'
        modules_of_interest[f'transformer.h.{i}.mlp.dense_4h_to_h'] = 'mlp'

    for exp_i in range(n_experiments):
        test_data_selections = [test_data[s] for s in selection_indices[exp_i]]
        with open(os.path.join(results_dir, f"{exp_i}.jsonl"), "w") as fp:
            for qi, test_sample in enumerate(tqdm(test_data_selections, desc=f"Exp {exp_i}")):
                query_grads = get_query_gradients(model=ft_model,
                                            modules_of_interest=modules_of_interest,
                                            tokenizer=tokenizer,
                                            sample=test_sample,
                                            relation_templates=relation_templates)
                scores = []
                for ci in tqdm(reranker_candidates[exp_i][qi], desc=f"Query [{qi}]"):
                    candidate_text = train_data[ci]['text']
                    cand_grads = get_candidate_gradients(model=pt_model,
                                               modules_of_interest=modules_of_interest,
                                               tokenizer=tokenizer,
                                               text=candidate_text)
                    infl, _ = ekfac_utils.ekfac_influence(modules_of_interest=modules_of_interest,
                                                          ihvps=query_grads,
                                                          grads=cand_grads)
                    scores.append(infl)
                scores = np.array(scores)
                idxs = np.argpartition(scores, -topk)[-topk :]
                nn_idxs = idxs[np.argsort(-scores[idxs])]
                
                nn_scores = scores[nn_idxs].tolist()
                neighbor_ids = [reranker_candidates[exp_i][qi][idx] for idx in nn_idxs]
                line = {"scores": nn_scores, "neighbor_ids": neighbor_ids}
                print(json.dumps(line), file=fp, flush=True)


if __name__ == "__main__":
    main()

