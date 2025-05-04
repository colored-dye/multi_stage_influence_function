"""LAMA use evidences, AKA excerpts from TREX, instead of templates."""

import os
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig, AutoTokenizer, GPTNeoXForCausalLM,
    GPT2LMHeadModel, BloomForCausalLM,
)


device = "cuda:3"

K = 3
BATCH_SIZE = 16

lama_base_dir = "../data/LAMA/TREx/"
lama_relations_path = "../data/LAMA/relations.jsonl"

model_path = "/home/Model/BLOOM/bloom-560m"

results_dir = "results/"


def load_lama_by_relation(lama_base_dir):
    lama_files = [os.path.join(lama_base_dir, f) for f in sorted(os.listdir(lama_base_dir)) if "json" in f]

    lama_all = {}
    for f in tqdm(lama_files, desc="Loading LAMA"):
        relation = f.strip(lama_base_dir).strip(".jsonl")
        lama_all[relation] = []
        with open(f, "r") as fp:
            for line in fp.readlines():
                data = json.loads(line)
                lama_all[relation].append(data)
    return lama_all


def load_relation_template(lama_relations_path):
    relations = {}
    with open(lama_relations_path, "r") as fp:
        for line in fp.readlines():
            relation = json.loads(line)
            rel = relation['relation']
            template = relation['template']
            relations[rel] = template
    return relations


def parse_template(template, subject_label, object_label):
    SUBJ_SYMBOL = "[X]"
    OBJ_SYMBOL = "[Y]"
    template = template.replace(SUBJ_SYMBOL, subject_label)
    template = template.replace(OBJ_SYMBOL, object_label)
    return template


def prepare_batch(texts: list[str], tokenizer: AutoTokenizer):
    tokenized_inputs = tokenizer(texts, padding='longest', truncation=True, max_length=2048, return_tensors='pt')
    masked_indices = []
    for ids in tokenized_inputs.input_ids:
        ind = torch.nonzero(ids==tokenizer.pad_token_id)
        if ind.numel() == 0:
            masked_indices.append(0)
        else:
            masked_indices.append(ind[0,0])
    return tokenized_inputs, masked_indices


@torch.no_grad
def top_k_accuracy(model: GPTNeoXForCausalLM,
                   tokenizer: AutoTokenizer,
                   lama_all: dict[str, list[dict]],
                   relation_templates: dict[str, str],
                   k: int):
    acc_all = 0
    n_samples = 0
    relation_to_acc = {}
    for relation, samples in tqdm(lama_all.items()):
        acc = 0

        evidences = []
        for sample in samples:
            evidences.append(sample['evidences'][0])

        n = len(evidences)
        n_samples += n

        n_batches = n / BATCH_SIZE
        n_batches = n_batches.__ceil__()

        template = relation_templates[relation]
        for batch_i in tqdm(range(n_batches), desc=relation):
            batch = []
            label_ids = []
            for sample in evidences[batch_i*BATCH_SIZE:(batch_i+1)*BATCH_SIZE]:
                prompt = sample['masked_sentence']
                mask_idx = prompt.find("[MASK]")
                prompt = prompt[:mask_idx].strip()
                batch.append(prompt)
                label_id = tokenizer.encode(" "+sample['obj_surface'])[0]
                label_ids.append(label_id)
            tokenized_inputs, masked_indices = prepare_batch(batch, tokenizer)
            tokenized_inputs = tokenized_inputs.to(device)
            logits = model(**tokenized_inputs).logits
            probs = torch.softmax(logits, dim=-1)

            for i in range(len(batch)):
                prob = probs[i, masked_indices[i]-1, :]
                weights, indices = torch.topk(prob, k=k)
                if label_ids[i] in indices:
                    acc += 1

        print(f"=={acc/n:.03f}==")
        relation_to_acc[relation] = acc / n
        acc_all += acc
    return acc_all / n_samples, relation_to_acc


def load_vocab(vocab_path):
    with open(vocab_path) as fp:
        tok_config = json.load(fp)
    vocab = tok_config['model']['vocab']
    vocab = set(vocab.keys())
    return vocab


def filter_lama_by_vocab(lama, vocab):
    lama_used = {}
    excluded = 0
    used = 0
    for relation, samples in lama.items():
        lama_used[relation] = []
        for sample in samples:
            if len(tokenizer.tokenize(" " + sample['obj_label'])) > 1:
                excluded += 1
            else:
                lama_used[relation].append(sample)
                used += 1
    print(f"Discarded samples: {excluded}, used: {used}")
    return lama_used


def process_state_dict(state_dict):
    res = {}
    for n, p in state_dict.items():
        n = n[len("model."):]
        res[n] = p
    return res


if __name__ == "__main__":
    lama = load_lama_by_relation(lama_base_dir)
    relation_templates = load_relation_template(lama_relations_path)

    # relation = "P17"
    # lama = {relation: lama[relation]}

    model_name = os.path.basename(model_path)
    print(f"Model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = BloomForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    vocab = tokenizer.vocab
    lama_used = filter_lama_by_vocab(lama, vocab)

    acc, relation_to_acc = top_k_accuracy(model, tokenizer, lama_used, relation_templates, K)
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"evidences-{model_name}-topk={K}.json"
    with open(os.path.join(results_dir, results_file), "w") as fp:
        json.dump({"acc_all": acc, "acc_by_relation": relation_to_acc}, fp)
