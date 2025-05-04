import os
import pickle
import json
import joblib
import numpy as np

from transformers import (
    AutoTokenizer
)

from loguru import logger

import utils


pile_path = "/home/Dataset/pile-wikipedia-20200301-en.pkl"

# knn_candidates_path = "candidates-toxic.json"
knn_candidates_path = "candidates-qa.json"

topK = 16
max_length = 512

param_count = (2.8, 3)
# param_count = (6.9, 7)
# param_count = (12, 12)
pretrained_model_path = f"Model/Pythia/pythia-{param_count[0]}b/"
infls_dir = f"infls/dolly-{param_count[1]}b/full-qa/"
results_dir = f"results/dolly-{param_count[1]}b/full-qa/"


def truncate_text_by_tokenized_length(texts: list[str], tokenizer: AutoTokenizer, max_length: int):
    ids = tokenizer(texts, max_length=max_length, truncation=True, padding=False, return_attention_mask=False)['input_ids']
    texts = [tokenizer.decode(i) for i in ids]
    return texts


def main():
    logger.info("Loading The Pile.")
    # pile = utils.load_partial_pile(pile_path=pile_path, file_no=0, percent_begin=0, percent_end=100)
    pile = joblib.load(pile_path)

    with open(knn_candidates_path, "r") as fp:
        candidate_indices = json.load(fp)
    candidate_indices = np.array(candidate_indices)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, padding_side='right')
    tokenizer.pad_token_id = tokenizer.eos_token_id

    os.makedirs(results_dir, exist_ok=True)
    logger.info("Finding top-influential examples.")
    for i, candidate_ind in enumerate(candidate_indices):
        infl_file = os.path.join(infls_dir, f"{i}.pkl")
        with open(infl_file, "rb") as fp:
            infls = pickle.load(fp)
        sorted_indices = np.argsort(-infls)
        sorted_infls = infls[sorted_indices]
        # candidate_texts = pile[candidate_ind[sorted_indices].tolist()]["text"]
        candidate_texts = [pile[x]["title"] + "\n\n" + pile[x]["text"] for x in candidate_ind[sorted_indices].tolist()]

        sorted_infls = sorted_infls[:topK]
        candidate_texts = candidate_texts[:topK]
        candidate_texts = truncate_text_by_tokenized_length(candidate_texts, tokenizer, max_length)

        results = []
        for infl, text in zip(sorted_infls, candidate_texts):
            results.append({"infl": infl, "text": text})
        results_file = os.path.join(results_dir, f"{i}.json")
        with open(results_file, "w") as fp:
            json.dump(results, fp)
            logger.info(f"Results saved to {fp.name}")


if __name__ == "__main__":
    main()
