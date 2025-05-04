import json
import jsonlines
import numpy as np
from absl import app, logging
from datasets import load_from_disk
import os
from rank_bm25 import BM25Plus
from tqdm import tqdm
import utils

topk = 100
n_selections = 200
results_dir = "bm25_candidates_evidences"


def main(_):
    train_data = load_from_disk("../data/HF_TREX_sentences")
    
    test_data = utils.load_lama_raw("../data/LAMA/TREx")
    relation_templates = utils.load_relation_template("../data/LAMA/relations.jsonl")

    corpus = train_data['text']
    logging.info("Preparing bm25...")
    bm25 = BM25Plus(corpus)
    logging.info("bm25 OK.")

    with open("test_data_selections.json", "r") as fp:
        selections = json.load(fp)

    n_experiments = len(selections)
    os.makedirs(results_dir, exist_ok=True)
    for exp_i in range(n_experiments):
        test_data_selections = []
        for s in selections[exp_i]:
            test_data_selections.append(test_data[s])
        with jsonlines.open(os.path.join(results_dir, f"{exp_i}.jsonl"), "w") as writer:
            for example in tqdm(test_data_selections, desc=f"Exp {exp_i}"):
                query = utils.get_tokenized_query(example, relation_templates)
                scores = bm25.get_scores(query)
                idxs = np.argpartition(scores, -topk)[-topk :]
                nn_idxs = idxs[np.argsort(-scores[idxs])]
                
                nn_scores = scores[nn_idxs].tolist()
                neighbor_ids = nn_idxs.tolist()

                line = {"scores": nn_scores, "neighbor_ids": neighbor_ids}
                writer.write(line)


if __name__ == "__main__":
    app.run(main)
