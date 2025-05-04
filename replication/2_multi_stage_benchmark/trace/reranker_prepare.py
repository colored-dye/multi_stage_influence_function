"""Prepare candidates for TDA methods to compute influence."""

from absl import app, flags, logging
from datasets import load_from_disk
import json
import numpy as np
import os
from tqdm import tqdm

import utils


FLAGS = flags.FLAGS
flags.DEFINE_string("bm25_candidates_dir", None, None, required=True)
flags.DEFINE_string("output_file", None, None, required=True)

test_size = 200
reranker_size = 100


def test_data_proponents(test_data_selections, fact_to_ids):
    proponents = []
    for rec in test_data_selections:
        predicate = rec['predicate_id']
        obj = rec['obj_uri']
        sub = rec['sub_uri']
        fact = ",".join((predicate, sub, obj))
        proponents.append(fact_to_ids[fact])
    return proponents


def random_samples_same_target(test_data_selections, target_sharing_ids, seed):
    ids = []
    np.random.seed(seed)
    for rec in test_data_selections:
        obj = rec['obj_uri']
        sharing_ids = target_sharing_ids[obj]
        if len(sharing_ids) <= reranker_size:
            ids.append(sharing_ids)
        else:
            choices = np.random.choice(len(sharing_ids), reranker_size, replace=True)
            sharing = []
            for c in choices:
                sharing.append(sharing_ids[c])
            ids.append(sharing)
    return ids


def union_candidates(global_random, groundtruth, bm25, sharing):
    candidates = []
    for x, y, z in zip(groundtruth, bm25, sharing):
        cand = []
        cand.extend(global_random)
        cand.extend(x)
        cand.extend(y)
        cand.extend(z)
        cand = list(set(cand))
        candidates.append(cand)
    return candidates


def main(_):
    with open("../data/fact_to_ids.json", "r") as fp:
        fact_to_ids = json.load(fp)
    with open("test_data_selections.json", "r") as fp:
        selection_indices = json.load(fp)
    with open("target_sharing_ids.json", "r") as fp:
        target_sharing_ids = json.load(fp)

    train_data = load_from_disk("../data/HF_TREX_sentences")
    test_data = utils.load_lama_raw("../data/LAMA/TREx")

    n_experiments = len(selection_indices)

    np.random.seed(42)
    random_choices = np.random.choice(train_data.num_rows, test_size, replace=False)
    global_random_ids = random_choices.tolist()

    reranker_candidates = []
    for exp_i in range(n_experiments):
        test_data_selections = []
        for s in selection_indices[exp_i]:
            test_data_selections.append(test_data[s])

        # 1. True proponents
        groundtruth_proponents = test_data_proponents(test_data_selections, fact_to_ids)

        # 2. Fetched by bm25
        bm25_candidates = []
        with open(os.path.join(FLAGS.bm25_candidates_dir, f"{exp_i}.jsonl"), 'r') as fp:
            for line in fp.readlines():
                data = json.loads(line)
                bm25_candidates.append(data['neighbor_ids'])

        # 3. 100 random samples that share the same target as query
        sharing_ids = random_samples_same_target(test_data_selections, target_sharing_ids, exp_i)

        candidates = union_candidates(global_random_ids, groundtruth_proponents, bm25_candidates, sharing_ids)
        reranker_candidates.append(candidates)
    with open(FLAGS.output_file, "w") as fp:
        json.dump(reranker_candidates, fp)


if __name__ == "__main__":
    app.run(main)

