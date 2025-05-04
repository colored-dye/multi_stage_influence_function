from absl import app, flags, logging
import json
import jsonlines
import numpy as np
import os
from tqdm import tqdm

import utils


FLAGS = flags.FLAGS
flags.DEFINE_string("candidates_dir", None, None, required=True)

recall_n = 10

def recall(neighbor_ids, real_ids, recall_n):
    nei = set(neighbor_ids[:recall_n])
    real = set(real_ids)
    return len(real.intersection(nei)) / len(real_ids)


def main(_):
    with open("../data/fact_to_ids.json", "r") as fp:
        fact_to_ids = json.load(fp)
    with open("test_data_selections.json", "r") as fp:
        selections = json.load(fp)
    
    test_data = utils.load_lama_raw("../data/LAMA/TREx")

    rrs_exps = []

    n_experiments = len(os.listdir(FLAGS.candidates_dir))
    # n_experiments = 2
    for exp_i in range(n_experiments):
        test_data_selections = [test_data[s] for s in selections[exp_i]]
        with jsonlines.open(os.path.join(FLAGS.candidates_dir, f"{exp_i}.jsonl"), "r") as reader:
            lines = tuple(reader.iter())
        test_data_selections = test_data_selections[:len(lines)] # TODO: Remove this line.
        # test_data_selections = test_data_selections[:4]
        rrs = []
        for i, example in enumerate(test_data_selections):
            fact = ",".join((example['predicate_id'], example['sub_uri'], example['obj_uri']))
            real_ids = fact_to_ids[fact]
            rr = recall(lines[i]['neighbor_ids'], real_ids, recall_n)
            rrs.append(rr)
        rrs_exps.append(np.mean(rrs))
    logging.info(f"{np.mean(rrs_exps)} +- {np.std(rrs_exps)}")


if __name__ == "__main__":
    app.run(main)
