import os
import json
from tqdm import tqdm

from datasets import load_from_disk

import utils


trex_hf_path = "HF_TREX_sentences/"


def fact_map_to_ids(trex, lama_facts):
    fact_to_ids = {}
    for index, sample in enumerate(tqdm(trex)):
        facts = sample['facts']
        for fact in facts:
            if fact in lama_facts:
                ids = fact_to_ids.setdefault(fact, set())
                ids.add(index)
    for fact, ids in fact_to_ids.items():
        fact_to_ids[fact] = sorted(list(ids))
    return fact_to_ids


if __name__ == "__main__":
    trex = load_from_disk(trex_hf_path)
    lama = utils.load_lama_raw("LAMA/TREx")
    lama_facts = utils.get_lama_facts(lama)

    fact_to_ids = fact_map_to_ids(trex, lama_facts)
    with open("fact_to_ids.json", "w") as fp:
        json.dump(fact_to_ids, fp)
