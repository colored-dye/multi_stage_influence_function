import os
from tqdm import tqdm
import json


def load_lama_raw(lama_base_dir):
    lama_files = [os.path.join(lama_base_dir, f) for f in sorted(
        os.listdir(lama_base_dir)) if "json" in f]

    lama_all = []
    for f in tqdm(lama_files, desc="Loading LAMA"):
        with open(f, "r") as fp:
            for line in fp.readlines():
                data = json.loads(line)
                lama_all.append(data)
    return lama_all


def get_lama_facts(lama_all):
    facts = set()
    for sample in lama_all:
        pred = sample['predicate_id']
        sub = sample['sub_uri']
        obj = sample['obj_uri']
        fact = ",".join((pred, sub, obj))
        facts.add(fact)
    return facts
