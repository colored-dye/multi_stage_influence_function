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


def get_tokenized_query(record, relation_templates) -> str:
    template = relation_templates[record['predicate_id']]
    text = parse_template(template, record['sub_label'], record['obj_label'])

    # evidence = record['evidences'][0]
    # text = evidence['masked_sentence'].replace('[MASK]', evidence['obj_surface'])
    return text
