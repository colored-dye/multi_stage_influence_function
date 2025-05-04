"""Find TREX samples with LAMA knowledge."""


from datasets import Dataset
import os
import json
from tqdm import tqdm
from loguru import logger

import utils


lama_base_dir = "LAMA/TREx/"

trex_base_dir = "TREX/"
trex_hf_path = "HF_TREX_sentences/"

n_samples_per_file = 2200

pred_uri_prefix = "http://www.wikidata.org/prop/direct/"
entity_uri_prefix = "http://www.wikidata.org/entity/"


def format_triples_filter_lama_facts(data: list[dict], lama_facts: set[str]):
    filtered_data = []
    for rec in data:
        if len(filtered_data) == n_samples_per_file:
            break
        triples = []
        uri_valid = True
        fact_in_lama = False
        for triple in rec['triples']:
            pred = triple['predicate']["uri"]
            if pred_uri_prefix not in pred:
                uri_valid = False
                break
            else:
                pred = pred.strip(pred_uri_prefix)
            
            sub = triple['subject']['uri']
            if entity_uri_prefix not in sub:
                uri_valid = False
                break
            else:
                sub = sub.strip(entity_uri_prefix)
            
            obj = triple['object']['uri']
            if entity_uri_prefix not in obj:
                uri_valid = False
                break
            else:
                obj = obj.strip(entity_uri_prefix)
            
            fact = ",".join((pred, sub, obj))
            if fact in lama_facts:
                fact_in_lama = True
            
            triples.append((pred, sub, obj))
        if not uri_valid or not fact_in_lama:
            continue
        for i in range(len(triples)):
            rec['triples'][i]['predicate']['uri'] = triples[i][0]
            rec['triples'][i]['subject']['uri'] = triples[i][1]
            rec['triples'][i]['object']['uri'] = triples[i][2]
        filtered_data.append(rec)
    return filtered_data


def parse_fact(triple: dict):
    pred = triple['predicate']["uri"]
    if pred_uri_prefix not in pred:
        return None
    else:
        pred = pred.strip(pred_uri_prefix)
    
    sub = triple['subject']['uri']
    if entity_uri_prefix not in sub:
        return None
    else:
        sub = sub.strip(entity_uri_prefix)
    
    obj = triple['object']['uri']
    if entity_uri_prefix not in obj:
        return None
    else:
        obj = obj.strip(entity_uri_prefix)
    
    fact = ",".join((pred, sub, obj))
    return fact


def load_trex_raw(trex_base_dir, lama_facts):
    """
    Load original TREX dataset.
    Separate each sample into multiple sentences, with each sentence paired with multiple fact triples.
    """
    trex_files = [os.path.join(trex_base_dir, f) for f in os.listdir(trex_base_dir) if "json" in f]
    trex_files = sorted(trex_files, key=lambda x: int(x.strip(os.path.join(trex_base_dir, "re-nlg_")).split("-")[0]))

    trex_all = []
    for f in tqdm(trex_files, desc="Loading TREX"):
        with open(f, "r") as fp:
            data = json.load(fp)

        for sample in data:
            text = sample['text']
            sentences_boundaries = sample['sentences_boundaries']
            sentences = [None] * len(sentences_boundaries)
            for triple in sample['triples']:
                sentence_id = triple['sentence_id']
                start, end = sentences_boundaries[sentence_id]
                if sentences[sentence_id] is None:
                    sentences[sentence_id] = {}
                sentences[sentence_id]['text'] = text[start:end]
                fact = parse_fact(triple)
                if fact is not None:
                    facts = sentences[sentence_id].setdefault('facts', [])
                    if fact not in facts:
                        facts.append(fact)
            indices_to_remove = []
            for i in range(len(sentences)):
                if sentences[i] is None:
                    indices_to_remove.append(i)
                    continue
                if 'facts' not in sentences[i]:
                    indices_to_remove.append(i)
                    continue
                fact_in_lama = False
                for fact in sentences[i]['facts']:
                    if fact in lama_facts:
                        fact_in_lama = True
                        break
                if not fact_in_lama:
                    indices_to_remove.append(i)
            for i in sorted(indices_to_remove, reverse=True):
                del sentences[i]

            trex_all.extend(sentences)
    return trex_all


if __name__ == "__main__":
    lama_all = utils.load_lama_raw(lama_base_dir)
    lama_facts = utils.get_lama_facts(lama_all)

    trex_all = load_trex_raw(trex_base_dir, lama_facts)
    logger.info(f"# of TREX samples: {len(trex_all)}")
    print(trex_all[0])

    ds = Dataset.from_list(trex_all)
    ds.save_to_disk(trex_hf_path)
    logger.info(ds)
