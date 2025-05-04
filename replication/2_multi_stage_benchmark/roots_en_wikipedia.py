import sys
sys.path.append("trace")
import utils

from tqdm import tqdm

test_data = utils.load_lama_raw("./data/LAMA/TREx")

objects = set()
subjects = set()

for d in test_data:
    objects.add(d['obj_uri'])
    subjects.add(d['sub_uri'])

entities = subjects.union(objects)
print(f"Total: {len(entities)}; Subjects: {len(subjects)}; Objects: {len(objects)}")

from datasets import load_dataset
wiki = load_dataset("/home/Dataset/bigscience-data/roots_en_wikipedia", split='train')

for rec in tqdm(wiki):
    id = eval(rec['meta'])['wikidata_id'].decode()
    if id in objects:
        objects.remove(id)
    if id in subjects:
        subjects.remove(id)

entities = subjects.union(objects)

print(f"Total left: {len(entities)}; Subjects left: {len(subjects)}; Objects left: {len(objects)}")
