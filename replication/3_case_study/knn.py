import os

from sentence_transformers import SentenceTransformer

device = "cuda:0"

model = SentenceTransformer("all-MiniLM-L6-v2", device=device, cache_folder="./sentence_transformers-cache/")

from tqdm import tqdm
import json
import utils

pretrained_model_path = f"Model/Pythia/pythia-2.8b/"

batch_size = 512
n_pile_train_files = 30

pile_path = "/home/Dataset/pile-uncopyrighted/"

import multiprocessing as mp
ctx = mp.get_context('spawn')

n_subprocess = 16

from transformers import (
    AutoTokenizer, AutoConfig
)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
max_length = 512

def truncate_text_by_tokenized_length(texts: list[str], tokenizer: AutoTokenizer, max_length: int):
    ids = tokenizer(texts, return_attention_mask=False)['input_ids']
    ids_all = []
    for id in ids:
        ids_all.extend(id)
    n_batches = len(ids_all)/max_length
    n_batches = n_batches.__ceil__()
    ids_all = [ids_all[i*max_length:(i+1)*max_length] for i in range(n_batches)]
    texts = [tokenizer.decode(i) for i in ids_all]
    return texts

subset_sizes = []
embeddings = []
for file_no in tqdm(range(n_pile_train_files), desc="Pile files"):
    # for percent in tqdm(range(0, 100, 10), desc=f"File [{file_no}/{n_pile_train_files}]"):
    ds = utils.load_partial_pile(pile_path, file_no, 0, 10)
    n_batches = len(ds)/batch_size
    n_batches = n_batches.__ceil__()
    for i in tqdm(range(n_batches), desc='Embeddings'):
        batch = ds['text'][i*batch_size:(i+1)*batch_size]
        batch = truncate_text_by_tokenized_length(batch, tokenizer, max_length)
        embed = model.encode(batch)
        embeddings.extend(embed)
    # ds = utils.load_partial_pile(pile_path, file_no, 0, 100)
    subset_sizes.append(ds.num_rows)
    # n_batches = len(ds)/batch_size
    # n_batches = n_batches.__ceil__()
    # for i in tqdm(range(n_batches), desc='Embeddings'):
    #     batch = ds['text'][i*batch_size:(i+1)*batch_size]
    #     embed = model.encode(batch)
    #     embeddings.extend(embed)


# with open("pile_subset_sizes.json", "w") as fp:
#     json.dump(subset_sizes, fp)

print(len(embeddings))

from sklearn.neighbors import NearestNeighbors

K = 10_000
knn = NearestNeighbors(n_neighbors=K,
                       metric='cosine')
knn.fit(embeddings)

import joblib

joblib.dump(knn, "knn.joblib")

query = ["Survival of the Tastiest"]
query_embedding = model.encode(query)

indices = knn.kneighbors(query_embedding, return_distance=False)

print(ds[indices[0][0].item()]['text'][:64])
