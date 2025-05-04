import os

from sentence_transformers import SentenceTransformer

device = "cuda:0"

from tqdm import tqdm
import json
import joblib

import utils

from transformers import (
    AutoTokenizer, AutoConfig
)

from loguru import logger

pretrained_model_path = f"Model/Pythia/pythia-2.8b/"

batch_size = 4096

pile_wikipedia_path = "/home/Dataset/pile-wikipedia-20200301-en.pkl"

logger.info("Loading dataset.")
ds = joblib.load(pile_wikipedia_path)

model = SentenceTransformer("all-MiniLM-L6-v2", device=device, cache_folder="./sentence_transformers-cache/")

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
max_length = 512

def truncate_text_by_tokenized_length(texts: list[str], tokenizer: AutoTokenizer, max_length: int):
    ids = tokenizer(texts, max_length=max_length, truncation=True, padding=False, return_attention_mask=False)['input_ids']
    texts = [tokenizer.decode(i) for i in ids]
    return texts

embeddings = []
n_batches = len(ds)/batch_size
n_batches = n_batches.__ceil__()
for i in tqdm(range(n_batches), desc='Embeddings'):
    batch = []
    for r in ds[i*batch_size:(i+1)*batch_size]:
        text = r['title'] + "\n\n" + r['text']
        batch.append(text)
    batch = truncate_text_by_tokenized_length(batch, tokenizer, max_length)
    embed = model.encode(batch)
    embeddings.extend(embed)


print(len(embeddings))

from sklearn.neighbors import NearestNeighbors

K = 10_000
knn = NearestNeighbors(n_neighbors=K,
                       metric='cosine')
knn.fit(embeddings)


joblib.dump(knn, "knn-wikipedia.joblib")
