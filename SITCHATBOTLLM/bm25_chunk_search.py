# bm25_helper.py

import pickle
import os
from rank_bm25 import BM25Okapi
from functools import lru_cache
import json
from nltk.tokenize import word_tokenize
import re
import time

# Normalize for case-insensitive exact match
def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower().strip()

# Load cached BM25 index (already built)
@lru_cache(maxsize=1)
def load_bm25_index():
    import time
    index_path = "bm25_index.pkl"

    start = time.time()
    if not os.path.exists(index_path):
        raise FileNotFoundError("BM25 index not found. Run bm25 indexer first.")
    with open(index_path, "rb") as f:
        bm25, documents, metadata_list = pickle.load(f)
    print(f"✅ BM25 index loaded in {time.time() - start:.2f} seconds")
    return bm25, documents, metadata_list

# Retrieve top N docs using BM25
def search_bm25(query, top_n=3):
    bm25, documents, metadata_list = load_bm25_index()

    norm_query = normalize_text(query)
    tokenized_query = norm_query.split()
    base_scores = bm25.get_scores(tokenized_query)

    # NO document normalization needed anymore

    top_indices = sorted(range(len(base_scores)), key=lambda i: base_scores[i], reverse=True)[:top_n]
    top_docs = [{"content": documents[i], "metadata": metadata_list[i]} for i in top_indices]

    return top_docs




# if __name__ == "__main__":
#     # Manual test for BM25
#     query = "Who are some profs working on speech privacy"
#     print(f"🔎 Searching BM25 for: '{query}'")
#     results = search_bm25(query, top_n=10)

#     for i, result in enumerate(results, 1):
#         print(f"\n--- Result {i} ---")
#         print("Metadata:", result["metadata"])
#         print("Snippet:", result["content"][:500])  # Print a preview of the chunk