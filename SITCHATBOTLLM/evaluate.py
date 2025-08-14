import json
import time
from pathlib import Path
from dotenv import load_dotenv
from query_test_new import query_faiss
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Config ---
DATA_FILE = "benchmark_dataset_5.json"
RESULTS_FILE = "benchmark_results.json"
SIMILARITY_THRESHOLD = 0.7  # Pass mark for semantic similarity

# --- Load environment ---
load_dotenv()

# --- Load benchmark dataset ---
if not Path(DATA_FILE).exists():
    raise FileNotFoundError(f"Benchmark dataset not found: {DATA_FILE}")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)

print(f"Loaded {len(benchmark_data)} benchmark questions.")

# --- Init embedder ---
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

def semantic_score(a: str, b: str) -> float:
    """Returns cosine similarity between two texts using embeddings."""
    try:
        a_vec = embedder.embed_query(a)
        b_vec = embedder.embed_query(b)
        return cosine_similarity([a_vec], [b_vec])[0][0]
    except Exception as e:
        print(f"Embedding error: {e}")
        return 0.0

# --- JSON-safe converter for NumPy types ---
def json_safe(obj):
    if isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    return obj

# --- Evaluation loop ---
results = []
pass_count = 0

for i, item in enumerate(benchmark_data, start=1):
    query = item["query"]
    expected = item["expected_answer"]

    print(f"\n[{i}/{len(benchmark_data)}] {query}")
    start_time = time.time()
    actual = query_faiss(query)
    elapsed = time.time() - start_time

    # Compute similarity score
    score = semantic_score(expected, actual)
    passed = score >= SIMILARITY_THRESHOLD
    if passed:
        pass_count += 1

    # Store results
    results.append({
        "id": int(i),
        "query": str(query),
        "expected": str(expected),
        "actual": str(actual),
        "similarity": float(round(score, 4)),
        "pass": bool(passed),
        "time_taken_sec": float(round(elapsed, 2)),
        "category": str(item.get("category", "")),
        "source_url": str(item.get("metadata", {}).get("source_url", ""))
    })

# --- Save results safely ---
with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=json_safe)

# --- Print summary ---
accuracy = pass_count / len(benchmark_data) * 100
print("\nBenchmark Completed")
print(f"Total queries: {len(benchmark_data)}")
print(f"Pass mark: {SIMILARITY_THRESHOLD*100:.0f}% similarity")
print(f"Passed: {pass_count} / {len(benchmark_data)}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Results saved to {RESULTS_FILE}")