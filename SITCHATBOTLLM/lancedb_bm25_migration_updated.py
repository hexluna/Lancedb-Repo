import os
import threading
import time

import lancedb
import pickle
import numpy as np
import pyarrow as pa
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from more_itertools import chunked

from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Constants
CHUNK_SIZE = 50000
BATCH_SIZE = 1000
MAX_WORKERS = 8  # tune it based on CPU

# Initialize embedder
embedder = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Load FAISS store
faiss_store = FAISS.load_local("combined_faiss", embedder, allow_dangerous_deserialization=True)

# Load BM25 model
with open("bm25_index.pkl", "rb") as f:
    bm25_model, bm25_corpus, bm25_tokens = pickle.load(f)

print(f"BM25 corpus length: {len(bm25_corpus)}")

# Connect to LanceDB
uri = "data/vector-index-lancedb"
db = lancedb.connect(uri)

# Define LanceDB schemas
faiss_schema = pa.schema([
    ("id", pa.int32()),
    ("text", pa.string()),
    ("metadata", pa.struct([])),
    ("vector", pa.list_(pa.float32(), 1536))
])
bm25_schema = pa.schema([
    ("id", pa.int32()),
    ("text", pa.string()),
    ("metadata", pa.struct([("source", pa.string())]))
])

# Create tables (overwrite if exist)
faiss_table = db.create_table("faiss_index", schema=faiss_schema, mode="overwrite")
bm25_table = db.create_table("bm25_index", schema=bm25_schema, mode="overwrite")

# ------------------------ FAISS PARALLEL MIGRATION ------------------------ #
def build_faiss_doc(i):
    try:
        doc = faiss_store.docstore.search(str(i)) or faiss_store.docstore.search(i)
        if not doc:
            return None
        vec = faiss_store.index.reconstruct(i)
        vec_array = np.array(vec, dtype=np.float32)
        assert vec_array.shape == (1536,), f"Vector at index {i} has wrong shape: {vec_array.shape}"
        return {
            "id": i,
            "text": doc if isinstance(doc, str) else getattr(doc, "page_content", ""),
            "metadata": getattr(doc, "metadata", {}),
            "vector": vec_array.tolist()
        }
    except Exception as e:
        print(f"Skipped index {i}: {e}")
        return None

print("Starting FAISS vector hybrid migration...")
faiss_total = faiss_store.index.ntotal
CHUNK_SIZE = 2000  # tune it based on memory capacity

for start in tqdm(range(0, faiss_total, CHUNK_SIZE), desc="Chunked Hybrid Migration"):
    end = min(start + CHUNK_SIZE, faiss_total)
    faiss_batch = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(build_faiss_doc, i): i for i in range(start, end)}

        for future in as_completed(futures):
            result = future.result()
            if result:
                faiss_batch.append(result)

            if len(faiss_batch) >= BATCH_SIZE:
                faiss_table.add(faiss_batch)
                faiss_batch.clear()

    if faiss_batch:
        faiss_table.add(faiss_batch)

print("FAISS hybrid migration complete")
# ------------------------------ Create HNSW Index ------------------------------ #
print("Creating HNSW index for FAISS vectors")

faiss_table.create_index(
    metric="cosine",
    vector_column_name="vector",
    replace=True,
    index_type="IVF_HNSW_SQ",
    m=20,
    ef_construction=300,
    num_partitions = 512,
    num_sub_vectors = 96,
)

print("HNSW index created.")

print("FAISS preview:")
print(faiss_table.head(3))
print(f"FAISS row count: {faiss_table.count_rows()}")



# ------------------------ BM25 MIGRATION (Batched) ------------------------ #
print("Starting BM25 corpus migration...")
for batch in tqdm(chunked(enumerate(bm25_corpus), BATCH_SIZE), total=len(bm25_corpus)//BATCH_SIZE + 1, desc="Migrating BM25"):
    bm25_table.add([{
        "id": i,
        "text": doc,
        "metadata": {"source": "bm25"}
    } for i, doc in batch])

print("BM25 migration complete")
print("BM25 preview:")
print(bm25_table.head(3))
print(f"BM25 row count: {bm25_table.count_rows()}")

print("Migration done: FAISS + BM25 stored in LanceDB")


# ------------------------------ Debug Info ------------------------------ #
# print("\n🧠 LanceDB Debug Info:")
# print("FAISS Table")
# print(f"- Schema: {faiss_table.schema}")
# print(f"- Partitions: {len(faiss_table.to_lance().get_fragments())}")
# print(f"- Row Count: {faiss_table.count_rows()}")
# print(f"- Indexes: {faiss_table.list_indexes()}")
# print("🧪 Sample rows:")
# print(faiss_table.head(2))
#
# print("\nBM25 Table")
# print(f"- Schema: {bm25_table.schema}")
# print(f"- Partitions: {len(bm25_table.to_lance().get_fragments())}")
# print(f"- Row Count: {bm25_table.count_rows()}")
# print("🧪 Sample rows:")
# print(bm25_table.head(2))

print("\n🎉 Migration and debug complete.")