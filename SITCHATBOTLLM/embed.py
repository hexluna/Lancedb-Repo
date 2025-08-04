import os
import json
import time
from dotenv import load_dotenv
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

# --- Load API Key ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file.")

# --- Config ---
INPUT_FILE = "batch_1_cleaned.jsonl"
INDEX_DIR = "faiss_batch_2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MAX_CHARS = 20000
BATCH_SIZE = 100
EMBED_MODEL = "text-embedding-3-small"
MAX_RETRIES = 3  # Number of retries on failure
RETRY_DELAY = 5  # Delay between retries (in seconds)

# --- Step 1: Load JSONL ---
print("🔍 Loading documents...")
raw_docs = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(tqdm(f, desc="📄 Reading JSONL")):
        try:
            item = json.loads(line)
            content = item.get("content", "").strip()
            if content:
                raw_docs.append(Document(
                    page_content=content,
                    metadata={"source": item.get("url", "")}
                ))
        except json.JSONDecodeError:
            print(f"⚠️ Skipping invalid JSON on line {i+1}")
print(f"✅ Loaded {len(raw_docs)} valid documents.")

# --- Step 2: Split and filter ---
print("✂️ Splitting documents...")
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
docs = splitter.split_documents(raw_docs)
docs = [doc for doc in docs if doc.page_content.strip()]
for doc in docs:
    if len(doc.page_content) > MAX_CHARS:
        doc.page_content = doc.page_content[:MAX_CHARS]
print(f"✅ Prepared {len(docs)} clean chunks.")

# --- Step 3: Initialize Embedder ---
print("🧠 Initializing OpenAI embedder...")
embedder = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=api_key)

# --- Step 4: Batch Embed ---
print("📡 Sending chunks to OpenAI in batches...")
all_vectors = []
all_docs = []

for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="🔄 Embedding Batches"):
    batch = docs[i:i + BATCH_SIZE]
    texts = [doc.page_content for doc in batch]
    
    retries = 0
    success = False
    while retries < MAX_RETRIES and not success:
        try:
            vectors = embedder.embed_documents(texts)
            all_vectors.extend(vectors)
            all_docs.extend(batch)
            success = True
        except Exception as e:
            print(f"❌ Error in batch {i}-{i+BATCH_SIZE}: {e}")
            retries += 1
            if retries < MAX_RETRIES:
                print(f"⏳ Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"⚠️ Max retries reached for batch {i}-{i+BATCH_SIZE}. Skipping.")
                break

print(f"✅ Embedded {len(all_vectors)} vectors.")

# --- Step 5: Save to FAISS ---
print("💾 Saving FAISS index...")
vectorstore = FAISS.from_documents(all_docs, embedder)
vectorstore.save_local(INDEX_DIR)
print(f"✅ FAISS index saved to '{INDEX_DIR}'")

# --- Step 6: Test Retrieval ---
print("🔎 Test retrieval...")
results = vectorstore.similarity_search("What is SIT's academic bulletin about?", k=3)
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---\n{doc.page_content[:300]}...")
