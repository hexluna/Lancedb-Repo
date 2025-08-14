import os
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
import tiktoken
from bm25_chunk_search import search_bm25
from langchain.schema import Document
import re
import json
import time
import lancedb
from langchain.callbacks.base import BaseCallbackHandler
from concurrent.futures import ThreadPoolExecutor
import tiktoken
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
rerank_embedder = SentenceTransformer("all-MiniLM-L6-v2")


class TimingStreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, retrieval_end_time):
        self.retrieval_end_time = retrieval_end_time
        self.first_token_reported = False

    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"\n Query sent to LLM at: {time.strftime('%X')}")

    def on_llm_new_token(self, token: str, **kwargs):
        if not self.first_token_reported:
            first_token_time = time.time()
            latency = first_token_time - self.retrieval_end_time
            print(f"\n First token streamed at: {time.strftime('%X')} (Latency after retrieval: {latency:.3f}s)\n")
            self.first_token_reported = True
        print(token, end="", flush=True)

    def on_llm_end(self, response, **kwargs):
        print("\n\n✅ Streaming finished.")

    def on_llm_error(self, error, **kwargs):
        print(f"\n❌ Error: {error}")

# Helper function to Deduplicate the FAISS docs --> So that they wont have repeated FAISS docs
def dedup_documents(docs):
    seen = set()
    deduped = []
    for doc in docs:
        key = doc.page_content.strip().lower()
        if key not in seen:
            deduped.append(doc)
            seen.add(key)
    return deduped


# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load .env variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

# Prompt to rewrite queries for better retrieval
rewrite_prompt = PromptTemplate.from_template(
    "Rewrite the following question to make it more suitable for document retrieval in a university knowledge base:\n\nOriginal: {query}\n\nRewritten:"
)
rewrite_chain = rewrite_prompt | ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.3) | StrOutputParser()

# Initialize embedding and retriever
embedder = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
start_faiss = time.time()
# vectorstore = FAISS.load_local("combined_faiss", embedder, allow_dangerous_deserialization=True)
# print(f"✅ FAISS loaded in {time.time() - start_faiss:.2f} seconds")

print("Connecting to LanceDB...")
db = lancedb.connect("data/vector-index-lancedb")
lance_table = db.open_table("faiss_index")
print(f"LanceDB connected in {time.time() - start_faiss:.2f} seconds")

# SIT-related keyword detection (optional feature, currently unused)
SIT_KEYWORDS = ["SIT", "Singapore Institute of Technology", "AI Centre", "undergraduate programs",
                "postgraduate programs", "admissions", "research", "alumni", "students", "campus", "Punggol"]


def preprocess_query(query):
    """Remove English stopwords for BM25-style keyword matching."""
    stop_words = set(stopwords.words("english"))
    query_tokens = query.split()
    return " ".join([word for word in query_tokens if word.lower() not in stop_words])


# Create a cache file for all abbreviations
ABBR_CACHE_FILE = "abbreviation_cache.json"

# Load cache if exists
if os.path.exists(ABBR_CACHE_FILE):
    with open(ABBR_CACHE_FILE, "r") as f:
        abbreviation_map = json.load(f)
else:
    abbreviation_map = {}


def detect_abbreviations(text):
    # Detect patterns like AAI, FSSD, SIT (3+ uppercase letters)
    return re.findall(r'\b[A-Z]{2,}\b', text)


# Resolve the abbreviation and save it in a cache file
def resolve_abbreviations(abbrs):
    for abbr in abbrs:
        if abbr not in abbreviation_map:
            user_input = input(f"❓ What does '{abbr}' stand for? ")
            abbreviation_map[abbr] = user_input.strip()
    with open(ABBR_CACHE_FILE, "w") as f:
        json.dump(abbreviation_map, f, indent=2)
    return abbreviation_map


def expand_query_with_abbreviations(query):
    abbrs = detect_abbreviations(query)
    if abbrs:
        print(f"🔍 Detected abbreviations: {abbrs}")
        resolve_abbreviations(abbrs)

        for abbr, full_form in abbreviation_map.items():
            # Replace abbreviation with abbreviation + explanation
            query = re.sub(rf"\b{abbr}\b", f"{abbr} ({full_form})", query)
    return query


def truncate_context_to_token_limit(docs, max_tokens=2500, chunk_token_limit=800):
    """
    Build context by interleaving chunks from multiple documents.
    Each document is split into chunks of chunk_token_limit.
    Chunks are added in round-robin until max_tokens is reached.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    token_count = 0
    context_chunks = []

    # split all docs into chunks
    all_chunks = []
    for i, doc in enumerate(docs):
        program = doc.metadata.get('program', 'Unknown Program')
        prefix = f"--- Source: {doc.metadata.get('source', 'unknown').upper()} | Program: {program} ---\n"
        tokens = enc.encode(doc.page_content.strip())
        start = 0
        while start < len(tokens):
            end = min(start + chunk_token_limit, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = prefix + enc.decode(chunk_tokens)
            all_chunks.append({
                "text": chunk_text,
                "token_len": len(chunk_tokens)
            })
            start = end

    # interleave chunks round-robin style
    # Sort chunks by original doc index to preserve ordering per doc
    # Then pick one chunk from each doc iteratively
    from collections import defaultdict, deque

    doc_chunk_map = defaultdict(deque)
    for chunk in all_chunks:
        doc_chunk_map[chunk['text']].append(chunk)

    # Flatten into a queue for round-robin interleaving
    chunk_queue = deque(all_chunks)
    while chunk_queue and token_count < max_tokens:
        chunk = chunk_queue.popleft()
        if token_count + chunk['token_len'] > max_tokens:
            break
        context_chunks.append(chunk['text'])
        token_count += chunk['token_len']

    final_context = "\n\n".join(context_chunks)
    print(f"🧮 Final token count in context: {token_count}")
    return final_context


def generate_response_from_context(query, context, retrieval_end_time):
    prompt_messages = [
        SystemMessage(content="""
                    You are an expert academic advisor for Singapore Institute of Technology (SIT). 
                    Use the provided context to **analyze and compare** the information in a helpful and structured way.

                    If the question asks for **differences or similarities**, look for clues across multiple documents.
                    If the context includes details for both topics, synthesize the comparison clearly.
                    If any topic is not mentioned in the context, say so explicitly.
                    """),

        HumanMessage(content=f"""
                    Context:
                    {context}

                    Question:
                    {query}

                    Instructions:
                    - If the question involves comparing two or more programs, look for and explain their differences.
                    - Use bullet points or a short paragraph format to clarify your answer.
                    - Do not say "I don’t know" unless **both** programs are completely missing from the context.
                    Answer:
                    """)

    ]
    callback_manager = CallbackManager([TimingStreamingCallbackHandler(retrieval_end_time)])

    model = ChatOpenAI(
        model_name="gpt-4.1-mini",
        temperature=0.2,
        streaming=True,
        callback_manager=callback_manager
    )

    # print("\n⏳ Streaming LLM response started at:", time.strftime("%X"))
    response = model.invoke(prompt_messages)
    print("\n\n✅ Streaming finished.")
    return response.content

    # model = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.2)
    # return model.invoke(prompt_messages).content
import time  # Make sure this is imported at the top


def truncate_for_rerank(text, token_limit=2000):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return enc.decode(tokens[:token_limit])


def safe_embed_document_local(text, chunk_token_limit=2000):
    """Split long docs into smaller chunks, embed locally, and average."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    if len(tokens) <= chunk_token_limit:
        return rerank_embedder.encode(text)

    # Split into chunks
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_token_limit, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        start = end

    # Embed each chunk and average
    chunk_embeddings = rerank_embedder.encode(chunks)
    return np.mean(chunk_embeddings, axis=0)

def batch_embed_documents_parallel_local(docs, max_workers=5):
    """Embed multiple documents in parallel locally."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        embeddings = list(executor.map(safe_embed_document_local, docs))
    return embeddings


def query_faiss(query):
    if not query.strip():
        return "Please enter a question."

    total_start = time.time()

    print(f"❓ Original Query: {query}")

    step_start = time.time()
    expanded_query = expand_query_with_abbreviations(query)
    rewritten = rewrite_chain.invoke({"query": expanded_query})
    print(f"✏️ Rewritten for LLM (with expansions): {rewritten}")
    print(f"⏱️ Query rewrite took {time.time() - step_start:.2f}s")

    bm25_keywords = rewritten.strip()

    cleaned_query = preprocess_query(rewritten)
    sit_bias_keywords = " Singapore Institute of Technology SIT university campus students programs"
    cleaned_query += sit_bias_keywords

    print("🔍 Running FAISS vector search...")
    step_start = time.time()
    query_vector = embedder.embed_query(cleaned_query)
    results = lance_table.search(query_vector, vector_column_name="vector").limit(10).to_list()
    raw_faiss_docs = [
        Document(
            page_content=row.get("text", ""),
            metadata=row.get("metadata", {}) | {"source": "faiss"}
        )
        for row in results
    ]
    print(f"⏱️ FAISS retrieval took {time.time() - step_start:.2f}s")

    deduped_faiss_docs = dedup_documents(raw_faiss_docs)

    print("🔍 Running BM25 search...")
    step_start = time.time()
    bm25_results = search_bm25(bm25_keywords, top_n=10)
    bm25_docs = [
        Document(
            page_content=doc["content"],
            metadata={"source": "bm25", **doc.get("metadata", {})}
        )
        for doc in bm25_results
    ]
    print(f"⏱️ BM25 search took {time.time() - step_start:.2f}s")

    # Merge & deduplicate
    desired_total = 20
    seen_texts = set()
    combined_docs = []

    for doc in deduped_faiss_docs:
        txt = doc.page_content.strip().lower()
        if txt not in seen_texts:
            combined_docs.append(doc)
            seen_texts.add(txt)

    for doc in bm25_docs:
        txt = doc.page_content.strip().lower()
        if txt not in seen_texts and len(combined_docs) < desired_total:
            combined_docs.append(doc)
            seen_texts.add(txt)

    print(f"✅ Combined {len(combined_docs)} docs before reranking.")

    # ✅ Rerank with cosine similarity
    print("📊 Reranking with cosine similarity...")
    docs_for_rerank = [truncate_for_rerank(doc.page_content) for doc in combined_docs]
    doc_embeddings = batch_embed_documents_parallel_local(docs_for_rerank, max_workers=5)

    query_vector_local = rerank_embedder.encode(truncate_for_rerank(cleaned_query, 1000))

    # Compute cosine similarity and sort
    sims = cosine_similarity([query_vector_local], doc_embeddings)[0]
    scored_docs = list(zip(combined_docs, sims))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, _ in scored_docs]

    print("Top 5 reranked docs preview:")
    for i, (doc, score) in enumerate(scored_docs[:5]):
        print(f"{i + 1}. ({score:.4f}) {doc.page_content[:150]}...\n")

    # Build context after reranking
    context_build_start = time.time()
    context = truncate_context_to_token_limit(reranked_docs)
    print(f"⏱️ Context building took {time.time() - context_build_start:.2f}s")

    retrieval_end_time = time.time()
    response = generate_response_from_context(query, context, retrieval_end_time)

    print(f"✅ Total query time: {time.time() - total_start:.2f}s")
    return response

# Main interactive loop
if __name__ == "__main__":
    print("Test FAISS index (type 'exit' to quit):")
    print(lance_table.schema)
    while True:
        query = input("Your question: ")
        if query.lower() in ("exit", "quit"):
            break
        context = query_faiss(query)
        print("\nGenerated Response:\n", context)
