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
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict
from datetime import datetime
from nltk.tokenize import sent_tokenize

bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
_embedding_cache = {}

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
import time  # Make sure this is imported at the top



#--------------------------------- Helpers for query_faiss ---------------------------------#
ABBREVIATION_GLOSSARY = {
    "LLM": "Large Language Model",
    "RAG": "Retrieval-Augmented Generation",
    "BM25": "Best Matching 25",
    "FAISS": "Facebook AI Similarity Search",
    "NLP": "Natural Language Processing"
}

def truncate_context_to_token_limit(docs, max_tokens=5000, chunk_token_limit=1000):
    """
    Split documents into semantic chunks (by paragraph/sentence) while respecting token limits.
    Then interleave chunks round-robin style until max_tokens is reached.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    token_count = 0
    context_chunks = []

    all_chunks = []
    for doc in docs:
        program = doc.metadata.get('program', 'Unknown Program')
        prefix = f"--- Source: {doc.metadata.get('source', 'unknown').upper()} | Program: {program} ---\n"

        # use paragraphs first, if too long, fallback to sentences
        paragraphs = [p for p in doc.page_content.split("\n\n") if p.strip()]
        for para in paragraphs:
            para_tokens = enc.encode(para.strip())
            if len(para_tokens) <= chunk_token_limit:
                all_chunks.append({
                    "text": prefix + para.strip(),
                    "token_len": len(para_tokens)
                })
            else:
                # split long paragraph into sentences
                sentences = sent_tokenize(para)
                current_chunk = ""
                current_len = 0
                for sent in sentences:
                    sent_tokens = enc.encode(sent)
                    if current_len + len(sent_tokens) > chunk_token_limit:
                        if current_chunk:
                            all_chunks.append({
                                "text": prefix + current_chunk.strip(),
                                "token_len": current_len
                            })
                        current_chunk = sent
                        current_len = len(sent_tokens)
                    else:
                        current_chunk += " " + sent
                        current_len += len(sent_tokens)
                if current_chunk:
                    all_chunks.append({
                        "text": prefix + current_chunk.strip(),
                        "token_len": current_len
                    })

    # round-robin interleaving
    from collections import deque
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

def auto_expand_abbreviations(query: str) -> str:
    words = query.split()
    expanded_words = [
        f"{w} ({ABBREVIATION_GLOSSARY[w.upper()]})" if w.upper() in ABBREVIATION_GLOSSARY else w
        for w in words
    ]
    return " ".join(expanded_words)

def compute_metadata_score(metadata):
    score = 0
    if "date" in metadata:
        try:
            doc_date = datetime.strptime(metadata["date"], "%Y-%m-%d")
            days_diff = (datetime.now() - doc_date).days
            score += max(0, (365 - days_diff) / 365)
        except:
            pass
    if "reliability" in metadata:
        score += float(metadata["reliability"]) * 0.1
    return score

def parallel_retrieve(query):
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_faiss = executor.submit(lambda q: lance_table.search(embedder.embed_query(q), vector_column_name="vector").limit(10).to_list(), query)
        future_bm25 = executor.submit(search_bm25, query, 10)
        faiss_results = future_faiss.result()
        bm25_results = future_bm25.result()
    return faiss_results, bm25_results

def rerank_semantic_docs(query, docs):
    """
    Rerank semantic-aware chunks using CrossEncoder and optional metadata scores.
    """
    seen_texts = set()
    unique_docs = []
    for doc in docs:
        txt = doc.page_content.strip().lower()
        if txt not in seen_texts:
            unique_docs.append(doc)
            seen_texts.add(txt)

    texts = [doc.page_content for doc in unique_docs]
    pairs = [(query, t) for t in texts]

    scores = cross_encoder.predict(pairs)  # [batch_size] float array

    scores = [s + compute_metadata_score(doc.metadata) for s, doc in zip(scores, unique_docs)]

    reranked_docs = [doc for _, doc in sorted(zip(scores, unique_docs), key=lambda x: x[0], reverse=True)]

    print("\nTop Reranked Semantic Chunks:")
    for i, doc in enumerate(reranked_docs[:5], start=1):
        snippet = doc.page_content[:200].replace("\n", " ")
        print(f"{i}. [Source: {doc.metadata.get('source', 'unknown')}] {snippet}...\n")

    return reranked_docs

#--------------------------------- Helpers for query_faiss ---------------------------------#

def query_faiss(query):
    if not query.strip():
        return "Please enter a question."

    total_start = time.time()

    print(f"❓ Original Query: {query}")

    step_start = time.time()
    expanded_query = expand_query_with_abbreviations(query)
    expanded_query = auto_expand_abbreviations(expanded_query)
    rewritten = rewrite_chain.invoke({"query": expanded_query})
    print(f"✏️ Rewritten for LLM (with expansions): {rewritten}")
    print(f"⏱️ Query rewrite took {time.time() - step_start:.2f}s")

    faiss_results, bm25_results = parallel_retrieve(rewritten)
    faiss_docs = [Document(page_content=row.get("text", ""), metadata=row.get("metadata", {}) | {"source": "faiss"}) for
                  row in faiss_results]
    bm25_docs = [Document(page_content=doc["content"], metadata={"source": "bm25", **doc.get("metadata", {})}) for doc
                 in bm25_results]

    print("\nTop FAISS Docs:")
    for d in faiss_docs[:5]:
        print(d.page_content[:200], "...\n")

    combined_docs = faiss_docs + bm25_docs
    reranked_docs = rerank_semantic_docs(rewritten, combined_docs)

    context = truncate_context_to_token_limit(reranked_docs)

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
