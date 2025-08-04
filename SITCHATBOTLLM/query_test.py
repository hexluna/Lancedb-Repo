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
    enc = tiktoken.get_encoding("cl100k_base")
    context = ""
    token_count = 0

    for i, doc in enumerate(docs):
        program = doc.metadata.get('program', 'Unknown Program')  # fallback if not present
        prefix = f"--- Source: {doc.metadata.get('source', 'unknown').upper()} | Program: {program} ---\n"

        raw_text = doc.page_content.strip()
        if len(raw_text) < 100:
            continue  # Skip too short

        tokens = enc.encode(raw_text)

        if len(tokens) == 0:
            print(f"⚠️ Skipped empty doc #{i + 1}")
            continue

        # Split long documents into smaller chunks
        start = 0
        while start < len(tokens):
            end = min(start + chunk_token_limit, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = prefix + enc.decode(chunk_tokens)
            chunk_len = len(chunk_tokens)

            if chunk_len > max_tokens:
                print(f"⚠️ Skipping doc #{i + 1} because one chunk exceeds token limit ({chunk_len} tokens).")
                break

            if token_count + chunk_len > max_tokens:
                print(f"🚫 Truncation reached at chunk of doc #{i + 1}. Total tokens: {token_count}")
                return context

            context += chunk_text + "\n\n"
            token_count += chunk_len
            start = end

    print(f"🧮 Final token count in context: {token_count}")
    return context


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
#------------------------------------ Use this to stream to elevenLabs ------------------------------------#
    # class TokenStreamer(BaseCallbackHandler):
    #     def __init__(self):
    #         self.first_token_reported = False
    #         self.retrieval_end_time = retrieval_end_time
    #
    #     def on_llm_new_token(self, token: str, **kwargs):
    #         if not self.first_token_reported:
    #             first_token_time = time.time()
    #             latency = first_token_time - self.retrieval_end_time
    #             print(f"\n First token streamed at: {time.strftime('%X')} (Latency after retrieval: {latency:.3f}s)\n")
    #             self.first_token_reported = True
    #         yield token
    #
    # streamer = TokenStreamer()
    #
    # model = ChatOpenAI(
    #     model_name="gpt-4.1-mini",
    #     temperature=0.2,
    #     streaming=True,
    #     callbacks=[streamer]
    # )
    # return model.stream(prompt_messages)
#---------------------------------------------------------------------------------------------------------------------------#

    # model = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.2)
    # return model.invoke(prompt_messages).content
import time  # Make sure this is imported at the top


def query_faiss(query):
    if not query.strip():
        return "Please enter a question."

    total_start = time.time()  # ⏱️ Start overall timing

    print(f"❓ Original Query: {query}")

    step_start = time.time()
    expanded_query = expand_query_with_abbreviations(query)
    rewritten = rewrite_chain.invoke({"query": expanded_query})
    print(f"✏️ Rewritten for LLM (with expansions): {rewritten}")
    print(f"⏱️ Query rewrite took {time.time() - step_start:.2f}s")

    bm25_keywords = rewritten.strip()
    print(f"🔑 Rewritten BM25 Query: {bm25_keywords}")

    bm25_priority = any(
        x.lower() in query.lower()
        for x in ["miao", "xiaoxiao", "lee", "benjamin", "soon", "chong", "li", "zhang", "prof", "dr."]
    )

    cleaned_query = preprocess_query(rewritten)
    sit_bias_keywords = " Singapore Institute of Technology SIT university campus students programs"
    cleaned_query += sit_bias_keywords
    if any(x in query.lower() for x in ["get to", "directions", "location", "where is"]):
        cleaned_query += " address map transportation MRT bus"

    print("🔍 Running FAISS vector search...")
    step_start = time.time()
    query_to_use = query if bm25_priority else cleaned_query
    query_vector = embedder.embed_query(query_to_use)
    results = lance_table.search(query_vector, vector_column_name="vector").limit(10).to_list()
    raw_faiss_docs = [
        Document(
            page_content=row.get("text", ""),
            metadata=row.get("metadata", {}) | {"source": "faiss"}
        )
        for row in results
    ]
    # raw_faiss_docs = vectorstore.similarity_search_by_vector(query_vector, k=10)
    print(f"⏱️ FAISS retrieval took {time.time() - step_start:.2f}s")

    step_start = time.time()
    deduped_faiss_docs = dedup_documents(raw_faiss_docs)
    for doc in deduped_faiss_docs:
        doc.metadata["source"] = "faiss"

    print("🔍 Running BM25 assist...")
    step_start = time.time()
    bm25_results = search_bm25(bm25_keywords, top_n=10)
    print(f"⏱️ BM25 search took {time.time() - step_start:.2f}s")

    step_start = time.time()
    bm25_docs = [
        Document(
            page_content=doc["content"],
            metadata={"source": "bm25", **doc.get("metadata", {})}
        )
        for doc in bm25_results
    ]
    print("\n🗂 Top BM25 docs preview:")
    for i, doc in enumerate(bm25_docs[:5]):
        print(f"{i + 1}. {doc.page_content[:200].strip()}...\n")

    desired_total = 20
    bm25_limit = 10
    deduped_faiss_docs = dedup_documents(raw_faiss_docs)
    for doc in deduped_faiss_docs:
        doc.metadata["source"] = "faiss"

    num_faiss = min(len(deduped_faiss_docs), desired_total)
    selected_faiss = deduped_faiss_docs[:num_faiss]

    seen_texts = set(doc.page_content.strip().lower() for doc in selected_faiss)
    bm25_dedup = []
    for doc in bm25_docs:
        content = doc.page_content.strip().lower()
        if content not in seen_texts and len(bm25_dedup) < (desired_total - num_faiss):
            bm25_dedup.append(doc)
            seen_texts.add(content)

    unique_docs = selected_faiss + bm25_dedup

    print(f"✅ Combined {len(unique_docs)} documents from FAISS + BM25.")
    faiss_count = sum(1 for d in unique_docs if d.metadata.get("source") == "faiss")
    bm25_count = sum(1 for d in unique_docs if d.metadata.get("source") == "bm25")
    print(f"📊 Doc breakdown — FAISS: {faiss_count}, BM25: {bm25_count}")
    print(f"✅ Combined {len(unique_docs)} unique documents from FAISS + BM25.")
    print(f"⏱️ Merge & dedup took {time.time() - step_start:.2f}s")

    context_build_end = time.time()
    context = truncate_context_to_token_limit(unique_docs)
    print(f"⏱️ Context building took {time.time() - context_build_end:.2f}s")

    print("\n🧠 Final context passed to LLM:\n", context)

    retrieval_end_time = time.time()
    response = generate_response_from_context(query, context, retrieval_end_time)
    print(f"⏱️ LLM response took {time.time() - retrieval_end_time:.2f}s")

    # step_start = time.time()
    # context = truncate_context_to_token_limit(unique_docs)
    # print(f"⏱️ Context building took {time.time() - step_start:.2f}s")
    #
    # print("\n🧠 Final context passed to LLM:\n", context)
    #
    # step_start = time.time()
    # response = generate_response_from_context(query, context)
    # print(f"⏱️ LLM response took {time.time() - step_start:.2f}s")

    print(f"✅ Total query time: {time.time() - total_start:.2f}s")
    return response

#-------------------------------------------- Use this to stream to elevenlabs --------------------------------------------#
# def query_faiss_stream(query):
#     if not query.strip():
#         yield "Please enter a question."
#         return
#
#     total_start = time.time()
#     print(f"Original Query: {query}")
#
#     expanded_query = expand_query_with_abbreviations(query)
#     rewritten = rewrite_chain.invoke({"query": expanded_query})
#     print(f"Rewritten for LLM (with expansions): {rewritten}")
#
#     cleaned_query = preprocess_query(rewritten)
#     sit_bias_keywords = " Singapore Institute of Technology SIT university campus students programs"
#     cleaned_query += sit_bias_keywords
#
#     if any(x in query.lower() for x in ["get to", "directions", "location", "where is"]):
#         cleaned_query += " address map transportation MRT bus"
#
#     query_vector = embedder.embed_query(cleaned_query)
#     results = lance_table.search(query_vector, vector_column_name="vector").limit(10).to_list()
#     raw_faiss_docs = [
#         Document(
#             page_content=row.get("text", ""),
#             metadata=row.get("metadata", {}) | {"source": "faiss"}
#         )
#         for row in results
#     ]
#     deduped_faiss_docs = dedup_documents(raw_faiss_docs)
#
#     bm25_results = search_bm25(rewritten, top_n=10)
#     bm25_docs = [
#         Document(
#             page_content=doc["content"],
#             metadata={"source": "bm25", **doc.get("metadata", {})}
#         )
#         for doc in bm25_results
#     ]
#
#     unique_docs = deduped_faiss_docs + bm25_docs
#     context = truncate_context_to_token_limit(unique_docs)
#
#     print("\nFinal context passed to LLM:\n", context)
#     retrieval_end_time = time.time()
#
#     # Yield tokens from LLM stream
#     for token in generate_response_from_context_stream(query, context, retrieval_end_time):
#         yield token
#--------------------------------------------------------------------------------------------------------------------------#
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
