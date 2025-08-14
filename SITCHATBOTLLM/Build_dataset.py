import json
import random
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Load your API key
load_dotenv()

INPUT_FILE = "all_batches_cleaned.jsonl"
OUTPUT_FILE = "benchmark_dataset.json"
TARGET_SIZE = 400  # number of benchmark Q&A

# Step 1: Load cleaned content from JSONL
raw_data = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        try:
            item = json.loads(line)
            if item.get("content"):
                raw_data.append(item)
        except json.JSONDecodeError:
            continue

print(f"Loaded {len(raw_data)} items from {INPUT_FILE}")

# Step 2: Randomly sample entries for variety
sampled_data = random.sample(raw_data, min(TARGET_SIZE, len(raw_data)))

# Step 3: Initialize LLM for Q/A generation
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.3)

def generate_qa(text, url):
    prompt = f"""
You are creating a benchmark dataset for Singapore Institute of Technology (SIT) FAQs.

Given the following official content:
\"\"\"{text}\"\"\"

1. Write ONE natural-sounding question a student might ask based on this content.
2. Provide a short, factually correct answer (1-2 sentences max) as the ground truth.
3. Suggest a category from: Admissions, Programmes, Fees & Scholarships, Campus Life, Academic Policies, Contact & Location, Research.

Respond in JSON like:
{{
  "query": "...",
  "expected_answer": "...",
  "category": "...",
  "metadata": {{"source_url": "..."}}
}}
"""
    resp = llm.invoke(prompt)
    try:
        data = json.loads(resp.content)
        # Ensure URL is attached
        data["metadata"]["source_url"] = url
        return data
    except Exception:
        return None

# Step 4: Build dataset
benchmark_data = []
for item in tqdm(sampled_data, desc="Generating Q&A"):
    qa = generate_qa(item["content"], item.get("url", ""))
    if qa:
        benchmark_data.append(qa)

# Step 5: Save to file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

print(f"Saved benchmark dataset with {len(benchmark_data)} entries to {OUTPUT_FILE}")