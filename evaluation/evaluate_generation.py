import os
import sys
import json
import django
from dotenv import load_dotenv

# -----------------------------
# ✅ Setup (VERY IMPORTANT)
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "codebase_assistant.settings")
django.setup()

load_dotenv()

# -----------------------------
# Imports (after setup)
# -----------------------------
from langchain_groq import ChatGroq
from rag_core.services.faiss_store import load_index
from eval_utils import run_pipeline_once  # reuse your pipeline if needed

from rag_core.langchain.nodes import get_faiss_ids_by_symbols
from rag_core.services.faiss_store import get_docs_by_faiss_ids


# -----------------------------
# CONFIG
# -----------------------------
INPUT_FILE = os.path.join(os.path.dirname(__file__), "results_pipeline.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "results_generation.json")

USER_ID = 3
PROJECT_ID = 22

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key,
    temperature=0
)

# -----------------------------
# Build Context from FAISS docs
# -----------------------------
def build_context(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        code = d.page_content
        symbol = d.metadata.get("symbol", "unknown")

        parts.append(
            f"Chunk {i}: ({symbol})\n```python\n{code}\n```"
        )

    return "\n\n".join(parts)


# -----------------------------
# LLM Judge
# -----------------------------
def evaluate_answer(query, context, answer):
    prompt = f"""
You are evaluating a RAG system.

Query:
{query}

Context:
{context}

Answer:
{answer}

Evaluate:

1. correctness (1-5)
2. grounding (1-5) → is answer supported by context
3. completeness (1-5)

Also return:
- hallucination: true/false

Return ONLY JSON:
{{
  "correctness": int,
  "grounding": int,
  "completeness": int,
  "hallucination": bool
}}
"""

    response = llm.invoke(prompt)

    try:
        return json.loads(response.content)
    except:
        return {
            "correctness": 0,
            "grounding": 0,
            "completeness": 0,
            "hallucination": True
        }


# -----------------------------
# Main
# -----------------------------
def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    results = data["results"]

    # 🔥 Load FAISS once
    store = load_index(USER_ID, PROJECT_ID)

    eval_results = []

    for i in range(0 , len(results)):
        item = results[i]
        query = item["query"]
        answer = item["answer"]
        retrieved_symbols = item["retrieved_chunks"]

        print(f"Evaluating {i+1}/{len(results)}")

        try:
            # 🔥 Step 1: symbols → FAISS ids
            faiss_ids = get_faiss_ids_by_symbols(PROJECT_ID, retrieved_symbols)

            # 🔥 Step 2: ids → docs
            docs = get_docs_by_faiss_ids(store, faiss_ids)

            # 🔥 Step 3: build context
            context = build_context(docs)

            # 🔥 Step 4: evaluate
            evaluation = evaluate_answer(query, context, answer)

        except Exception as e:
            print("Error:", e)
            continue

        eval_results.append({
            "query": query,
            "retrieved_chunks": retrieved_symbols,
            "answer": answer,
            "evaluation": evaluation
        })

    # -----------------------------
    # Summary
    # -----------------------------
    total = len(eval_results)

    avg_correctness = sum(r["evaluation"]["correctness"] for r in eval_results) / total
    avg_grounding = sum(r["evaluation"]["grounding"] for r in eval_results) / total
    avg_completeness = sum(r["evaluation"]["completeness"] for r in eval_results) / total
    hallucination_rate = sum(r["evaluation"]["hallucination"] for r in eval_results) / total

    output = {
        "summary": {
            "total_queries": total,
            "avg_correctness": round(avg_correctness, 2),
            "avg_grounding": round(avg_grounding, 2),
            "avg_completeness": round(avg_completeness, 2),
            "hallucination_rate": round(hallucination_rate, 2)
        },
        "results": eval_results
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print("\n✅ Generation evaluation complete!")
    print(output["summary"])


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    main()