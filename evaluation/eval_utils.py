import os
import sys
import django
import json

# ✅ Fix Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# ✅ Initialize Django
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "codebase_assistant.settings"
)
django.setup()

# ✅ Now safe to import
from rag_core.langchain.nodes import stream_answer


def run_pipeline_once(query, user_id, project_id):
    """
    Runs stream_answer and extracts:
    - retrieved chunks
    - final answer
    - usage
    """

    retrieved_chunks = []
    full_text = ""
    usage = {}
    rewritten_query = None

    generator = stream_answer(
        query=query,
        user_id=user_id,
        project_id=project_id,
        chat_history=[]
    )

    for event in generator:
        data = json.loads(event.replace("data: ", "").strip())

        # metadata → contains retrieved chunks
        if data["type"] == "metadata":
            cited = data.get("cited_chunks", [])
            retrieved_chunks = [c["symbol"] for c in cited]

        # final output
        elif data["type"] == "done":
            full_text = data.get("full_text", "")
            usage = data.get("usage", {})

    return {
        "retrieved_chunks": retrieved_chunks,
        "answer": full_text,
        "usage": usage,
        "rewritten_query": rewritten_query  # optional (not exposed yet)
    }