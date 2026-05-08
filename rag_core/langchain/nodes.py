"""
nodes.py

Streaming Adaptive RAG pipeline.

Pipeline:
    retrieve → rerank → evaluate_and_answer
                             ↓              ↓
                        sufficient?    not sufficient + rewritten_query
                        stream answer  → retrieve again (max 2 retries)

No separate query resolution step.
evaluate_and_answer has full chat history so it understands
follow-up questions and rewrites the query correctly for FAISS.
"""

import json
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from django.conf import settings

from rag_core.services.faiss_store import query_index
from rag_core.services.reranker import rerank
from utils.llm_callback import TokenTrackingCallback
from rag_core.models import ChunkIndex

from langchain_groq import ChatGroq


TOP_K_LLM = 5  # Number of top reranked chunks to include in LLM context
MAX_RETRIES = 2

# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class EvaluationResult(BaseModel):
    """
    Structured response from the evaluate_and_answer step.

    The LLM can now do THREE things:

    1. Answer the question:
       → sufficient=True, answer=<text>

    2. Rewrite the query for better retrieval:
       → sufficient=False, rewritten_query=<text>

    3. Request dependency code (multi-hop retrieval):
       → sufficient=False, request_dependencies={...}
    """

    sufficient: bool = Field(
        description=(
            "True if the retrieved context contains enough information "
            "to answer the question accurately."
        )
    )

    answer: Optional[str] = Field(
        default=None,
        description=(
            "Full answer to the question if sufficient is True. "
            "Must be None if sufficient is False."
        )
    )

    rewritten_query: Optional[str] = Field(
        default=None,
        description=(
            "A better FAISS search query if sufficient is False. "
            "Must be None if sufficient is True."
        )
    )

    request_dependencies: Optional[dict] = Field(
        default=None,
        description=(
            "Request specific dependency implementations when the main function "
            "is present but its dependencies are needed.\n\n"
            "Json Format:\n"
            "symbol: <main function name>,\n"
            "dependencies: [dep1, dep2]\n"
            "Must be None if sufficient is True or rewritten_query is used."
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODEL SINGLETONS
# ══════════════════════════════════════════════════════════════════════════════

_base_llm           = None
_evaluator_chain    = None
_force_answer_chain = None


def _get_base_llm() -> ChatGoogleGenerativeAI:
    global _base_llm
    if _base_llm is None:
        # if not settings.GOOGLE_API_KEY:
        #     raise ValueError("GOOGLE_API_KEY is not set in .env")
        _base_llm = ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0.2,
        )
    return _base_llm


def get_evaluator_chain():
    """
    Chain that evaluates context and either answers or rewrites the query.
    Receives full chat history so it understands follow-up questions like
    "what does it return" without needing a separate resolution step.
    """
    global _evaluator_chain
    if _evaluator_chain is None:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
                "You are a strict code assistant AND a retrieval quality evaluator.\n\n"

                "You will receive:\n"
                "1. Conversation history (for resolving references like 'it', 'that function')\n"
                "2. Retrieved code chunks (with dependencies and summaries)\n"
                "3. The user's question\n\n"

                "STEP 1: IDENTIFY TARGET ENTITY\n"
                "Determine the exact target entity (function, class, or concept).\n"
                "Resolve ambiguous references using conversation history.\n\n"

                "STEP 2: VALIDATE RETRIEVED CONTEXT\n"
                "Check whether the retrieved chunks refer to the SAME entity.\n\n"

                "DECISION LOGIC:\n"

                "CASE 1 — WRONG ENTITY:\n"
                "If chunks are about a DIFFERENT function/class:\n"
                "→ sufficient = false\n"
                "→ answer = null\n"
                "→ rewritten_query = better query with CORRECT entity name\n"
                "→ request_dependencies = null\n\n"

                "CASE 2 — PARTIAL CONTEXT (NEEDS BETTER SEARCH):\n"
                "If chunks are incomplete or vague:\n"
                "→ sufficient = false\n"
                "→ answer = null\n"
                "→ rewritten_query = more specific query\n"
                "→ request_dependencies = null\n\n"

                "CASE 3 — NEED DEPENDENCY IMPLEMENTATION (MULTI-HOP):\n"
                "If the correct function is present BUT you need implementation\n"
                "of its dependencies to answer properly:\n"
                "→ sufficient = false\n"
                "→ answer = null\n"
                "→ rewritten_query = null\n"
                "→ request_dependencies = {{\n"
                "     \"symbol\": \"<main function>\",\n"
                "     \"dependencies\": [\"dep1\", \"dep2\"]\n"
                "  }}\n\n"

                "IMPORTANT:\n"
                "- Use this ONLY if the main function is correct but dependencies are missing\n"
                "- Only include dependencies explicitly needed to answer\n"
                "- Do NOT request unnecessary dependencies\n\n"

                "CASE 4 — SUFFICIENT CONTEXT:\n"
                "If the retrieved chunks clearly contain enough information:\n"
                "→ sufficient = true\n"
                "→ answer = full grounded answer\n"
                "→ rewritten_query = null\n"
                "→ request_dependencies = null\n\n"

                "STRICT RULES\n"
                "• NEVER answer using unrelated chunks\n"
                "• NEVER hallucinate missing code\n"
                "• ALWAYS ground answers in provided code\n"
                "• ONLY ONE of (answer, rewritten_query, request_dependencies) should be used\n\n"

                "REWRITING GUIDELINES\n"
                "When rewriting queries, include:\n"
                "- exact function/class name\n"
                "- keywords like 'implementation', 'return value'\n"
                "- terms likely to appear in code\n\n"

                "OUTPUT FORMAT\n"
                "Return a structured response with:\n"
                "- sufficient (boolean)\n"
                "- answer (string or null)\n"
                "- rewritten_query (string or null)\n"
                "- request_dependencies (object or null)\n\n"

                "Format any code in answers using triple backticks."
            ),

            ("human",
                "Conversation history:\n{history}\n\n"
                "Retrieved code chunks:\n{context}\n\n"
                "Question: {question}"
            ),
        ])
        _evaluator_chain = (
            prompt | _get_base_llm().with_structured_output(EvaluationResult)
        )
    return _evaluator_chain


def get_force_answer_chain():
    """
    Chain used when max retries are exhausted.
    Answers with whatever context is available — no structured output needed
    since we just want any answer at this point.
    """
    global _force_answer_chain
    if _force_answer_chain is None:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful code assistant. "
             "Answer the question only using the provided context as best you can. "
             "If the context is not sufficient, do your best to answer based on partial information, but do NOT hallucinate details. "
             "Say provided context is not sufficient if you truly have no information to answer. "
             "Format code with triple backticks."),
            ("human",
             "Conversation history:\n{history}\n\n"
             "Retrieved code context:\n{context}\n\n"
             "Question: {question}\n\n"
             "Note: This is the best available context after {retry_count} "
             "retrieval attempts."),
        ])
        _force_answer_chain = prompt | _get_base_llm()
    return _force_answer_chain


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE STEPS
# ══════════════════════════════════════════════════════════════════════════════

def retrieve(query: str, user_id: int, project_id: int) -> list:
    """Query FAISS and return a list of LangChain Documents."""
    return query_index(
        user_id=user_id,
        project_id=project_id,
        query=query,
    )


def rerank_docs(query: str, documents: list) -> list:
    """Rerank retrieved documents using CrossEncoder."""
    docs = [
        Document(page_content=doc.page_content, metadata=doc.metadata)
        for doc in documents
    ]
    return rerank(query, docs)[:TOP_K_LLM]  # Keep top K for LLM context


def build_context(ranked: list) -> str:
    """
    Build LLM context in format:

    Chunk 1:
    <code>

    Dependencies:
    dep1: summary
    dep2: summary
    """

    parts = []

    for i, c in enumerate(ranked[:TOP_K_LLM], 1):

        code = c.code
        dep_summaries = getattr(c, "dependency_summaries", [])
        one_liner = getattr(c, "one_line_summary", "")

        dep_block = ""
        if dep_summaries:
            dep_lines = "\n".join(dep_summaries[:5])
            dep_block = f"\n\nDependencies:\n{dep_lines}"

        parts.append(
            f"Chunk {i}: ({one_liner})"
            f"\n```python\n{code}\n```"
            f"{dep_block}"
        )

    return "\n\n".join(parts) if parts else "No relevant code found."



def build_cited_chunks(ranked: list) -> list[dict]:
    """Convert ranked results into the JSON structure the frontend expects."""
    return [
        {
            "symbol":     c.symbol,
            "chunk_type": c.chunk_type,
            "start_line": c.start_line,
            "end_line":   c.end_line,
            "score":      getattr(c, "score", None),
            "code":       c.code,
        }
        for c in ranked
    ]


def build_history_text(chat_history: list[dict]) -> str:
    """Format the last 3 turns of chat history into plain text for the prompt."""
    if not chat_history:
        return "No previous conversation."
    return "\n".join([
        f"{msg['role'].upper()}: {msg['content'][:300]}"
        for msg in chat_history[-6:]
    ])


def evaluate_and_answer(
    query:          str,
    ranked:         list,
    chat_history:   list[dict],
    original_query: str,
    retry_count:    int,
) -> EvaluationResult:
    """
    Single LLM call — evaluates context and either answers or rewrites query.

    The LLM receives full chat history so it can resolve follow-up questions
    like "what does it return" without any separate resolution step.

    Returns a guaranteed EvaluationResult object:
        .sufficient      — bool
        .answer          — str | None
        .rewritten_query — str | None
    """

    try:
        callback = TokenTrackingCallback()

        result: EvaluationResult = get_evaluator_chain().invoke(
            {
                "history": build_history_text(chat_history),
                "context": build_context(ranked),
                "question": query,
            },
            config={"callbacks": [callback]}
        )

        # Debug prints
        print(f'[context] :\n{build_context(ranked)}\n')

        # ✅ Token usage
        print("Total tokens:", callback.total_tokens)
        print("Prompt tokens:", callback.prompt_tokens)
        print("Completion tokens:", callback.completion_tokens)

        usage = {
            "prompt_tokens": callback.prompt_tokens,
            "completion_tokens": callback.completion_tokens,
        }

        return result, usage

    except Exception as e:
        return EvaluationResult(
            sufficient=True,
            answer=f"Error evaluating context: {e}",
            rewritten_query=None,
        ), {"prompt_tokens": 0, "completion_tokens": 0}


def force_answer(
    query:          str,
    ranked:         list,
    chat_history:   list[dict],
    original_query: str,
    retry_count:    int,
) -> str:
    """Called when max retries exhausted. Answers with best available context."""
    try:
        callback = TokenTrackingCallback()

        response = get_force_answer_chain().invoke(
            {
                "history": build_history_text(chat_history),
                "context": build_context(ranked),
                "question": original_query,
                "retry_count": retry_count,
            },
            config={"callbacks": [callback]}
        )

        # ✅ Token usage
        print("Total tokens:", callback.total_tokens)
        print("Prompt tokens:", callback.prompt_tokens)
        print("Completion tokens:", callback.completion_tokens)

        usage = {
            "prompt_tokens": callback.prompt_tokens,
            "completion_tokens": callback.completion_tokens,
        }

        return response.content, usage

    except Exception as e:
        return (
            f"Could not generate answer after {retry_count} attempts: {e}",
            {"prompt_tokens": 0, "completion_tokens": 0}
        )


def print_final_state(
    original_query: str,
    final_query:    str,
    retry_count:    int,
    sufficient:     bool,
    rewrite_reason: str,
    ranked:         list,
    answer_length:  int,
):
    """Print pipeline summary to terminal after answer is generated."""
    scores = [c.score for c in ranked]
    print("\n" + "=" * 60)
    print("FINAL STATE SUMMARY")
    print("=" * 60)
    print(f"Original Query    : {original_query}")
    print(f"Final Query       : {final_query}")
    print(f"Retries Taken     : {retry_count}")
    print(f"Context Sufficient: {sufficient}")
    if retry_count > 0:
        print(f"Rewrite Reason    : {rewrite_reason}")
    print(f"Chunks Used       : {len(ranked)}")
    if ranked:
        print(f"Top Chunk         : {ranked[0].symbol} "
              f"(score={scores[0]:.4f}  {ranked[0].file_path})")
    print(f"Answer Length     : {answer_length} chars")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# STREAM ANSWER — orchestrates all steps
# ══════════════════════════════════════════════════════════════════════════════


def get_faiss_ids_by_symbols(project, symbols):
    '''Helper to get FAISS IDs for a list of symbols by querying the ChunkIndex.'''
    from rag_core.models import ChunkIndex

    rows = []

    for sym in symbols:
        matches = ChunkIndex.objects.filter(
            project=project,
            symbol__iendswith=sym   # 🔥 handles class.method vs method
        )
        rows.extend(matches)

    return [r.faiss_id for r in rows]


def stream_answer(
    query:        str,
    user_id:      int,
    project_id:   int,
    chat_history: list[dict] = [],
):
    """
    Adaptive + Multi-hop RAG pipeline.

    Flow:
        retrieve → rerank → evaluate
                               ↓
        ┌───────────────┬───────────────┬────────────────────┐
        │ sufficient    │ rewrite       │ dependency request │
        │ → answer      │ → FAISS retry │ → symbol lookup    │
        └───────────────┴───────────────┴────────────────────┘
    """

    original_query  = query
    current_query   = query
    retry_count     = 0
    sufficient      = False
    rewrite_reason  = ""
    ranked          = []
    full_text       = ""
    result          = None  # track last result

    # prevent infinite dependency loops
    used_dependency_symbols = set()

    total_usage = {
    "prompt_tokens": 0,
    "completion_tokens": 0
    }   

    def normalize(name: str) -> str:
        return name.split(".")[-1]

    print('inside nodes.py stream_answer function')

    while True:

        # safety net to prevent infinite loops in edge cases where LLM keeps asking for dependencies or rewriting without improving retrieval
        if retry_count >= MAX_RETRIES:
            sufficient = True
            full_text ,usage  = force_answer(
                query=original_query,
                ranked=ranked,
                chat_history=chat_history,
                original_query=original_query,
                retry_count=retry_count,
            )
            total_usage["prompt_tokens"] += usage["prompt_tokens"]
            total_usage["completion_tokens"] += usage["completion_tokens"]
            break

        # ── Step 1: Retrieve ONLY when needed ─────────────────────────────
        if retry_count == 0 or (result and result.rewritten_query):
            try:
                documents = retrieve(current_query, user_id, project_id)
                print("Retrieved docs count:", len(documents))
            except ValueError as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                return

            ranked = rerank_docs(current_query, documents)

            if not ranked:
                yield f"data: {json.dumps({'type': 'metadata', 'message': 'No relevant code found.'})}\n\n"
                break

        # ── Step 2: Send chunks to frontend ───────────────────────────────
        cited = build_cited_chunks(ranked)
        yield f"data: {json.dumps({'type': 'metadata', 'cited_chunks': cited})}\n\n"

        # ── Step 3: Evaluate ──────────────────────────────────────────────
        result ,usage = evaluate_and_answer(
            query=original_query,
            ranked=ranked,
            chat_history=chat_history,
            original_query=original_query,
            retry_count=retry_count,
        )

        total_usage["prompt_tokens"] += usage["prompt_tokens"]
        total_usage["completion_tokens"] += usage["completion_tokens"]


        # ─────────────────────────────────────────────────────────────────
        # ✅ CASE 1: SUFFICIENT → FINAL ANSWER
        # ─────────────────────────────────────────────────────────────────
        if result.sufficient and result.answer:
            sufficient = True
            full_text  = result.answer
            break

        # ─────────────────────────────────────────────────────────────────
        # 🔥 CASE 2: MULTI-HOP DEPENDENCY REQUEST
        # ─────────────────────────────────────────────────────────────────
        if result.request_dependencies:
            deps = result.request_dependencies.get("dependencies", [])
            print(f"[MULTI-HOP] Requested deps: {deps}")

            new_deps = [d for d in deps if d not in used_dependency_symbols]

            if not new_deps:
                print("[MULTI-HOP] No new dependencies to fetch")

                # 🔥 STOP infinite loop → force answer
                full_text ,usage = force_answer(
                    query=original_query,
                    ranked=ranked,
                    chat_history=chat_history,
                    original_query=original_query,
                    retry_count=retry_count,
                )
                total_usage["prompt_tokens"] += usage["prompt_tokens"]
                total_usage["completion_tokens"] += usage["completion_tokens"]
                sufficient = True
                break

            used_dependency_symbols.update(new_deps)

            # Step 1: get FAISS ids
            faiss_ids = get_faiss_ids_by_symbols(project_id, new_deps)

            if not faiss_ids:
                print("[MULTI-HOP] No matching FAISS IDs found")

            # Step 2: load FAISS
            from rag_core.services.faiss_store import load_index, get_docs_by_faiss_ids
            store = load_index(user_id, project_id)

            # Step 3: fetch docs
            dep_docs = get_docs_by_faiss_ids(store, faiss_ids)

            print(f"[MULTI-HOP] Retrieved {len(dep_docs)} docs from FAISS")

            if not dep_docs:
                print("[MULTI-HOP] No docs found → forcing answer")

                full_text ,usage = force_answer(
                    query=original_query,
                    ranked=ranked,
                    chat_history=chat_history,
                    original_query=original_query,
                    retry_count=retry_count,
                )
                total_usage["prompt_tokens"] += usage["prompt_tokens"]
                total_usage["completion_tokens"] += usage["completion_tokens"]
                sufficient = True
                break

            # Step 4: rerank deps
            dep_ranked = rerank_docs(original_query, dep_docs)

            # 🔥 Step 5: PRIORITY MERGE (critical fix)
            ranked = dep_ranked + [
                c for c in ranked if c.symbol not in {d.symbol for d in dep_ranked}
            ]

            print(f"[MULTI-HOP] Added {len(dep_ranked)} dependency chunks")
            print(f"[MULTI-HOP] Ranked size after merge: {len(ranked)}")

            retry_count += 1
            continue

        # ─────────────────────────────────────────────────────────────────
        # 🔁 CASE 3: REWRITE QUERY
        # ─────────────────────────────────────────────────────────────────
        if result.rewritten_query:
            rewrite_reason = result.rewritten_query
            current_query  = result.rewritten_query
            retry_count   += 1

            print(f"[REWRITE] New query: {current_query}")
            continue
    

        # fallback safety
        retry_count += 1

    # ── Stream answer ─────────────────────────────────────────────────────
    for word in full_text.split(" "):
        yield f"data: {json.dumps({'type': 'token', 'text': word + ' '})}\n\n"

    # ── Debug summary ─────────────────────────────────────────────────────
    print_final_state(
        original_query=original_query,
        final_query=current_query,
        retry_count=retry_count,
        sufficient=sufficient,
        rewrite_reason=rewrite_reason,
        ranked=ranked,
        answer_length=len(full_text),
    )

    yield f"data: {json.dumps({'type': 'done',
                     'full_text': full_text, 
                     "usage": {
                        "prompt_tokens": total_usage["prompt_tokens"],
                        "completion_tokens": total_usage["completion_tokens"]
        }
    })}\n\n"

