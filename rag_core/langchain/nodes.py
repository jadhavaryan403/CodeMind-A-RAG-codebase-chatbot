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


TOP_K_LLM = 5  # Number of top reranked chunks to include in LLM context

# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class EvaluationResult(BaseModel):
    """
    Structured response from the evaluate_and_answer step.

    The LLM either:
      - Answers the question  → sufficient=True,  answer=<text>, rewritten_query=None
      - Rewrites the query    → sufficient=False, answer=None,   rewritten_query=<text>
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
            "Should use specific function names, class names, variable names, "
            "or technical keywords likely to appear in the source code. "
            "Must be None if sufficient is True."
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
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set in .env")
        _base_llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.3,
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
                "You are a helpful code assistant AND a strict retrieval quality judge.\n\n"

                "You will receive:\n"
                "  1. Conversation history (to understand references like 'it', 'that function')\n"
                "  2. Retrieved code chunks\n"
                "  3. The user's question\n\n"

                "IMPORTANT: You must FIRST identify the TARGET ENTITY.\n"
                "This could be a function, class, variable, or concept.\n"
                "If the question uses words like 'it', 'that', etc., resolve it using conversation history.\n\n"

                "Then you MUST verify:\n"
                "Do the retrieved chunks actually refer to THIS SAME entity?\n\n"

                "STRICT RULES:\n"
                "1. If retrieved chunks are about a DIFFERENT function/class/entity:\n"
                "   → sufficient = false\n"
                "   → answer = null\n"
                "   → rewritten_query = a better query using the CORRECT entity name\n\n"

                "2. If retrieved chunks are PARTIALLY relevant but missing key details:\n"
                "   → sufficient = false\n"
                "   → answer = null\n"
                "   → rewritten_query = more specific query\n\n"

                "3. ONLY if the chunks clearly contain information about the CORRECT entity:\n"
                "   → sufficient = true\n"
                "   → answer = full answer grounded in the provided code\n"
                "   → rewritten_query = null\n\n"

                "4. NEVER answer using unrelated chunks.\n"
                "   If the entity in the question does NOT match the entity in the context,\n"
                "   you MUST mark sufficient=false.\n\n"

                "5. The answer MUST be grounded in the retrieved code.\n"
                "   If you cannot point to relevant code, mark sufficient=false.\n\n"

                "REWRITING RULES:\n"
                "When rewriting, include:\n"
                "  - exact function/class name\n"
                "  - relevant keywords like 'return value', 'implementation', etc.\n"
                "  - terms likely to appear in source code explanation\n\n"

                "EXAMPLE:\n"
                "Conversation:\n"
                "User: is register_view implemented?\n"
                "User: what does it return?\n\n"
                "Retrieved chunks: about get_bert_embedding\n\n"
                "→ These are NOT relevant\n\n"
                "Output:\n"
                "sufficient = false\n"
                "answer = null\n"
                "rewritten_query = \"register_view return value Django\"\n\n"

                "Format any code in answers with triple backticks."
                ),
                ("human",
                "Conversation history:\n{history}\n\n"
                "Retrieved code chunks:\n{context}\n\n"
                "Question: {question}"),
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
             "Answer the question using the provided context as best you can. "
             "If the context is limited, say so and answer with what is available. "
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
    return rerank(query, docs)


def build_context(ranked: list) -> str:
    """Format top 3 reranked chunks into a context string for the LLM."""
    parts = []
    for i, c in enumerate(ranked[:TOP_K_LLM], 1):
        parts.append(
            f"[{i}] {c.symbol} L{c.start_line}–{c.end_line})\n"
            f"Code:\n```python\n{c.code}\n```"
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
            "score":      c.score,
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
        result: EvaluationResult = get_evaluator_chain().invoke({
            "history":  build_history_text(chat_history),
            "context":  build_context(ranked),
            "question": query,
        })
        return result
    except Exception as e:
        return EvaluationResult(
            sufficient=True,
            answer=f"Error evaluating context: {e}",
            rewritten_query=None,
        )


def force_answer(
    query:          str,
    ranked:         list,
    chat_history:   list[dict],
    original_query: str,
    retry_count:    int,
) -> str:
    """Called when max retries exhausted. Answers with best available context."""
    try:
        response = get_force_answer_chain().invoke({
            "history":     build_history_text(chat_history),
            "context":     build_context(ranked),
            "question":    original_query,
            "retry_count": retry_count,
        })
        return response.content
    except Exception as e:
        return f"Could not generate answer after {retry_count} attempts: {e}"


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

MAX_RETRIES = 2


def stream_answer(
    query:        str,
    user_id:      int,
    project_id:   int,
    chat_history: list[dict] = [],
):
    """
    Full adaptive RAG pipeline. Yields SSE strings to the frontend.

    Flow:
        retrieve → rerank → evaluate_and_answer
                                 ↓              ↓
                            sufficient?    rewritten_query
                            stream answer  → retrieve again
    """
    original_query  = query
    current_query   = query       # updated by LLM on each retry
    retry_count     = 0
    sufficient      = False
    rewrite_reason  = ""
    ranked          = []
    full_text       = ""

    # ── Adaptive retrieval loop ───────────────────────────────────────────────
    while True:

        # Step 1: Retrieve using current query
        try:
            documents = retrieve(current_query, user_id, project_id)
        except ValueError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        # Step 2: Rerank
        ranked = rerank_docs(current_query, documents)

        # Step 3: Send cited chunks to frontend
        cited = build_cited_chunks(ranked)
        yield f"data: {json.dumps({'type': 'metadata', 'cited_chunks': cited})}\n\n"

        # Step 4: Single LLM call — evaluate + answer or rewrite
        # Passes original user query + full history so LLM understands
        # follow-up questions without needing a separate resolution step
        result = evaluate_and_answer(
            query=original_query,     # always the original user question
            ranked=ranked,
            chat_history=chat_history,
            original_query=original_query,
            retry_count=retry_count,
        )

        if result.sufficient and result.answer:
            sufficient = True
            full_text  = result.answer
            break

        if retry_count >= MAX_RETRIES:
            # Max retries hit — force an answer with best available context
            sufficient = True
            full_text  = force_answer(
                query=original_query,
                ranked=ranked,
                chat_history=chat_history,
                original_query=original_query,
                retry_count=retry_count,
            )
            break

        # Not sufficient — use LLM's rewritten query for next FAISS retrieval
        rewrite_reason = result.rewritten_query or ""
        current_query  = result.rewritten_query or f"{original_query} implementation"
        retry_count   += 1

    # ── Stream answer word by word ────────────────────────────────────────────
    for word in full_text.split(" "):
        yield f"data: {json.dumps({'type': 'token', 'text': word + ' '})}\n\n"

    # ── Print terminal summary ────────────────────────────────────────────────
    print_final_state(
        original_query=original_query,
        final_query=current_query,
        retry_count=retry_count,
        sufficient=sufficient,
        rewrite_reason=rewrite_reason,
        ranked=ranked,
        answer_length=len(full_text),
    )

    yield f"data: {json.dumps({'type': 'done', 'full_text': full_text})}\n\n"


