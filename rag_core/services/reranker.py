"""
reranker.py

CrossEncoder reranking of retrieved FAISS chunks.

The CrossEncoder jointly encodes (query, passage) pairs and produces
relevance scores — far more accurate than bi-encoder cosine similarity alone.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (local, no API key needed)
"""

from functools import lru_cache
from dataclasses import dataclass

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from django.conf import settings


@lru_cache(maxsize=1)
def _get_cross_encoder() -> CrossEncoder:
    """Singleton CrossEncoder — loaded from disk once per process."""
    return CrossEncoder(settings.RERANKER_MODEL_NAME)


@dataclass
class RankedChunk:
    explanation: str      # What we embedded (the FAISS page_content)
    code: str             # Original source code (from metadata)
    symbol: str           # Function/class name
    chunk_type: str       # "function" | "class" | "method"
    file_path: str
    start_line: int
    end_line: int
    score: float          # CrossEncoder relevance score (higher = more relevant)


def rerank(query: str, documents: list[Document]) -> list[RankedChunk]:
    """
    Rerank `documents` against `query` using the CrossEncoder.

    Args:
        query:     The user's natural language question.
        documents: FAISS retrieval results (Document objects).

    Returns:
        List of RankedChunk sorted by descending relevance score.
    """
    if not documents:
        return []

    encoder = _get_cross_encoder()

    # Build (query, explanation) pairs for joint scoring
    pairs = [(query, doc.page_content) for doc in documents]
    scores: list[float] = encoder.predict(pairs).tolist()

    ranked = [
        RankedChunk(
            explanation=doc.page_content,
            code=doc.metadata.get("code", ""),
            symbol=doc.metadata.get("symbol", "unknown"),
            chunk_type=doc.metadata.get("chunk_type", "unknown"),
            file_path=doc.metadata.get("file_path", ""),
            start_line=doc.metadata.get("start_line", 0),
            end_line=doc.metadata.get("end_line", 0),
            score=score,
        )
        for doc, score in zip(documents, scores)
    ]

    return sorted(ranked, key=lambda c: c.score, reverse=True)