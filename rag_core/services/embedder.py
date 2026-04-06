"""
embedder.py

Wraps HuggingFace sentence-transformers/all-MiniLM-L6-v2 into a
LangChain-compatible embedding object.

Model runs 100% locally — no API key needed.
"""

from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from django.conf import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Singleton HuggingFaceEmbeddings instance.
    Cached with lru_cache so the model is loaded from disk only once
    per process — avoids repeated 80 MB model loads.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )