"""
reranker.py (Pure ONNX Runtime - no torch, no transformers model)

Fast CrossEncoder reranking using ONNX Runtime.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (ONNX)
"""

from functools import lru_cache
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer 

from langchain_core.documents import Document
from django.conf import settings


# =========================
# SINGLETONS
# =========================

@lru_cache(maxsize=1)
def _get_session():
    return ort.InferenceSession(
        settings.RERANKER_ONNX_PATH,  # path to model.onnx
        providers=["CPUExecutionProvider"]
    )


@lru_cache(maxsize=1)
def _get_tokenizer():
    """
    Load fast tokenizer from tokenizer.json
    (export this once from HF)
    """
    return Tokenizer.from_file(settings.RERANKER_TOKENIZER_JSON_PATH)


# =========================
# DATA STRUCTURE
# =========================

@dataclass
class RankedChunk:
    explanation: str
    code: str
    symbol: str
    chunk_type: str
    file_path: str
    start_line: int
    end_line: int
    score: float
    one_line_summary: str = ""
    dependency_summaries: list[str] = None


# =========================
# TOKENIZATION HELPER
# =========================

def _encode_pairs(tokenizer, pairs, max_length=512):
    """
    Encode (query, passage) pairs into ONNX inputs.
    """
    input_ids = []
    attention_masks = []
    token_type_ids = []

    for query, passage in pairs:
        encoding = tokenizer.encode(query, passage)

        ids = encoding.ids[:max_length]
        mask = [1] * len(ids)
        types = encoding.type_ids[:max_length]

        # padding
        pad_len = max_length - len(ids)
        if pad_len > 0:
            ids += [0] * pad_len
            mask += [0] * pad_len
            types += [0] * pad_len

        input_ids.append(ids)
        attention_masks.append(mask)
        token_type_ids.append(types)

    return {
        "input_ids": np.array(input_ids, dtype=np.int64),
        "attention_mask": np.array(attention_masks, dtype=np.int64),
        "token_type_ids": np.array(token_type_ids, dtype=np.int64),
    }


# =========================
# CORE FUNCTION
# =========================

def rerank(query: str, documents: list[Document]) -> list[RankedChunk]:
    if not documents:
        return []

    session = _get_session()
    tokenizer = _get_tokenizer()

    pairs = [(query, doc.page_content) for doc in documents]

    inputs = _encode_pairs(tokenizer, pairs)

    outputs = session.run(None, inputs)

    # logits → (batch, 1)
    scores = outputs[0].squeeze(-1).tolist()

    ranked = [
        RankedChunk(
            explanation=doc.page_content,
            code=doc.metadata.get("code", ""),
            symbol=doc.metadata.get("symbol", "unknown"),
            chunk_type=doc.metadata.get("chunk_type", "unknown"),
            file_path=doc.metadata.get("file_path", ""),
            start_line=doc.metadata.get("start_line", 0),
            end_line=doc.metadata.get("end_line", 0),
            score=float(score),
            one_line_summary=doc.metadata.get("one_line_summary", ""),
            dependency_summaries=doc.metadata.get("dependency_summaries", []),
        )
        for doc, score in zip(documents, scores)
    ]

    return sorted(ranked, key=lambda x: x.score, reverse=True)