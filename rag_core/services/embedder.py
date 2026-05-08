"""
embedder.py (Pure ONNX Runtime)

Custom LangChain-compatible embedding class using:
- sentence-transformers/all-MiniLM-L6-v2 (ONNX)
- NO torch
- NO transformers runtime
"""

from functools import lru_cache
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from django.conf import settings


# =========================
# SINGLETONS
# =========================

@lru_cache(maxsize=1)
def _get_session():
    return ort.InferenceSession(
        settings.EMBEDDING_ONNX_PATH,  # path to model.onnx
        providers=["CPUExecutionProvider"]
    )


@lru_cache(maxsize=1)
def _get_tokenizer():
    return Tokenizer.from_file(settings.EMBEDDING_TOKENIZER_PATH)


# =========================
# CORE EMBEDDING CLASS
# =========================

class ONNXEmbeddings:
    """
    Drop-in replacement for HuggingFaceEmbeddings.
    Compatible with LangChain vector stores.
    """

    def __init__(self, max_length: int = 256):
        self.session = _get_session()
        print([i.name for i in self.session.get_inputs()])
        self.tokenizer = _get_tokenizer()
        self.max_length = max_length

    def __call__(self, text: str):
        return self.embed_query(text)

    # -------------------------
    # TOKENIZATION
    # -------------------------
    def _encode(self, texts):
        input_ids = []
        attention_masks = []
        token_type_ids = []

        for text in texts:
            enc = self.tokenizer.encode(text)

            ids = enc.ids

            # ensure at least CLS + SEP
            if len(ids) == 0:
                ids = [101, 102]
            else:
                if ids[0] != 101:
                    ids = [101] + ids
                if ids[-1] != 102:
                    ids = ids + [102]

            ids = ids[:self.max_length]

            mask = [1] * len(ids)

            # 🔥 IMPORTANT: MiniLM expects token_type_ids
            types = [0] * len(ids)

            pad_len = self.max_length - len(ids)
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
            "token_type_ids": np.array(token_type_ids, dtype=np.int64),  # ✅ FIX
        }

    # -------------------------
    # MEAN POOLING
    # -------------------------
    def _mean_pooling(self, token_embeddings, attention_mask):
        mask = np.expand_dims(attention_mask, axis=-1)
        summed = (token_embeddings * mask).sum(axis=1)
        counts = mask.sum(axis=1)
        counts = np.clip(counts, 1e-9, None)

        return summed / counts

    # -------------------------
    # PUBLIC METHODS
    # -------------------------
    def embed_documents(self, texts):
        inputs = self._encode(texts)

        outputs = self.session.run(None, inputs)
        token_embeddings = outputs[0]

        embeddings = self._mean_pooling(
            token_embeddings,
            inputs["attention_mask"]
        )

        # Normalize (important for cosine similarity / FAISS)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-9, None)
        embeddings = embeddings / norms

        return embeddings.tolist()

    def embed_query(self, text):
        print("Embedding query:", text)

        vec = self.embed_documents([text])[0]

        print("Embedding shape:", len(vec))
        print("First 5 values:", vec[:5])

        return vec


# =========================
# SINGLETON ACCESSOR
# =========================

@lru_cache(maxsize=1)
def get_embeddings() -> ONNXEmbeddings:
    """
    Singleton embedding model (loaded once per process)
    """
    return ONNXEmbeddings()