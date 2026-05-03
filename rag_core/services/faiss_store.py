"""
faiss_store.py

Per-user, per-project FAISS vector store management.

Directory layout (relative to settings.VECTORSTORE_ROOT):
    user_<user_id>/project_<project_id>/index.faiss
                                        index.pkl

All paths are derived from integer PKs only — never from user-supplied strings.
"""

from pathlib import Path
from typing import Optional
import pickle
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from django.conf import settings

from rag_core.services.embedder import get_embeddings
from rag_core.services.ast_chunker import CodeChunk


# ── Path helpers ──────────────────────────────────────────────────────────────

def _store_dir(user_id: int, project_id: int) -> Path:
    """Return (and create) the FAISS directory for this user/project pair."""
    path = settings.VECTORSTORE_ROOT / f"user_{user_id}" / f"project_{project_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── Build & Save ──────────────────────────────────────────────────────────────

def load_symbol_index(path):
    '''Helper to load the symbol index (pickle) for a given project.'''
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)

def get_docs_by_faiss_ids(store, ids: list[int]):
    '''Helper to retrieve Documents from the FAISS store given a list of FAISS IDs.'''
    docs = []

    for i in ids:
        try:
            doc_id = store.index_to_docstore_id[i]   # 🔥 correct mapping
            doc = store.docstore.search(doc_id)
            docs.append(doc)
        except Exception as e:
            print(f"[FAISS FETCH ERROR] id={i} -> {e}")

    return docs

def build_and_save_index(
    user_id:      int,
    project_id:   int,
    chunks:       list[CodeChunk],
    explanations: list[dict],
    usages:       list[dict],
    project=None,
) -> None:
    '''Builds a FAISS index from code chunks and their explanations, then saves it to disk.'''

    if not chunks:
        raise ValueError("Cannot build FAISS index with zero chunks.")

    def normalize(name: str) -> str:
        return name.split(".")[-1]

    symbol_to_summary = {
        normalize(chunk.symbol_name): exp["one_line_summary"]
        for chunk, exp in zip(chunks, explanations)
    }

    print(f"Symbol to summary: {symbol_to_summary}")

    documents = []
    symbol_index = {}   # ✅ NEW: symbol → Document mapping

    for chunk, explanation, usage in zip(chunks, explanations, usages):

        deps = explanation["dependencies"]
        dep_summaries = []

        for d in deps:
            key = normalize(d)

            if key in symbol_to_summary:
                dep_summaries.append(
                    f"{key}: {symbol_to_summary[key]}"
                )

        # ✅ Create document first
        doc = Document(
            page_content=explanation["detailed_explanation"],
            metadata={
                "code": chunk.code_text,
                "symbol": chunk.symbol_name,
                "chunk_type": chunk.chunk_type,
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "one_line_summary": explanation["one_line_summary"],
                "dependencies": deps,
                "dependency_summaries": dep_summaries,
                "input_tokens": usage["prompt_tokens"] if usage else 0,
                "output_tokens": usage["completion_tokens"] if usage else 0,
            },
        )

        # ✅ Store in both places
        documents.append(doc)
        symbol_index[chunk.symbol_name] = doc

        print(f"Dependency summaries for {chunk.symbol_name}: {dep_summaries}")

    # ✅ Build FAISS index
    store = FAISS.from_documents(documents, get_embeddings())

    index_path = str(_store_dir(user_id, project_id))
    store.save_local(index_path)

    # ✅ Save symbol index (pickle)
    with open(os.path.join(index_path, "symbol_index.pkl"), "wb") as f:
        pickle.dump(symbol_index, f)

    # ---------------- DB PART (unchanged) ---------------- #

    if project is not None:
        from rag_core.models import ChunkIndex
        from rag_core.services.incremental_indexer import compute_hash

        ChunkIndex.objects.filter(project=project).delete()

        rows = [
            ChunkIndex(
                project=project,
                symbol=chunk.symbol_name,
                file_path=chunk.file_path,
                original_path=getattr(chunk, 'original_path', '') or chunk.file_path,
                chunk_type=chunk.chunk_type,
                code_hash=compute_hash(chunk.code_text),
                faiss_id=i,
                explanation=explanation["detailed_explanation"],
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                input_tokens=usage["prompt_tokens"] if usage else 0,
                output_tokens=usage["completion_tokens"] if usage else 0,
            )
            for i, (chunk, explanation, usage) in enumerate(zip(chunks, explanations, usages))
        ]

        ChunkIndex.objects.bulk_create(rows)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_index(user_id: int, project_id: int) -> Optional[FAISS]:
    """
    Load an existing FAISS index from disk.
    Returns None if the index has not been built yet.
    """
    store_dir  = _store_dir(user_id, project_id)
    faiss_file = store_dir / "index.faiss"
    if not faiss_file.exists():
        return None
    return FAISS.load_local(
        str(store_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


# ── NEW: Save ─────────────────────────────────────────────────────────────────

def save_index(store: FAISS, user_id: int, project_id: int) -> None:
    """
    Persist an in-memory FAISS store to disk.
    Used by the incremental indexer after modifying the store.
    """
    store.save_local(str(_store_dir(user_id, project_id)))


# ── Query ─────────────────────────────────────────────────────────────────────

def query_index(
    user_id:    int,
    project_id: int,
    query:      str,
    top_k:      int | None = None,
) -> list[Document]:
    """
    Retrieve the top-k most similar Documents for `query`.

    Returns:
        List of Document objects (page_content=explanation, metadata has code).
    Raises:
        ValueError if the index doesn't exist for this project.
    """
    if top_k is None:
        top_k = settings.FAISS_TOP_K

    store = load_index(user_id, project_id)
    if store is None:
        raise ValueError(
            f"No FAISS index found for user={user_id}, project={project_id}. "
            "Please index files first."
        )
    return store.similarity_search(query, k=top_k)

