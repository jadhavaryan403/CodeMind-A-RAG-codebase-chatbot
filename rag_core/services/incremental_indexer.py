"""
incremental_indexer.py

Handles incremental re-indexing of a project.

Only processes chunks that are new or changed.
Unchanged chunks are skipped entirely — no LLM calls, no re-embedding.
Deleted chunks are removed from FAISS.
"""

import hashlib
import numpy as np
from pathlib import Path

from django.conf import settings

from rag_core.models import Project, ChunkIndex
from rag_core.services.ast_chunker import CodeChunk, chunk_file
from rag_core.services.explainer import generate_explanations_batch
from rag_core.services.embedder import get_embeddings
from rag_core.services.faiss_store import load_index, save_index, _store_dir


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_hash(code_text: str) -> str:
    """SHA256 hash of the chunk's source code."""
    return hashlib.sha256(code_text.encode('utf-8')).hexdigest()


def chunk_all_files_with_paths(
    file_path_pairs: list[tuple[str, str]]   # [(disk_path, original_path), ...]
) -> list[CodeChunk]:
    """
    Chunk all files and attach original_path to each chunk.
    original_path is the stable GitHub path (e.g. 'app.py', 'myapp/views.py').
    """
    all_chunks = []
    for disk_path, original_path in file_path_pairs:
        try:
            chunks = chunk_file(disk_path)
            for chunk in chunks:
                chunk.original_path = original_path
            all_chunks.extend(chunks)
        except Exception:
            continue
    return all_chunks


# ── Diff logic ────────────────────────────────────────────────────────────────

def diff_chunks(
    project:    Project,
    new_chunks: list[CodeChunk],
) -> tuple[list[CodeChunk], list[CodeChunk], list[ChunkIndex]]:

    # Build new lookup — key is (symbol, original_path)
    new_lookup = {}
    for c in new_chunks:
        orig = getattr(c, 'original_path', '') or c.file_path
        key  = (c.symbol_name, orig)
        new_lookup[key] = c

    # Build stored lookup from DB
    stored_rows   = ChunkIndex.objects.filter(project=project)
    stored_lookup = {}
    for row in stored_rows:
        orig = row.original_path or row.file_path
        key  = (row.symbol, orig)
        stored_lookup[key] = row

    # ── Debug print — remove after confirming it works ────────────────────────
    print("\n=== DIFF DEBUG ===")
    print(f"New chunks    : {len(new_chunks)}")
    print(f"Stored rows   : {stored_rows.count()}")
    print(f"New keys      : {list(new_lookup.keys())}")
    print(f"Stored keys   : {list(stored_lookup.keys())}")

    added_chunks   = []
    changed_chunks = []
    deleted_rows   = []

    for key, chunk in new_lookup.items():
        new_hash = compute_hash(chunk.code_text)
        stored   = stored_lookup.get(key)

        if stored is None:
            print(f"  ADDED  : {key}")
            added_chunks.append(chunk)
        elif stored.code_hash != new_hash:
            print(f"  CHANGED: {key}  old={stored.code_hash[:8]} new={new_hash[:8]}")
            changed_chunks.append(chunk)
        else:
            print(f"  SAME   : {key}")

    for key, row in stored_lookup.items():
        if key not in new_lookup:
            print(f"  DELETED: {key}")
            deleted_rows.append(row)

    print("=== END DIFF ===\n")

    return added_chunks, changed_chunks, deleted_rows


# ── FAISS operations ──────────────────────────────────────────────────────────

def remove_from_faiss(
    store,
    faiss_ids: list[int],
) -> object:
    """
    Remove vectors at the given FAISS positions from the index.
    FAISS doesn't support true deletion so we rebuild without those IDs.
    Returns the updated store.
    """
    if not faiss_ids or not hasattr(store, 'index_to_docstore_id'):
        return store

    ids_to_remove = set(faiss_ids)

    # Collect all documents that should be kept
    kept_docs         = []
    kept_explanations = []

    for idx, doc_id in store.index_to_docstore_id.items():
        if idx in ids_to_remove:
            continue
        doc = store.docstore.search(doc_id)
        if doc:
            kept_docs.append(doc)
            kept_explanations.append(doc.page_content)

    if not kept_docs:
        return store

    # Rebuild FAISS with only the kept documents
    embeddings = get_embeddings()
    new_store  = type(store).from_documents(kept_docs, embeddings)
    return new_store


def add_to_faiss(
    store,
    chunks:       list[CodeChunk],
    explanations: list[str],
) -> tuple[object, list[int]]:
    """
    Add new chunks to the FAISS store.

    Returns:
        updated store
        list of new FAISS IDs assigned to each chunk
    """
    from langchain_core.documents import Document

    symbol_to_summary = {
        chunk.symbol_name: exp["one_line_summary"]
        for chunk, exp in zip(chunks, explanations)
    }

    documents = []

    for chunk, explanation in zip(chunks, explanations):

        deps = explanation["dependencies"]

        dep_summaries = [
            f"{d}: {symbol_to_summary[d]}"
            for d in deps
            if d in symbol_to_summary
        ]

        documents.append(
            Document(
                page_content=explanation["detailed_explanation"],
                metadata={
                    "code": chunk.code_text,
                    "symbol": chunk.symbol_name,
                    "chunk_type": chunk.chunk_type,
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,

                    # 🔥 core additions
                    "one_line_summary": explanation["one_line_summary"],
                    "dependencies": deps,
                    "dependency_summaries": dep_summaries,
                },
            )
        )

    # Get the current max index before adding
    start_id = store.index.ntotal if hasattr(store, 'index') else 0

    store.add_documents(documents)

    # New IDs are sequential from start_id
    new_ids = list(range(start_id, start_id + len(documents)))
    return store, new_ids


# ── Main entry point ──────────────────────────────────────────────────────────

class IncrementalIndexResult:
    """Summary of what happened during incremental re-indexing."""
    def __init__(self):
        self.added_count   = 0
        self.changed_count = 0
        self.deleted_count = 0
        self.skipped_count = 0

    def __str__(self):
        return (
            f"Added: {self.added_count} | "
            f"Changed: {self.changed_count} | "
            f"Deleted: {self.deleted_count} | "
            f"Skipped (unchanged): {self.skipped_count}"
        )


def incremental_reindex(
    project:     Project,
    file_pairs: list[tuple[str, str]],
    user_id:     int,
) -> IncrementalIndexResult:
    """
    Incrementally re-index a project.

    Only processes chunks that are new or changed.
    Unchanged chunks are skipped — no LLM calls, no re-embedding.
    Deleted chunks are removed from FAISS.

    Args:
        project:    the Project instance
        file_paths: list of absolute paths to the latest source files
        user_id:    the owner's user ID (used to locate the FAISS store)

    Returns:
        IncrementalIndexResult summary
    """
    result = IncrementalIndexResult()

    # ── Step 1: Chunk with original paths attached ────────────────────────────
    new_chunks = chunk_all_files_with_paths(file_pairs)
    if not new_chunks:
        raise ValueError("No parseable chunks found.")

    # ── Step 2: Diff against stored chunks ───────────────────────────────────
    added_chunks, changed_chunks, deleted_rows = diff_chunks(project, new_chunks)

    result.added_count   = len(added_chunks)
    result.changed_count = len(changed_chunks)
    result.deleted_count = len(deleted_rows)
    result.skipped_count = (
        len(new_chunks) - len(added_chunks) - len(changed_chunks)
    )

    # Nothing to do
    if not added_chunks and not changed_chunks and not deleted_rows:
        return result

    # ── Step 3: Load current FAISS store ─────────────────────────────────────
    store = load_index(user_id, project.pk)

    # ── Step 4: Remove deleted chunks from FAISS ─────────────────────────────
    if deleted_rows:
        faiss_ids_to_remove = [row.faiss_id for row in deleted_rows]
        store = remove_from_faiss(store, faiss_ids_to_remove)

        # Delete from DB
        ChunkIndex.objects.filter(
            id__in=[row.id for row in deleted_rows]
        ).delete()

    # ── Step 5: Remove changed chunks from FAISS (will re-add below) ─────────
    if changed_chunks:
        changed_symbols = {(c.symbol_name, c.file_path) for c in changed_chunks}
        rows_to_replace = ChunkIndex.objects.filter(
            project=project,
            symbol__in=[s for s, _ in changed_symbols],
        )
        faiss_ids_to_remove = [row.faiss_id for row in rows_to_replace]
        store = remove_from_faiss(store, faiss_ids_to_remove)
        rows_to_replace.delete()

    # ── Step 6: Generate explanations for new + changed chunks ───────────────
    chunks_to_add = added_chunks + changed_chunks
    explanations  = []

    explanations = generate_explanations_batch(chunks_to_add)

    # ── Step 7: Add new + changed chunks to FAISS ────────────────────────────
    store, new_faiss_ids = add_to_faiss(store, chunks_to_add, explanations)

    # ── Step 8: Save updated FAISS store to disk ─────────────────────────────
    save_index(store, user_id, project.pk)

    # ── Step 9: Save new ChunkIndex rows to DB ───────────────────────────────
    chunk_index_rows = [
        ChunkIndex(
            project=project,
            symbol=chunk.symbol_name,
            file_path=chunk.file_path,
            chunk_type=chunk.chunk_type,
            code_hash=compute_hash(chunk.code_text),
            faiss_id=faiss_id,
            explanation=explanation["detailed_explanation"],
            start_line=chunk.start_line,
            end_line=chunk.end_line,
        )
        for chunk, explanation, faiss_id in zip(
            chunks_to_add, explanations, new_faiss_ids
        )
    ]
    ChunkIndex.objects.bulk_create(chunk_index_rows)

    return result