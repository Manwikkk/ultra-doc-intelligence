"""
vector_store.py — Per-document FAISS index management.

Each document gets its own IndexFlatIP (inner product = cosine sim on
normalized vectors). Indexes and chunk metadata are persisted to disk
under {storage_dir}/{doc_id}/.

Layout on disk:
  storage/
  └── {doc_id}/
      ├── index.faiss      # FAISS binary index
      └── metadata.json    # List of chunk dicts {text, page, chunk_index}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def _doc_dir(storage_path: Path, doc_id: str) -> Path:
    d = storage_path / doc_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_index(
    storage_path: Path,
    doc_id: str,
    embeddings: np.ndarray,
    chunks: list[dict],
) -> None:
    """
    Build a FAISS IndexFlatIP, add embeddings, and persist to disk.

    Args:
        storage_path: Root storage directory.
        doc_id: Unique document identifier.
        embeddings: float32 array of shape (N, dim), already L2-normalized.
        chunks: List of chunk metadata dicts.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    doc_dir = _doc_dir(storage_path, doc_id)
    faiss.write_index(index, str(doc_dir / "index.faiss"))

    with open(doc_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(
        "Saved FAISS index for doc_id=%s (%d vectors, dim=%d)",
        doc_id,
        index.ntotal,
        dim,
    )


def load_index(
    storage_path: Path,
    doc_id: str,
) -> tuple[faiss.Index, list[dict]]:
    """
    Load a previously saved FAISS index and chunk metadata from disk.

    Returns:
        (faiss_index, chunks_metadata_list)

    Raises:
        FileNotFoundError if doc_id has not been indexed.
    """
    doc_dir = storage_path / doc_id
    index_path = doc_dir / "index.faiss"
    meta_path = doc_dir / "metadata.json"

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"No index found for doc_id='{doc_id}'. "
            "Please upload the document first via POST /upload."
        )

    index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(
        "Loaded FAISS index for doc_id=%s (%d vectors)",
        doc_id,
        index.ntotal,
    )
    return index, chunks


def doc_exists(storage_path: Path, doc_id: str) -> bool:
    """Check whether a document's index exists."""
    doc_dir = storage_path / doc_id
    return (doc_dir / "index.faiss").exists() and (doc_dir / "metadata.json").exists()


def list_documents(storage_path: Path) -> list[str]:
    """Return all stored doc_ids."""
    if not storage_path.exists():
        return []
    return [
        d.name
        for d in storage_path.iterdir()
        if d.is_dir() and (d / "index.faiss").exists()
    ]
