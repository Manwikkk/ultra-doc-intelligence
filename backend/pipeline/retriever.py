"""
retriever.py — Query retrieval from FAISS index.

Accepts a pre-computed query embedding (float32, shape (1, dim)) from the
caller — no embedding model is loaded on the backend.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from pipeline.vector_store import load_index
from models import RetrievedChunk

logger = logging.getLogger(__name__)


def retrieve(
    storage_path: Path,
    doc_id: str,
    query_vector: np.ndarray,   # shape (1, dim), float32, already L2-normalized
    top_k: int = 4,
) -> list[RetrievedChunk]:
    """
    Retrieve the top_k most relevant chunks for a pre-computed query embedding.

    Steps:
      1. Load the doc's FAISS IndexFlatIP
      2. Search with the provided query_vector
      3. Return chunks sorted by descending cosine similarity

    Args:
        storage_path: Root storage directory.
        doc_id:       Unique document identifier.
        query_vector: float32 array of shape (1, dim), L2-normalized.
        top_k:        Number of results to return.

    Returns:
        List of RetrievedChunk objects, sorted by similarity descending.
    """
    if query_vector.ndim != 2 or query_vector.shape[0] != 1:
        raise ValueError(
            f"query_vector must have shape (1, dim), got {query_vector.shape}"
        )

    query_vector = query_vector.astype(np.float32)

    # Load FAISS index + chunk metadata
    index, chunks_meta = load_index(storage_path, doc_id)

    # Adjust top_k to not exceed the number of indexed vectors
    k = min(top_k, index.ntotal)
    if k == 0:
        logger.warning("Empty index for doc_id=%s", doc_id)
        return []

    # Search — returns (1, k) arrays of distances and indices
    distances, indices = index.search(query_vector, k)

    # distances[0] are inner products = cosine similarities (on normalized vecs)
    results: list[RetrievedChunk] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks_meta):
            continue  # FAISS can return -1 for unfilled slots
        meta = chunks_meta[idx]
        similarity = float(np.clip(dist, 0.0, 1.0))
        results.append(
            RetrievedChunk(
                text=meta["text"],
                page=meta.get("page"),
                chunk_index=meta["chunk_index"],
                similarity=similarity,
            )
        )

    # Sort descending by similarity (should already be sorted by FAISS)
    results.sort(key=lambda x: x.similarity, reverse=True)

    logger.info(
        "Retrieved %d chunks for doc_id=%s | top similarity=%.4f",
        len(results),
        doc_id,
        results[0].similarity if results else 0.0,
    )
    return results
