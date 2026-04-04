"""
retriever.py — Query retrieval from FAISS index.

Embeds the user query, searches the document's FAISS index,
and returns top_k chunks with their cosine similarity scores.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from pipeline.embedder import embed_query
from pipeline.vector_store import load_index
from models import RetrievedChunk

logger = logging.getLogger(__name__)


def retrieve(
    storage_path: Path,
    doc_id: str,
    question: str,
    top_k: int = 4,
    model_name: str = "BAAI/bge-small-en",
) -> list[RetrievedChunk]:
    """
    Retrieve the top_k most relevant chunks for a question.

    Steps:
      1. Embed the query with instruction prefix "query: <text>"
      2. Search the doc's FAISS IndexFlatIP
      3. Return chunks sorted by descending similarity

    Returns:
        List of RetrievedChunk objects, sorted by similarity descending.
    """
    # Embed query
    query_vec = embed_query(question, model_name=model_name)  # shape (1, dim)

    # Load FAISS index + chunk metadata
    index, chunks_meta = load_index(storage_path, doc_id)

    # Adjust top_k to not exceed the number of indexed vectors
    k = min(top_k, index.ntotal)
    if k == 0:
        logger.warning("Empty index for doc_id=%s", doc_id)
        return []

    # Search — returns (1, k) arrays of distances and indices
    distances, indices = index.search(query_vec, k)

    # distances[0] are inner products = cosine similarities (since normalized)
    results: list[RetrievedChunk] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks_meta):
            continue  # FAISS can return -1 for unfilled slots
        meta = chunks_meta[idx]
        # Clamp similarity to [0, 1] — can be slightly > 1 due to float precision
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
