"""
embedder.py — Sentence-transformers wrapper for BAAI/bge-small-en.

Instruction-formatted embeddings:
  - Queries  → "query: <text>"
  - Passages → "passage: <text>"

All vectors are L2-normalized for cosine similarity via FAISS inner product.
"""
from __future__ import annotations

import logging
import numpy as np
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    """Load and cache the embedding model (loaded once at startup)."""
    # Restrict PyTorch memory/threads to prevent Render OOM on 512MB tier
    import torch
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    
    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name, device="cpu")
    logger.info("Embedding model loaded successfully")
    return model


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row vector."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)  # avoid division by zero
    return vectors / norms


def embed_passages(texts: list[str], model_name: str = "BAAI/bge-small-en") -> np.ndarray:
    """
    Embed a list of document passages.
    Prefix: "passage: <text>"
    Returns: float32 array of shape (N, dim), L2-normalized.
    """
    model = _load_model(model_name)
    prefixed = [f"passage: {t}" for t in texts]
    embeddings = model.encode(
        prefixed,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,  # We normalize manually for control
    )
    embeddings = embeddings.astype(np.float32)
    return _normalize(embeddings)


def embed_query(text: str, model_name: str = "BAAI/bge-small-en") -> np.ndarray:
    """
    Embed a single user query.
    Prefix: "query: <text>"
    Returns: float32 array of shape (1, dim), L2-normalized.
    """
    model = _load_model(model_name)
    prefixed = f"query: {text}"
    embedding = model.encode(
        [prefixed],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embedding = embedding.astype(np.float32)
    return _normalize(embedding)


def get_embedding_dim(model_name: str = "BAAI/bge-small-en") -> int:
    """Return the embedding dimensionality."""
    model = _load_model(model_name)
    return model.get_sentence_embedding_dimension()
