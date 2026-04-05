"""
embedder.py — STUB (v2: client-side embeddings).

Embedding computation has been moved to the Streamlit frontend to avoid loading
sentence-transformers / PyTorch on the backend, which would exceed the 512 MB
memory limit on Render's free tier.

The backend now receives pre-computed float32 embeddings via JSON and passes
them directly to FAISS — no model is loaded here.
"""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row.  Applied defensively if client sends un-normalized vecs."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    return vectors / norms


def ensure_normalized(embeddings: np.ndarray) -> np.ndarray:
    """
    Validate shape and dtype; return a float32, L2-normalized array.
    Called by the backend when it receives embeddings from the client.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
    embeddings = embeddings.astype(np.float32)
    return _normalize(embeddings)
