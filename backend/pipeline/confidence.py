"""
confidence.py — Confidence score computation for RAG answers.

Formula:
    confidence = 0.5 * max_similarity
               + 0.3 * avg_similarity
               + 0.2 * answer_coverage

Where:
    max_similarity  = highest cosine similarity among retrieved chunks
    avg_similarity  = mean cosine similarity of top_k chunks
    answer_coverage = fraction of answer key-words present in retrieved context
                      (same tokenizer as guardrails.py)

Score is clamped to [0.0, 1.0].
"""
from __future__ import annotations

import re
import logging

from models import RetrievedChunk

logger = logging.getLogger(__name__)

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "and", "or", "but", "not", "it",
    "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "what", "which", "who",
}


def _key_words(text: str) -> set[str]:
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return {w for w in words if w not in _STOP_WORDS}


def compute_answer_coverage(answer: str, chunks: list[RetrievedChunk]) -> float:
    """
    Compute the fraction of answer key-words found in the retrieved context.
    Returns a value in [0.0, 1.0].
    """
    answer_words = _key_words(answer)
    if not answer_words:
        return 1.0  # trivial answer (e.g., a number) — don't penalize

    context_words: set[str] = set()
    for chunk in chunks:
        context_words |= _key_words(chunk.text)

    overlap = answer_words & context_words
    coverage = len(overlap) / len(answer_words)
    return min(coverage, 1.0)


def compute_confidence(
    answer: str,
    chunks: list[RetrievedChunk],
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> float:
    """
    Compute overall confidence score.

    Args:
        answer:  The LLM-generated answer string.
        chunks:  Retrieved chunks with similarity scores.
        weights: (w_max, w_avg, w_coverage) — must sum to 1.0.

    Returns:
        Confidence score in [0.0, 1.0].
    """
    if not chunks:
        return 0.0

    # "Not found" answers get forced low confidence
    if "not found" in answer.lower():
        return 0.0

    w_max, w_avg, w_cov = weights
    similarities = [c.similarity for c in chunks]

    max_sim = max(similarities)
    avg_sim = sum(similarities) / len(similarities)
    coverage = compute_answer_coverage(answer, chunks)

    score = w_max * max_sim + w_avg * avg_sim + w_cov * coverage

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))

    logger.debug(
        "Confidence: max_sim=%.4f avg_sim=%.4f coverage=%.4f → score=%.4f",
        max_sim,
        avg_sim,
        coverage,
        score,
    )
    return round(score, 4)
