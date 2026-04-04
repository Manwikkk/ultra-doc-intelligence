"""
guardrails.py — Pre-LLM and post-LLM safety checks.

Guardrails applied in order:
  1. Empty context check        — no chunks retrieved
  2. Similarity threshold       — max similarity < 0.75
  3. Answer grounding check     — answer words not in context (post-LLM)
  4. Confidence-based refusal   — confidence < 0.5 (post-scoring)

Each check returns a GuardrailResult indicating whether to proceed or refuse.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Optional

from models import RetrievedChunk

logger = logging.getLogger(__name__)

NOT_FOUND_MSG = "Not found in document."


@dataclass
class GuardrailResult:
    passed: bool
    reason: str = ""
    triggered_by: str = ""


# ── 1. Empty context ─────────────────────────────────────────────────────────

def check_empty_context(chunks: list[RetrievedChunk]) -> GuardrailResult:
    """Fail if no chunks were retrieved."""
    if not chunks:
        msg = "No relevant content found in the document for this question."
        logger.warning("Guardrail [empty_context] triggered")
        return GuardrailResult(passed=False, reason=msg, triggered_by="empty_context")
    return GuardrailResult(passed=True)


# ── 2. Similarity threshold ───────────────────────────────────────────────────

def check_similarity_threshold(
    chunks: list[RetrievedChunk],
    threshold: float = 0.75,
) -> GuardrailResult:
    """
    Fail if the highest similarity score is below the threshold.
    This prevents hallucination on weakly-related content.
    """
    if not chunks:
        return GuardrailResult(passed=False, reason=NOT_FOUND_MSG, triggered_by="similarity_threshold")

    max_similarity = max(c.similarity for c in chunks)
    if max_similarity < threshold:
        msg = (
            f"Not found in document. "
            f"(Best match relevance: {max_similarity:.0%}, required: {threshold:.0%})"
        )
        logger.warning(
            "Guardrail [similarity_threshold] triggered: %.4f < %.4f",
            max_similarity,
            threshold,
        )
        return GuardrailResult(
            passed=False,
            reason=msg,
            triggered_by="similarity_threshold",
        )
    return GuardrailResult(passed=True)


# ── 3. Answer grounding (post-LLM) ───────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenizer, removes stopwords."""
    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "and", "or", "but", "not", "it",
        "its", "this", "that", "these", "those", "i", "you", "he",
        "she", "we", "they", "what", "which", "who", "found", "document",
    }
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return {w for w in words if w not in STOP_WORDS}


def check_answer_grounding(
    answer: str,
    chunks: list[RetrievedChunk],
    min_coverage: float = 0.25,
) -> GuardrailResult:
    """
    Post-LLM grounding check:
    Verify that a meaningful fraction of the answer's key words
    appear in the retrieved context.

    If coverage < min_coverage, the answer may be hallucinated.
    """
    # If LLM already said "not found", skip grounding check
    if "not found" in answer.lower() or "i don't know" in answer.lower():
        return GuardrailResult(passed=True)

    answer_words = _tokenize(answer)
    if not answer_words:
        return GuardrailResult(passed=True)

    context_words: set[str] = set()
    for chunk in chunks:
        context_words |= _tokenize(chunk.text)

    overlap = answer_words & context_words
    coverage = len(overlap) / len(answer_words)

    logger.debug(
        "Grounding check: coverage=%.4f (overlap=%d / answer=%d)",
        coverage,
        len(overlap),
        len(answer_words),
    )

    if coverage < min_coverage:
        logger.warning(
            "Guardrail [answer_grounding] triggered: coverage=%.4f < %.4f",
            coverage,
            min_coverage,
        )
        return GuardrailResult(
            passed=False,
            reason=NOT_FOUND_MSG + " (Answer could not be grounded in document content.)",
            triggered_by="answer_grounding",
        )
    return GuardrailResult(passed=True)


# ── 4. Confidence refusal (post-scoring) ─────────────────────────────────────

def check_confidence_threshold(
    confidence: float,
    threshold: float = 0.5,
) -> GuardrailResult:
    """
    Refuse to return an answer if the confidence score is too low.
    This is the final safety net before returning to the user.
    """
    if confidence < threshold:
        msg = (
            f"Not found in document. "
            f"(Confidence {confidence:.0%} is below the required {threshold:.0%}.)"
        )
        logger.warning(
            "Guardrail [confidence_threshold] triggered: %.4f < %.4f",
            confidence,
            threshold,
        )
        return GuardrailResult(
            passed=False,
            reason=msg,
            triggered_by="confidence_threshold",
        )
    return GuardrailResult(passed=True)


# ── Convenience: run all pre-LLM checks ──────────────────────────────────────

def run_pre_llm_guardrails(
    chunks: list[RetrievedChunk],
    similarity_threshold: float = 0.75,
) -> Optional[GuardrailResult]:
    """
    Run empty-context and similarity-threshold checks.
    Returns the first failed GuardrailResult, or None if all pass.
    """
    for check_fn, kwargs in [
        (check_empty_context, {"chunks": chunks}),
        (check_similarity_threshold, {"chunks": chunks, "threshold": similarity_threshold}),
    ]:
        result: GuardrailResult = check_fn(**kwargs)
        if not result.passed:
            return result
    return None
