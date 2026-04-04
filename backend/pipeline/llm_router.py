"""
llm_router.py — Multi-provider LLM fallback system.

Priority order: Groq → OpenAI → Gemini → Ollama

Each provider is tried in order. Providers with missing API keys are
skipped immediately (no network call made). On failure, logs the error
and falls through to the next provider.

Returns:
    LLMResult(provider, response, logs)

IMPORTANT — Groq model names:
  Valid:   llama-3.3-70b-versatile, llama-3.1-8b-instant,
           mixtral-8x7b-32768, gemma2-9b-it
  Invalid: llama3-8b-8192 (decommissioned)
  Always check https://console.groq.com/docs/models for current list.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from config import settings
from models import LLMResult

logger = logging.getLogger(__name__)

# ── Startup log — always print the active model so problems are obvious ───────
logger.info("=" * 60)
logger.info("LLM Router initialised")
logger.info("  GROQ_MODEL  : %s", settings.groq_model)
logger.info("  GROQ_KEY    : %s...(hidden)", settings.groq_api_key[:12] if settings.groq_api_key else "NOT SET")
logger.info("  OPENAI_KEY  : %s", "configured" if settings.openai_api_key else "NOT SET")
logger.info("  GEMINI_KEY  : %s", "configured" if settings.gemini_api_key else "NOT SET")
logger.info("  LLM_TIMEOUT : %ds", settings.llm_timeout)
logger.info("=" * 60)


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_rag_prompt(context: str, question: str) -> str:
    return f"""You are an AI assistant specialised in Transportation Management System (TMS) documents.
Answer the question using ONLY the document context provided below.

RULES:
1. Answer only from the context. Do NOT make up information.
2. If the answer is not in the context, say: "Not found in document."
3. Be concise and precise. Quote specific values when they appear.
4. For dates/times, include the full value as shown.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


# ── Provider implementations ──────────────────────────────────────────────────

def _call_groq(prompt: str) -> str:
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY not configured")

    from groq import Groq
    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.groq_model,          # read from .env — NEVER hardcoded
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.1,
        timeout=settings.llm_timeout,
    )
    return response.choices[0].message.content.strip()


def _call_openai(prompt: str) -> str:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY not configured")

    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.1,
        timeout=settings.llm_timeout,
    )
    return response.choices[0].message.content.strip()


def _call_gemini(prompt: str) -> str:
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY not configured")

    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.gemini_model)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=1024,
            temperature=0.1,
        ),
    )
    return response.text.strip()


def _call_ollama(prompt: str) -> str:
    import requests as req
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1024},
    }
    resp = req.post(
        f"{settings.ollama_base_url}/api/generate",
        json=payload,
        timeout=settings.llm_timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# ── Provider registry ─────────────────────────────────────────────────────────

_PROVIDERS = [
    ("Groq",   _call_groq),
    ("OpenAI", _call_openai),
    ("Gemini", _call_gemini),
    ("Ollama", _call_ollama),
]


# ── Router ────────────────────────────────────────────────────────────────────

def _run_providers(prompt: str) -> LLMResult:
    """
    Try each LLM provider in priority order.
    - ValueError (missing key) → skip silently.
    - Any other exception      → log warning, try next.
    Raises RuntimeError if all providers fail.
    """
    logs: list[str] = []

    for provider_name, call_fn in _PROVIDERS:
        try:
            logger.info("Trying provider: %s", provider_name)
            start    = time.time()
            response = call_fn(prompt)
            elapsed  = time.time() - start
            logger.info("%s responded in %.2fs", provider_name, elapsed)
            logs.append(f"✓ {provider_name} responded in {elapsed:.1f}s")
            return LLMResult(provider=provider_name, response=response, logs=logs)

        except ValueError as e:
            # Missing API key — skip without logging as a failure
            msg = f"⚠ {provider_name} skipped: {e}"
            logs.append(msg)
            logger.info(msg)

        except Exception as e:
            msg = f"✗ {provider_name} failed: {type(e).__name__}: {str(e)[:200]}"
            logs.append(msg)
            logger.warning(msg)

    raise RuntimeError(
        "All LLM providers failed. Check your API keys or network connection.\n"
        "Logs: " + " | ".join(logs)
    )


def route_llm(context: str, question: str) -> LLMResult:
    """RAG-formatted prompt through the fallback chain."""
    prompt = build_rag_prompt(context, question)
    return _run_providers(prompt)


def route_llm_raw(prompt: str) -> LLMResult:
    """Raw prompt (e.g. structured extraction) through the fallback chain."""
    return _run_providers(prompt)
