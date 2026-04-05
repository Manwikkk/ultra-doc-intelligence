"""
config.py — Central configuration using pydantic-settings.
All settings are loaded from environment variables / .env file.
"""
from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM Provider Keys ────────────────────────────────────────────────────
    groq_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # ── Ollama ───────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # ── Model Names ──────────────────────────────────────────────────────────
    # IMPORTANT: These must match supported models.
    # For Groq, valid models include: llama-3.3-70b-versatile, llama-3.1-8b-instant,
    # mixtral-8x7b-32768, gemma2-9b-it
    groq_model: str = "llama-3.3-70b-versatile"
    openai_model: str = "gpt-3.5-turbo"
    gemini_model: str = "gemini-1.5-flash"

    # ── NOTE: Embedding model runs client-side (Streamlit) ────────────────────
    # embedding_model removed in v2 — backend is PyTorch-free.

    # ── Storage ───────────────────────────────────────────────────────────────
    storage_dir: str = "./storage"

    # ── RAG ──────────────────────────────────────────────────────────────────
    top_k: int = 5
    # Lowered from 0.75 — TMS documents often have moderate similarity scores
    similarity_threshold: float = 0.35
    # Lowered from 0.5 — allows more answers through
    confidence_threshold: float = 0.3
    chunk_size: int = 600
    chunk_overlap: int = 100

    # ── LLM Timeout (seconds) ─────────────────────────────────────────────────
    llm_timeout: int = 60

    @property
    def storage_path(self) -> Path:
        p = Path(self.storage_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
