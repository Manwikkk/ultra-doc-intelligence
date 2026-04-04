"""
models.py — Pydantic request/response schemas for all API endpoints.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── /upload ───────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_count: int
    message: str = "Document processed successfully"


# ── /ask ─────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    doc_id: str = Field(..., description="Document ID returned by /upload")
    question: str = Field(..., min_length=3, description="Natural language question")


class SourceChunk(BaseModel):
    text: str
    page: Optional[int] = None
    chunk_index: int
    similarity: float


class AskResponse(BaseModel):
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: list[SourceChunk]
    provider: str
    logs: list[str]
    guardrail_triggered: bool = False


# ── /extract ─────────────────────────────────────────────────────────────────

class ExtractRequest(BaseModel):
    doc_id: str = Field(..., description="Document ID returned by /upload")


class ShipmentData(BaseModel):
    shipment_id: Optional[str] = None
    shipper: Optional[str] = None
    consignee: Optional[str] = None
    pickup_datetime: Optional[str] = None
    delivery_datetime: Optional[str] = None
    equipment_type: Optional[str] = None
    mode: Optional[str] = None
    rate: Optional[float] = None
    currency: Optional[str] = None
    weight: Optional[str] = None
    carrier_name: Optional[str] = None


class ExtractResponse(BaseModel):
    doc_id: str
    data: ShipmentData
    provider: str
    logs: list[str]
    confidence: float


# ── Internal Pipeline Types ───────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    text: str
    page: Optional[int] = None
    chunk_index: int
    similarity: float


class LLMResult(BaseModel):
    provider: str
    response: str
    logs: list[str]
