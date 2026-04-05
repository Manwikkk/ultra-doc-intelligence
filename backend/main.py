"""
main.py — FastAPI application for Ultra Doc-Intelligence (v2).

Architecture (v2):
  - Embeddings are computed in the Streamlit frontend (BAAI/bge-small-en).
  - Backend receives pre-computed float32 embeddings via JSON.
  - No sentence-transformers / PyTorch loaded on the server.
  - Enables deployment on Render free tier (512 MB RAM).

Endpoints:
  POST /upload_embeddings → Accept chunks + embeddings, store in FAISS
  POST /ask               → RAG Q&A (uses client-provided query embedding)
  POST /extract           → Structured shipment data extraction
  GET  /health            → Health check
  GET  /documents         → List indexed doc_ids
"""
from __future__ import annotations

import logging
import traceback
import os
from pathlib import Path

# ── Force CPU libraries into low-memory mode immediately ──
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from models import (
    UploadEmbeddingsRequest,
    UploadResponse,
    AskRequest,
    AskResponse,
    ExtractRequest,
    ExtractResponse,
    SourceChunk,
)
from pipeline.embedder import ensure_normalized
from pipeline.vector_store import save_index, doc_exists, list_documents
from pipeline.retriever import retrieve
from pipeline.guardrails import (
    run_pre_llm_guardrails,
    check_answer_grounding,
    check_confidence_threshold,
)
from pipeline.confidence import compute_confidence
from pipeline.llm_router import route_llm
from pipeline.extractor import extract_structured_data


# ── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

logger.info("[DEBUG] GROQ_MODEL: %s", settings.groq_model)
logger.info("[DEBUG] GROQ_API_KEY: %s... (hidden)", settings.groq_api_key[:8] if settings.groq_api_key else "MISSING")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ultra Doc-Intelligence",
    description=(
        "AI-powered document intelligence for Transportation Management Systems. "
        "Upload logistics documents and query them with natural language. "
        "v2: Embeddings computed client-side — backend is PyTorch-free."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_PATH = settings.storage_path


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Returns system status and list of indexed documents."""
    docs = list_documents(STORAGE_PATH)
    return {
        "status": "ok",
        "embedding_location": "client-side (Streamlit)",
        "embedding_model": "BAAI/bge-small-en",
        "documents_indexed": len(docs),
        "storage_path": str(STORAGE_PATH),
    }


@app.get("/documents", tags=["System"])
async def list_docs():
    """Returns all available doc_ids."""
    docs = list_documents(STORAGE_PATH)
    return {"doc_ids": docs, "count": len(docs)}


# ── POST /upload_embeddings ───────────────────────────────────────────────────

@app.post("/upload_embeddings", response_model=UploadResponse, tags=["Pipeline"])
async def upload_embeddings(request: UploadEmbeddingsRequest):
    """
    Accept pre-computed chunk embeddings from the client and store in FAISS.

    The client (Streamlit) is responsible for:
      1. Parsing the document
      2. Chunking the text
      3. Computing embeddings with BAAI/bge-small-en
      4. Sending chunks + embeddings here

    Backend:
      - Converts embeddings list → float32 numpy array
      - Normalizes (defensively)
      - Stores in per-document FAISS IndexFlatIP
    """
    try:
        chunks = request.chunks
        if not chunks:
            raise HTTPException(status_code=422, detail="No chunks provided.")

        if len(chunks) != len(request.embeddings):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Mismatch: {len(chunks)} chunks but "
                    f"{len(request.embeddings)} embeddings provided."
                ),
            )

        # Convert to float32 numpy array and normalize defensively
        embeddings_np = np.array(request.embeddings, dtype=np.float32)
        embeddings_np = ensure_normalized(embeddings_np)

        # Store in FAISS
        save_index(STORAGE_PATH, request.doc_id, embeddings_np, chunks)

        logger.info(
            "Stored index: doc_id=%s, filename=%s, chunks=%d, dim=%d",
            request.doc_id,
            request.filename,
            len(chunks),
            embeddings_np.shape[1],
        )

        return UploadResponse(
            doc_id=request.doc_id,
            filename=request.filename,
            chunks_count=len(chunks),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("upload_embeddings failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# ── POST /ask ─────────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse, tags=["Pipeline"])
async def ask_question(request: AskRequest):
    """
    RAG question answering with full guardrails.

    Client sends the query embedding (pre-computed with BAAI/bge-small-en).
    Backend performs FAISS retrieval, applies guardrails, calls LLM router,
    and returns answer + sources + confidence.

    Pipeline:
      1. Validate doc_id exists
      2. Retrieve top_k chunks via FAISS (using client-provided embedding)
      3. Pre-LLM guardrails (empty context, similarity threshold)
      4. Call LLM router (Groq → OpenAI → Gemini → Ollama)
      5. Post-LLM guardrails (answer grounding)
      6. Compute confidence score
      7. Confidence-based refusal if score < threshold
    """
    # Validate doc exists
    if not doc_exists(STORAGE_PATH, request.doc_id):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.doc_id}' not found. Please upload it first.",
        )

    logs: list[str] = []

    try:
        # Convert client query embedding to numpy
        query_vec = np.array(request.query_embedding, dtype=np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        logs.append(f"Query embedding received (dim={query_vec.shape[1]})")

        # Step 1: Retrieve
        chunks = retrieve(
            STORAGE_PATH,
            request.doc_id,
            query_vector=query_vec,
            top_k=settings.top_k,
        )
        logs.append(f"Retrieved {len(chunks)} chunks from index")

        # Step 2: Pre-LLM guardrails
        pre_check = run_pre_llm_guardrails(
            chunks,
            similarity_threshold=settings.similarity_threshold,
        )
        if pre_check is not None:
            logs.append(f"Guardrail triggered: {pre_check.triggered_by}")
            return AskResponse(
                answer=pre_check.reason,
                confidence=0.0,
                sources=[],
                provider="None",
                logs=logs,
                guardrail_triggered=True,
            )

        # Step 3: Build context
        context = "\n\n---\n\n".join(c.text for c in chunks)

        # Step 4: LLM
        llm_result = route_llm(context, request.question)
        logs.extend(llm_result.logs)
        answer = llm_result.response

        # Step 5: Post-LLM grounding check
        grounding = check_answer_grounding(answer, chunks, min_coverage=0.25)
        if not grounding.passed:
            logs.append(f"Guardrail triggered: {grounding.triggered_by}")
            return AskResponse(
                answer=grounding.reason,
                confidence=0.0,
                sources=[
                    SourceChunk(
                        text=c.text,
                        page=c.page,
                        chunk_index=c.chunk_index,
                        similarity=c.similarity,
                    )
                    for c in chunks
                ],
                provider=llm_result.provider,
                logs=logs,
                guardrail_triggered=True,
            )

        # Step 6: Confidence
        confidence = compute_confidence(answer, chunks)
        logs.append(f"Confidence score: {confidence:.2%}")

        # Step 7: Confidence-based refusal
        conf_check = check_confidence_threshold(confidence, settings.confidence_threshold)
        if not conf_check.passed:
            logs.append(f"Guardrail triggered: {conf_check.triggered_by}")
            return AskResponse(
                answer=conf_check.reason,
                confidence=confidence,
                sources=[
                    SourceChunk(
                        text=c.text,
                        page=c.page,
                        chunk_index=c.chunk_index,
                        similarity=c.similarity,
                    )
                    for c in chunks
                ],
                provider=llm_result.provider,
                logs=logs,
                guardrail_triggered=True,
            )

        return AskResponse(
            answer=answer,
            confidence=confidence,
            sources=[
                SourceChunk(
                    text=c.text,
                    page=c.page,
                    chunk_index=c.chunk_index,
                    similarity=c.similarity,
                )
                for c in chunks
            ],
            provider=llm_result.provider,
            logs=logs,
            guardrail_triggered=False,
        )

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("All LLM providers failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Ask failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")


# ── POST /extract ─────────────────────────────────────────────────────────────

@app.post("/extract", response_model=ExtractResponse, tags=["Pipeline"])
async def extract_data(request: ExtractRequest):
    """
    Extract structured shipment data using hybrid regex + LLM approach.
    Returns null for any field not found in the document.
    """
    if not doc_exists(STORAGE_PATH, request.doc_id):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.doc_id}' not found. Please upload it first.",
        )

    try:
        shipment_data, llm_result = extract_structured_data(STORAGE_PATH, request.doc_id)

        # Compute a simple completeness score as confidence
        fields = shipment_data.model_dump()
        non_null = sum(1 for v in fields.values() if v is not None)
        confidence = round(non_null / len(fields), 4)

        return ExtractResponse(
            doc_id=request.doc_id,
            data=shipment_data,
            provider=llm_result.provider,
            logs=llm_result.logs,
            confidence=confidence,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Extract failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
