"""
main.py — FastAPI application for Ultra Doc-Intelligence.

Endpoints:
  POST /upload    → Ingest document, embed, store in FAISS
  POST /ask       → RAG question answering with guardrails + confidence
  POST /extract   → Structured shipment data extraction
  GET  /health    → Health check
  GET  /docs      → Available doc_ids
"""
from __future__ import annotations

import logging
import tempfile
import traceback
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from models import (
    UploadResponse,
    AskRequest,
    AskResponse,
    ExtractRequest,
    ExtractResponse,
    SourceChunk,
)
from pipeline.ingestor import ingest_file
from pipeline.embedder import embed_passages
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

# ── Debug: Print active Groq model and API key (first 8 chars) ──
logger.info(f"[DEBUG] GROQ_MODEL: {settings.groq_model}")
logger.info(f"[DEBUG] GROQ_API_KEY: {settings.groq_api_key[:8]}... (hidden)")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ultra Doc-Intelligence",
    description=(
        "AI-powered document intelligence for Transportation Management Systems. "
        "Upload logistics documents and query them with natural language."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_PATH = settings.storage_path
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Returns system status and list of indexed documents."""
    docs = list_documents(STORAGE_PATH)
    return {
        "status": "ok",
        "embedding_model": settings.embedding_model,
        "documents_indexed": len(docs),
        "storage_path": str(STORAGE_PATH),
    }


@app.get("/documents", tags=["System"])
async def list_docs():
    """Returns all available doc_ids."""
    docs = list_documents(STORAGE_PATH)
    return {"doc_ids": docs, "count": len(docs)}


# ── POST /upload ──────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse, tags=["Pipeline"])
async def upload_document(file: UploadFile = File(...)):
    """
    Ingest a logistics document:
      1. Parse (PDF/DOCX/TXT)
      2. Clean & intelligently chunk
      3. Embed with BAAI/bge-small-en
      4. Store in per-document FAISS index
    """
    # Validate file type
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        logger.info("Processing upload: %s (%d bytes)", file.filename, len(content))

        # Step 1: Parse + chunk
        ingested = ingest_file(
            tmp_path,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        ingested["filename"] = file.filename  # Preserve original name

        chunks = ingested["chunks"]
        if not chunks:
            raise HTTPException(status_code=422, detail="Document appears to be empty or unreadable.")

        # Step 2: Embed passages
        texts = [c["text"] for c in chunks]
        embeddings = embed_passages(texts, model_name=settings.embedding_model)

        # Step 3: Store in FAISS
        save_index(STORAGE_PATH, ingested["doc_id"], embeddings, chunks)

        logger.info(
            "Upload complete: doc_id=%s, chunks=%d",
            ingested["doc_id"],
            len(chunks),
        )

        return UploadResponse(
            doc_id=ingested["doc_id"],
            filename=file.filename,
            chunks_count=len(chunks),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
    finally:
        tmp_path.unlink(missing_ok=True)


# ── POST /ask ─────────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse, tags=["Pipeline"])
async def ask_question(request: AskRequest):
    """
    RAG question answering with full guardrails:
      1. Validate doc_id exists
      2. Retrieve top_k chunks via FAISS
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
        # Step 1: Retrieve
        chunks = retrieve(
            STORAGE_PATH,
            request.doc_id,
            request.question,
            top_k=settings.top_k,
            model_name=settings.embedding_model,
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
        # All LLM providers failed
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
