# Ultra Doc-Intelligence — Implementation Plan

A production-quality POC for an AI-powered assistant inside a Transportation Management System (TMS), built with FastAPI + Streamlit + RAG.

---

## Proposed Project Structure

```
Ultra Doc-Intelligence/
├── backend/
│   ├── main.py                  # FastAPI app
│   ├── config.py                # API keys, settings
│   ├── models.py                # Pydantic schemas
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── ingestor.py          # Document parsing & chunking
│   │   ├── embedder.py          # BAAI/bge-small-en embeddings
│   │   ├── vector_store.py      # FAISS index management
│   │   ├── retriever.py         # Query retrieval
│   │   ├── guardrails.py        # Similarity/confidence guardrails
│   │   ├── confidence.py        # Confidence score computation
│   │   ├── llm_router.py        # Multi-provider LLM fallback
│   │   └── extractor.py         # Structured data extraction
│   ├── storage/                 # FAISS indexes + metadata (gitignored)
│   └── requirements.txt
├── frontend/
│   └── app.py                   # Streamlit UI
├── .env.example                 # API key template
├── .gitignore
└── README.md
```

---

## Proposed Changes

### Backend — Core Pipeline

#### [NEW] backend/config.py
Loads environment variables for Groq, OpenAI, Gemini, Ollama base URLs. Central settings object.

#### [NEW] backend/models.py
Pydantic models for API request/response:
- `UploadResponse` → `{ doc_id, filename, chunks_count }`
- `AskRequest` / `AskResponse` → question/answer with confidence, sources, provider, logs
- `ExtractResponse` → structured shipment fields

#### [NEW] backend/pipeline/ingestor.py
- PDF: PyMuPDF (`fitz`) with fallback to `pdfplumber`
- DOCX: `python-docx`
- TXT: plain read
- Intelligent chunking: 500–700 token windows with 100-token overlap, section-boundary-aware (Pickup/Drop/Rate/Instructions keywords)
- Text cleaning: whitespace normalization, header/footer strip

#### [NEW] backend/pipeline/embedder.py
- Loads `BAAI/bge-small-en` via `sentence-transformers`
- `embed_query("query: <text>")` for questions
- `embed_passage("passage: <text>")` for chunks
- Normalized L2 embeddings for cosine similarity via FAISS inner product

#### [NEW] backend/pipeline/vector_store.py
- `IndexFlatIP` per `doc_id`
- Saves/loads FAISS indexes to `backend/storage/`
- Stores chunk metadata (text, source, page) in parallel JSON

#### [NEW] backend/pipeline/retriever.py
- `retrieve(doc_id, query, top_k=4)` → returns chunks with similarity scores

#### [NEW] backend/pipeline/guardrails.py
1. Max similarity threshold (< 0.75 → reject)
2. Empty context check
3. Confidence-based refusal (< 0.5 → reject)
4. Answer grounding check (word overlap)

#### [NEW] backend/pipeline/confidence.py
```
confidence = 0.5 * max_sim + 0.3 * avg_sim + 0.2 * answer_coverage
```

#### [NEW] backend/pipeline/llm_router.py
Priority: Groq → OpenAI → Gemini → Ollama
- Each wrapped in try/except with timeout
- Returns `{ provider, response, logs }`
- Clean user-friendly fallback messages

#### [NEW] backend/pipeline/extractor.py
- Structured extraction of 11 shipment fields
- Hybrid: regex for numbers/dates/currency, LLM for semantic fields
- Returns null for missing fields

#### [NEW] backend/main.py
FastAPI app with 3 endpoints:
- `POST /upload`
- `POST /ask`
- `POST /extract`
- CORS enabled for Streamlit

### Frontend

#### [NEW] frontend/app.py
Streamlit UI with:
- File uploader (PDF, DOCX, TXT)
- Question input + submit
- Answer display with confidence progress bar
- Source snippet expander
- Provider badge + fallback logs
- "Extract Structured Data" button → JSON viewer

### Configuration & Docs

#### [NEW] .env.example
```
GROQ_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434
```

#### [NEW] README.md
Architecture diagram, all system explanations per spec.

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| FAISS index type | IndexFlatIP | Exact cosine sim via normalized embeddings |
| Chunking | Sliding window + section-aware | Preserves semantic sections |
| Guardrails | Pre-LLM + post-LLM | Prevents wasted API calls |
| Extraction | Hybrid regex + LLM | More reliable for structured numeric data |

---

## Verification Plan

### Automated
- Start `uvicorn backend.main:app --reload`
- Start `streamlit run frontend/app.py`
- Upload one of the sample PDFs in the workspace
- Run /ask with a logistics question
- Run /extract and verify JSON output

### Manual
- Confirm fallback logs appear when primary LLM is unavailable
- Confirm confidence < 0.5 triggers "Not found in document"
- Confirm structured extraction returns null for missing fields
