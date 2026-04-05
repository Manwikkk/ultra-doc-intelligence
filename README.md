<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=34&duration=2800&pause=1400&color=6366F1&center=true&vCenter=true&width=720&height=80&lines=Ultra+Doc-Intelligence;RAG+for+Transportation+Docs;Guardrails+%C2%B7+Confidence+%C2%B7+Multi-LLM" alt="Ultra Doc-Intelligence" />

<br/>

[![Live App](https://img.shields.io/badge/Live%20App-ultraship--doc--intelligence.streamlit.app-6366f1?style=for-the-badge&logo=streamlit&logoColor=white)](https://ultraship-doc-intelligence.streamlit.app/)
<br/>
[![Backend](https://img.shields.io/badge/Backend-Render%20Free%20Tier-00d4aa?style=for-the-badge&logo=render&logoColor=white)](https://render.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

<br/>

> **Production-grade RAG system for Transportation Management System documents.**  
> Upload a Bill of Lading, Rate Confirmation, or Carrier RC — then query it in natural language  
> or extract all structured shipment fields automatically.

</div>

---

## ── What It Does

| Capability | Description |
|---|---|
| **Natural Language Q&A** | Ask *"Who is the consignee?"* — get a grounded, cited answer |
| **Structured Extraction** | Pulls 11 shipment fields via hybrid regex + LLM |
| **Confidence Scoring** | 3-component weighted score on every response |
| **4-Stage Guardrails** | Blocks hallucinations before and after LLM calls |
| **Multi-LLM Fallback** | Auto-routes Groq → OpenAI → Gemini → Ollama |
| **Persistent Chat** | Full conversation history, newest messages on top |
| **Memory-Optimised** | Embedding runs client-side — backend needs only ~80 MB RAM |

---

## ── Architecture

The system is split across two hosted services to stay within free-tier memory limits.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   STREAMLIT CLOUD · frontend/app.py                     │
│                ultraship-doc-intelligence.streamlit.app                 │
│                                                                         │
│  On upload ──────────────────────────────────────────────────────────   │
│  1. Parse PDF / DOCX / TXT   (PyMuPDF → pdfplumber fallback)            │
│  2. Clean + section-aware chunk (regex boundary detection)              │
│  3. BAAI/bge-small-en.encode(passages, normalize=True)  ◄── PyTorch     │
│  4. POST /upload_embeddings  { doc_id, filename, chunks[], embs[] }     │
│                                                                         │
│  On question ────────────────────────────────────────────────────       │
│  5. BAAI/bge-small-en.encode("query: …", normalize=True)                │
│  6. POST /ask  { doc_id, question, query_embedding }                    │
└───────────────────────────────────┼─────────────────────────────────────┘
                         HTTPS JSON │  (float32 arrays)
┌───────────────────────────────────▼─────────────────────────────────────┐
│            RENDER FREE TIER · backend/main.py  (~80 MB RAM)             │
│                                                                         │
│  POST /upload_embeddings                                                │
│  ├── Defensive L2 re-normalization (ensure_normalized)                  │
│  └── faiss.IndexFlatIP.add()  →  storage/{doc_id}/index + meta.json     │
│                                                                         │
│  POST /ask                                                              │
│  ├── faiss.IndexFlatIP.search(query_vector, top_k=5)                    │
│  ├── Pre-LLM guardrails  (empty context · similarity threshold)         │
│  ├── LLM Router: Groq → OpenAI → Gemini → Ollama                        │
│  ├── Post-LLM guardrails  (answer grounding · confidence threshold)     │
│  └── Return: answer · confidence · sources · provider · logs            │
│                                                                         │
│             POST /extract   GET /health   GET /documents                │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why split the embedding?**  
`sentence-transformers` with PyTorch needs ~400–600 MB at startup.  
Render's free tier provides only 512 MB total. Moving embedding to Streamlit Cloud  
keeps the backend under **~80 MB** and eliminates all OOM crashes.

---

## ── Live App

| Service | Platform | URL / Detail |
|---|---|---|
| Frontend | Streamlit Cloud | [ultraship-doc-intelligence.streamlit.app](https://ultraship-doc-intelligence.streamlit.app/) |
| Backend API | Render Free Tier | Set via `BACKEND_URL` secret in Streamlit Cloud |
| API Docs | FastAPI Swagger | `<render-url>/docs` |

> Render free tier sleeps after inactivity. The first request after sleep takes ~30–60 s to warm up.

---

## ── Running Locally

### Prerequisites

- Python 3.11+  
- At least one LLM API key (`GROQ_API_KEY` is free and recommended)  
- Git

### 1 — Clone

```bash
git clone https://github.com/Manwikkk/ultra-doc-intelligence.git
cd ultra-doc-intelligence
```

### 2 — Start the Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # macOS / Linux

pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` — at minimum add:

```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Start the server **from inside** the `backend/` directory:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Verify:

```bash
curl http://localhost:8000/health
# → {"status":"ok","embedding_location":"client-side (Streamlit)",...}
```

### 3 — Start the Frontend

Open a second terminal (keep backend running):

```bash
cd frontend
pip install -r requirements.txt

# Tell the frontend where the local backend is
set BACKEND_URL=http://localhost:8000         # Windows CMD
# $env:BACKEND_URL="http://localhost:8000"   # Windows PowerShell
# export BACKEND_URL=http://localhost:8000   # macOS / Linux

streamlit run app.py
```

### 4 — Open

| Service | URL |
|---|---|
| UI | http://localhost:8501 |
| API Docs | http://localhost:8000/docs |

> First run downloads `BAAI/bge-small-en` (~130 MB). Cached via `@st.cache_resource` for all future sessions.

---

## ── Deployment Guide

### Frontend — Streamlit Cloud

1. Push to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → connect repo.
3. **Main file path**: `frontend/app.py`
4. **Advanced settings → Secrets**:
   ```toml
   BACKEND_URL = "https://your-render-service.onrender.com"
   ```
5. Deploy — Streamlit auto-installs `frontend/requirements.txt`.

### Backend — Render

1. New **Web Service** on [render.com](https://render.com) → connect GitHub repo.
2. Configure:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
3. Add Environment Variables:

   | Key | Value |
   |---|---|
   | `GROQ_API_KEY` | `gsk_...` |
   | `GROQ_MODEL` | `llama-3.3-70b-versatile` |
   | `OPENAI_API_KEY` | *(optional)* |
   | `GEMINI_API_KEY` | *(optional)* |
   | `SIMILARITY_THRESHOLD` | `0.35` |
   | `CONFIDENCE_THRESHOLD` | `0.30` |
   | `TOP_K` | `5` |
   | `LLM_TIMEOUT` | `60` |

4. Take the Render URL → paste it as `BACKEND_URL` in Streamlit Secrets.

---

## ── Project Structure

```
ultra-doc-intelligence/
├── backend/
│   ├── main.py              # FastAPI: /upload_embeddings /ask /extract /health /documents
│   ├── config.py            # Pydantic-settings — all config from .env
│   ├── models.py            # Pydantic v2 request/response schemas
│   ├── .env.example         # Copy → .env, fill API keys
│   ├── requirements.txt     # No sentence-transformers / PyTorch
│   └── pipeline/
│       ├── embedder.py      # Stub: normalization helper only (no model)
│       ├── vector_store.py  # FAISS IndexFlatIP save/load per document
│       ├── retriever.py     # Top-K retrieval using pre-computed query vector
│       ├── guardrails.py    # 4-stage hallucination prevention
│       ├── confidence.py    # Weighted confidence scorer
│       ├── llm_router.py    # Multi-provider fallback chain + RAG prompt builder
│       ├── ingestor.py      # Raw text reader (for /extract endpoint)
│       └── extractor.py     # Hybrid regex + LLM structured extraction
│
├── frontend/
│   ├── app.py               # Streamlit UI + full client-side embedding pipeline
│   ├── requirements.txt     # sentence-transformers + PyMuPDF + pdfplumber + python-docx
│   └── Document.png         # Favicon
│
├── .devcontainer/
│   └── devcontainer.json    # GitHub Codespaces / VS Code devcontainer
├── .python-version          # 3.11
├── .gitignore
└── README.md
```

---

## ── Chunking Strategy

**Algorithm**: Section-aware sliding window, purpose-built for TMS document structure.

```
Raw text
   │
   ├── 1. Page tag injection  [Page N] markers preserved through parsing
   │
   ├── 2. Section boundary scan
   │      Regex detects headers matching TMS keywords:
   │      pickup · drop-off · delivery · consignee · shipper
   │      rate breakdown · freight charges · special instructions
   │      bill of lading · pro number · carrier · equipment
   │      weight · hazmat · commodity
   │
   ├── 3. Section-first split
   │      Text divided at every detected boundary line.
   │      Boundaries must be ≥200 chars apart to avoid over-splitting.
   │
   ├── 4. Sliding window (oversized sections only)
   │      Threshold: >2400 chars (~600 tokens × 4 chars/token)
   │      Window:    600 tokens · Overlap: 100 tokens
   │
   ├── 5. Tiny chunk merging
   │      Chunks under 150 chars are appended to the previous chunk.
   │
   └── 6. Page attribution
          Each chunk records its source page number for citations.
          Cap: 100 chunks maximum per document.
```

| Parameter | Default | Where set |
|---|---|---|
| `CHUNK_SIZE` | 600 tokens | `frontend/app.py` (hardcoded) |
| `CHUNK_OVERLAP` | 100 tokens | `frontend/app.py` (hardcoded) |
| Min chunk length | 150 chars | `frontend/app.py` (hardcoded) |
| Max chunks | 100 | `frontend/app.py` (hardcoded) |

> Chunking runs entirely in the Streamlit frontend — the backend never sees raw text for the Q&A flow.

---

## ── Retrieval Method

```
Query string
   │
   ├── Prefix: "query: <question>"   (BGE instruction format)
   ├── BAAI/bge-small-en.encode()
   ├── L2-normalize  →  vector shape (1, 384)
   └── POST to /ask as query_embedding (float32 list)

Backend /ask
   ├── np.array(query_embedding, dtype=float32)
   ├── Defensive re-normalization via ensure_normalized()
   ├── faiss.IndexFlatIP.search(query_vector, k=top_k)
   │      Inner product on L2-normalized vectors = cosine similarity
   ├── Clip scores to [0.0, 1.0]   (IP can return slightly >1.0 due to float32 drift)
   └── Sort descending · return top 5 RetrievedChunk objects
```

| Property | Value |
|---|---|
| Embedding model | `BAAI/bge-small-en` (384-dim, ~130 MB) |
| Index type | `faiss.IndexFlatIP` — exact search, no approximation |
| Similarity metric | Cosine (inner product of L2-normalized vectors) |
| Document isolation | Each doc has its own `storage/{doc_id}/` FAISS index + `meta.json` |
| Query prefix | `"query: <question>"` |
| Passage prefix | `"passage: <chunk text>"` |
| Top-K | 5 (configurable via `TOP_K` env var) |

---

## ── Guardrails

Four sequential safety checks prevent hallucinations and low-quality answers. Pre-LLM checks save API calls entirely.

```
          Retrieved chunks
                 │
                 ▼
  ┌─────────────────────────────────┐  FAIL → "No relevant content found."
  │ 1. Empty Context Check          │─────────────────────────────────────►  return
  │    Trigger: 0 chunks retrieved  │
  └──────────────┬──────────────────┘
                 │ PASS
                 ▼
  ┌─────────────────────────────────┐  FAIL → "Not found. (Best match: X%, needed Y%)"
  │ 2. Similarity Threshold         │─────────────────────────────────────►  return
  │    Trigger: max_sim < 0.35      │  (saves LLM API call)
  └──────────────┬──────────────────┘
                 │ PASS
                 ▼
          ┌────────────┐
          │  LLM Call  │  (Groq → OpenAI → Gemini → Ollama)
          └─────┬──────┘
                │ answer text
                ▼
  ┌─────────────────────────────────┐  FAIL → "Not found. (Answer not grounded.)"
  │ 3. Answer Grounding Check       │─────────────────────────────────────►  return
  │    Method: keyword overlap       │
  │    Trigger: coverage < 25%      │
  │    (stopwords removed)          │
  └──────────────┬──────────────────┘
                 │ PASS
                 ▼
        Confidence Scoring
                 │
                 ▼
  ┌─────────────────────────────────┐  FAIL → "Not found. (Confidence X% < Y%)"
  │ 4. Confidence Threshold         │─────────────────────────────────────►  return
  │    Trigger: score < 0.30        │
  └──────────────┬──────────────────┘
                 │ PASS
                 ▼
         ✓  Return answer
```

| # | Guardrail | Phase | Trigger | Effect |
|---|---|---|---|---|
| 1 | Empty context | Pre-LLM | 0 chunks retrieved | Immediate refusal |
| 2 | Similarity threshold | Pre-LLM | max cosine < 0.35 | Refusal + saved API call |
| 3 | Answer grounding | Post-LLM | keyword coverage < 25% | Answer refused |
| 4 | Confidence threshold | Post-scoring | score < 0.30 | Answer refused |

**Grounding check detail:** The answer and all retrieved context chunks are tokenized (words ≥3 chars, stopwords removed). Coverage = `len(answer_words ∩ context_words) / len(answer_words)`. Answers already containing "not found" or "I don't know" bypass this check.

---

## ── Confidence Scoring

Every answer that passes guardrails is assigned a confidence score using a 3-component weighted formula, computed entirely from FAISS similarity scores and lexical overlap — no additional model calls.

```
confidence = 0.50 × max_similarity
           + 0.30 × avg_similarity
           + 0.20 × answer_coverage
```

| Component | Weight | Source | Description |
|---|---|---|---|
| `max_similarity` | 50% | FAISS | Highest cosine score among top-5 chunks |
| `avg_similarity` | 30% | FAISS | Mean cosine across all retrieved chunks |
| `answer_coverage` | 20% | Lexical | Fraction of answer keywords found in context |

**Properties:**
- Score clamped to `[0.0, 1.0]`
- "Not found" answers are force-set to `0.0` without computing
- Trivial answers (e.g., a single number with no keywords) give `answer_coverage = 1.0` to avoid unfair penalisation
- Final score rounded to 4 decimal places

**Confidence tiers displayed in UI:**

| Tier | Score | Badge colour |
|---|---|---|
| High | ≥ 70% | Green |
| Medium | 45 – 69% | Amber |
| Low / Blocked | < 30% | Red (guardrail 4 blocks before reaching UI) |

> Thresholds are intentionally lower than generic RAG defaults. TMS documents are  
> structured, repetitive forms — embedding similarity naturally lands in the 0.35–0.65  
> range even for correct answers.

---

## ── LLM Routing

The router tries providers in fixed priority order on every request. No results are cached.

```
Groq  ──►  OpenAI  ──►  Gemini  ──►  Ollama
```

**Decision logic (per provider):**

```python
try:
    call_provider(prompt)          # timeout: LLM_TIMEOUT seconds (default 60)
    log("✓ Provider responded in Xs")
    return result                  # stop chain immediately on success

except ValueError:                 # missing API key
    log("⚠ Provider skipped: key not configured")
    continue                       # next provider

except Exception:                  # timeout, rate limit, API error, etc.
    log("✗ Provider failed: <reason>")
    continue                       # next provider

# All providers exhausted:
raise RuntimeError → 503 Service Unavailable
```

**RAG prompt template used by all providers:**

```
You are an AI assistant specialised in Transportation Management System (TMS) documents.
Answer the question using ONLY the document context provided below.

RULES:
1. Answer only from the context. Do NOT make up information.
2. If the answer is not in the context, say: "Not found in document."
3. Be concise and precise. Quote specific values when they appear.
4. For dates/times, include the full value as shown.

DOCUMENT CONTEXT:
{top-5 chunks joined by ---}

QUESTION: {user question}

ANSWER:
```

**Model settings for all providers:** `temperature=0.1`, `max_tokens=1024`

---

## ── Structured Extraction

Extracts 11 fields from a document using two parallel methods, merged with regex taking priority.

**Fields extracted:**

| Field | Method | Rationale |
|---|---|---|
| `rate` | Regex | Numeric — LLMs may round or reformat |
| `currency` | Regex | Short pattern, deterministic |
| `weight` | Regex | Numeric + unit — deterministic |
| `shipment_id` | Regex | Fixed format (e.g., `LD53657`) |
| `mode` | Regex | Limited vocabulary (FTL/LTL/etc.) |
| `shipper` | LLM | Free-form company names |
| `consignee` | LLM | Free-form company names |
| `carrier_name` | LLM | Free-form names |
| `equipment_type` | LLM | Varied phrasing (e.g. "53' Dry Van") |
| `pickup_datetime` | LLM | Multiple date formats |
| `delivery_datetime` | LLM | Multiple date formats |

Merge rule: `final[field] = regex_result if regex_result else llm_result`

Endpoint does **not** use embeddings — it reads the stored raw text chunks directly from the FAISS metadata JSON.

---

## ── API Reference

### `GET /health`
```json
{
  "status": "ok",
  "embedding_location": "client-side (Streamlit)",
  "embedding_model": "BAAI/bge-small-en",
  "documents_indexed": 3,
  "storage_path": "./storage"
}
```

### `POST /upload_embeddings`
```json
// Request
{
  "doc_id":     "uuid4-generated-by-client",
  "filename":   "bol.pdf",
  "chunks":     [{"text": "...", "page": 1, "chunk_index": 0}],
  "embeddings": [[0.12, -0.05, ...]]   // float32, L2-normalized, shape (N, 384)
}
// Response
{ "doc_id": "uuid", "filename": "bol.pdf", "chunks_count": 14, "message": "..." }
```

### `POST /ask`
```json
// Request
{
  "doc_id":          "uuid",
  "question":        "Who is the consignee?",
  "query_embedding": [[0.08, -0.12, ...]]   // shape [[384]] — note double-nested
}
// Response
{
  "answer":             "ABC Manufacturing",
  "confidence":         0.8741,
  "sources":            [{"text":"...","page":2,"chunk_index":3,"similarity":0.91}],
  "provider":           "Groq",
  "logs":               ["✓ Groq responded in 1.2s"],
  "guardrail_triggered": false
}
```

### `POST /extract`
```json
// Request  { "doc_id": "uuid" }
// Response
{
  "doc_id": "uuid",
  "data": {
    "shipment_id":       "LD53657",
    "shipper":           "XYZ Logistics",
    "consignee":         "ABC Manufacturing",
    "pickup_datetime":   "2024-01-15 08:00",
    "delivery_datetime": "2024-01-16 14:00",
    "equipment_type":    "53' Dry Van",
    "mode":              "FTL",
    "rate":              1850.00,
    "currency":          "USD",
    "weight":            "42000 lbs",
    "carrier_name":      "Fast Freight Inc"
  },
  "provider":    "Groq",
  "confidence":  0.8182,
  "logs":        ["✓ Groq responded in 1.8s"]
}
```

---

## ── Failure Cases

| Scenario | HTTP | System Response |
|---|---|---|
| Off-topic question (similarity < threshold) | 200 | Guardrail 2 → "Not found (Best match: X%)" |
| LLM answer not grounded in context | 200 | Guardrail 3 → "Not found (Answer not grounded.)" |
| Confidence score too low | 200 | Guardrail 4 → "Not found (Confidence X% < Y%)" |
| All LLM providers unavailable | 503 | Full provider error log returned |
| Missing API key | — | Provider silently skipped, next tried |
| Document not uploaded / expired | 404 | "Document not found. Please upload it first." |
| Empty document / no extractable text | 422 | Clear message |
| Unsupported file type | 400 | "Supported: PDF, DOCX, TXT" |
| Chunk/embedding count mismatch | 422 | "Mismatch: N chunks but M embeddings" |
| Backend cold start (Render free tier) | — | ~30–60 s delay on first request after sleep |

---

## ── Environment Variables

### Backend (`.env` / Render dashboard)

| Variable | Default | Required | Description |
|---|---|---|---|
| `GROQ_API_KEY` | — | Yes (primary) | Groq API key |
| `OPENAI_API_KEY` | — | No | OpenAI fallback |
| `GEMINI_API_KEY` | — | No | Gemini fallback |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | — | Groq model name |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | — | OpenAI model |
| `GEMINI_MODEL` | `gemini-1.5-flash` | — | Gemini model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | — | Local Ollama server |
| `OLLAMA_MODEL` | `llama3` | — | Ollama model |
| `STORAGE_DIR` | `./storage` | — | FAISS index root |
| `TOP_K` | `5` | — | Chunks per query |
| `SIMILARITY_THRESHOLD` | `0.35` | — | Pre-LLM cutoff |
| `CONFIDENCE_THRESHOLD` | `0.30` | — | Post-scoring cutoff |
| `LLM_TIMEOUT` | `60` | — | Per-provider timeout (seconds) |

### Frontend (Streamlit Secrets / local shell)

| Variable | Default | Description |
|---|---|---|
| `BACKEND_URL` | `http://localhost:8000` | FastAPI backend URL |

> **Important:** After any `.env` change, restart uvicorn. Settings are read at startup only.  
> Always run uvicorn **from inside `backend/`** — not from the project root.

---

## ── Supported Groq Models (April 2026)

| Model | Recommended | Notes |
|---|---|---|
| `llama-3.3-70b-versatile` | Yes | Best accuracy |
| `llama-3.1-8b-instant` | — | Faster, lower quality |
| `mixtral-8x7b-32768` | — | Large context window |
| `gemma2-9b-it` | — | Google-architecture alternative |
| `llama3-8b-8192` | Decommissioned | Will fail with API error |

---

## ── Improvement Ideas

The following improvements would meaningfully extend the system's capabilities:

| Priority | Improvement | Rationale |
|---|---|---|
| High | **Cross-encoder re-ranking** | After FAISS retrieval, a cross-encoder (e.g. `ms-marco-MiniLM`) re-scores chunks — improves answer quality significantly for ambiguous questions |
| High | **Streaming LLM output** | Token-by-token streaming to UI reduces perceived latency from ~3 s to near-instant first token |
| High | **OCR support** | Many logistics docs are scanned images. Tesseract or PaddleOCR integration would handle these |
| Medium | **Multi-document queries** | Answer questions across an entire shipment library, not just one doc at a time |
| Medium | **Persistent metadata (SQLite)** | Currently FAISS indexes are in-memory per session. Storing `doc_id → filename` mapping + metadata in SQLite would survive restarts |
| Medium | **Evaluation harness** | Automated RAG accuracy benchmarking against labeled TMS QA pairs to track improvements quantitatively |
| Medium | **API key authentication** | Rate-limited bearer token auth before exposing the backend publicly |
| Low | **Cloud vector DB** | Swap per-file FAISS for Pinecone / Weaviate / Qdrant for concurrent multi-user sessions |
| Low | **Semantic chunking** | Replace regex-boundary detection with an embedding-based segmentation model for higher coherence |
| Low | **Answer caching** | Cache recent `(doc_id, question)` pairs with Redis to avoid redundant LLM calls for repeat queries |

---

<div align="center">

Built with precision by **Manvik Siddhpura** &nbsp;·&nbsp; 2025  
Ultra Doc-Intelligence &nbsp;·&nbsp; AI-Powered Document Analysis for Transportation Management

</div>
