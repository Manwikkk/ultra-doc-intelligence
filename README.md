<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=36&duration=3000&pause=1500&color=6366F1&center=true&vCenter=true&width=700&height=80&lines=Ultra+Doc-Intelligence;AI+Document+Analysis" alt="Ultra Doc-Intelligence" />

<br/>

[![Live App](https://img.shields.io/badge/Live%20App-ultraship--doc--intelligence.streamlit.app-6366f1?style=for-the-badge&logo=streamlit&logoColor=white)](https://ultraship-doc-intelligence.streamlit.app/)
[![Backend](https://img.shields.io/badge/Backend-Render%20Free%20Tier-00d4aa?style=for-the-badge&logo=render&logoColor=white)](https://render.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

<br/>

> **AI-powered document intelligence for Transportation Management Systems.**  
> Upload logistics documents — query with natural language — extract structured shipment data.

<br/>

---

</div>

## ── What It Does

Ultra Doc-Intelligence is a production-grade RAG (Retrieval-Augmented Generation) system specifically tuned for logistics documents — **Bills of Lading**, **Rate Confirmations**, and **Carrier RCs**.

| Capability | Description |
|---|---|
| **Natural Language Q&A** | Ask questions like *"Who is the consignee?"* against any uploaded document |
| **Structured Extraction** | Automatically extracts 11 shipment fields (shipper, rate, equipment, dates…) |
| **Confidence Scoring** | Every answer is scored with a weighted similarity model |
| **Guardrails** | 4-stage hallucination prevention pipeline blocks low-quality answers |
| **Multi-LLM Fallback** | Routes through Groq → OpenAI → Gemini → Ollama automatically |
| **Persistent Chat** | Full conversation history preserved across all questions |

---

## ── Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│             STREAMLIT CLOUD  (ultraship-doc-intelligence.streamlit.app)  │
│                           frontend/app.py                           │
│                                                                     │
│  upload()                                                           │
│  ├── Parse PDF / DOCX / TXT  (PyMuPDF + pdfplumber + python-docx)  │
│  ├── Section-aware chunking  (regex boundary detection)             │
│  ├── BAAI/bge-small-en.encode(passages, normalize=True)  ◄─PyTorch │
│  └── POST /upload_embeddings  { chunks[], embeddings[] }   ──────┐  │
│                                                                   │  │
│  ask()                                                            │  │
│  ├── BAAI/bge-small-en.encode("query: …", normalize=True)        │  │
│  └── POST /ask  { doc_id, question, query_embedding }  ───────┐  │  │
└───────────────────────────────────────────────────────────────┼──┼──┘
                              HTTPS JSON  ◄─────────────────────┘  │
┌─────────────────────────────────────────────────────────────────┐ │
│             RENDER FREE TIER  (~50 MB RAM — no PyTorch)         │ │
│                           backend/main.py                       │ │
│                                                                 │ │
│  POST /upload_embeddings  ◄─────────────────────────────────────┘ │
│  ├── np.array(embeddings, dtype=float32)                          │
│  └── FAISS IndexFlatIP  →  storage/{doc_id}/                     │
│                                                                   │
│  POST /ask  ◄─────────────────────────────────────────────────────┘
│  ├── FAISS search  (pre-computed query vector from client)
│  ├── Guardrails  (similarity + grounding + confidence)
│  └── LLM Router: Groq → OpenAI → Gemini → Ollama
│
│  POST /extract  (regex + LLM hybrid, no embeddings needed)
│  GET  /health   GET  /documents
└────────────────────────────────────────────────────────────────────
```

> **Why this split?** Loading `sentence-transformers` (PyTorch) requires ~400–600 MB RAM.  
> Render's free tier gives 512 MB total. By moving all embedding to Streamlit Cloud,  
> the backend stays under **~80 MB RAM** and runs reliably for free.

---

## ── Live App

| Service | Platform | URL |
|---|---|---|
| **Frontend** | Streamlit Cloud | [ultraship-doc-intelligence.streamlit.app](https://ultraship-doc-intelligence.streamlit.app/) |
| **Backend API** | Render Free Tier | Configured via `BACKEND_URL` secret in Streamlit Cloud |
| **API Docs** | FastAPI Swagger | `<your-render-url>/docs` |

---

## ── Running Locally

### Prerequisites

- Python 3.11+
- At least one LLM API key (Groq is recommended — it's free)
- Git

### 1 — Clone the repo

```bash
git clone https://github.com/Manwikkk/ultra-doc-intelligence.git
cd ultra-doc-intelligence
```

### 2 — Start the Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

cp .env.example .env
# Open .env and add your API key(s) — at least GROQ_API_KEY is required
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Start the server:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Verify it's running:

```bash
curl http://localhost:8000/health
# {"status":"ok","embedding_location":"client-side (Streamlit)","embedding_model":"BAAI/bge-small-en",...}
```

### 3 — Start the Frontend

Open a **new terminal** (keep the backend running):

```bash
cd frontend
pip install -r requirements.txt

# Point the frontend at your LOCAL backend:
set BACKEND_URL=http://localhost:8000        # Windows CMD
# $env:BACKEND_URL="http://localhost:8000"  # Windows PowerShell
# export BACKEND_URL=http://localhost:8000  # macOS / Linux

streamlit run app.py
```

### 4 — Open the App

- **UI**: [http://localhost:8501](http://localhost:8501)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

> On first run, Streamlit downloads `BAAI/bge-small-en` (~130 MB). This is cached for all future sessions.

---

## ── Deployment Guide

### Frontend — Streamlit Cloud

1. Push your code to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set **Main file path** to `frontend/app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   BACKEND_URL = "https://your-render-app.onrender.com"
   ```
5. Deploy. Streamlit handles the `frontend/requirements.txt` automatically.

### Backend — Render

1. Create a new **Web Service** on [render.com](https://render.com).
2. Connect your GitHub repo.
3. Set:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
4. Add **Environment Variables** in Render dashboard:

   | Key | Value |
   |---|---|
   | `GROQ_API_KEY` | `gsk_...` |
   | `GROQ_MODEL` | `llama-3.3-70b-versatile` |
   | `OPENAI_API_KEY` | *(optional)* |
   | `GEMINI_API_KEY` | *(optional)* |
   | `SIMILARITY_THRESHOLD` | `0.35` |
   | `CONFIDENCE_THRESHOLD` | `0.30` |
   | `TOP_K` | `5` |

5. Deploy. The Render URL (e.g. `https://ultra-doc.onrender.com`) is what you add to Streamlit Cloud as `BACKEND_URL`.

> **Note:** Render free tier spins down after inactivity. The first request after sleep takes ~30–60 seconds.

---

## ── Project Structure

```
ultra-doc-intelligence/
├── backend/
│   ├── main.py                  # FastAPI — /upload_embeddings, /ask, /extract, /health
│   ├── config.py                # Pydantic-settings — reads from .env
│   ├── models.py                # Pydantic v2 request/response schemas
│   ├── .env.example             # Template — copy to .env and fill keys
│   ├── requirements.txt         # No sentence-transformers / PyTorch
│   └── pipeline/
│       ├── embedder.py          # Normalization helper only (no model loading)
│       ├── vector_store.py      # FAISS IndexFlatIP read/write per document
│       ├── retriever.py         # Top-K similarity retrieval from stored index
│       ├── guardrails.py        # 4-stage hallucination prevention
│       ├── confidence.py        # Weighted confidence scorer
│       ├── llm_router.py        # Multi-provider LLM fallback chain
│       ├── ingestor.py          # Raw text reader (used by /extract)
│       └── extractor.py        # Hybrid regex + LLM structured extraction
│
├── frontend/
│   ├── app.py                   # Streamlit UI + full embedding pipeline (client-side)
│   ├── requirements.txt         # sentence-transformers + parsers live here
│   └── Document.png             # App favicon
│
├── .devcontainer/
│   └── devcontainer.json        # GitHub Codespaces config
├── .gitignore
├── .python-version              # 3.11
└── README.md
```

---

## ── Environment Variables

### Backend (`.env` / Render dashboard)

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq API key (primary LLM) |
| `OPENAI_API_KEY` | — | Optional fallback |
| `GEMINI_API_KEY` | — | Optional fallback |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | OpenAI model |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama server |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `STORAGE_DIR` | `./storage` | FAISS index storage path |
| `TOP_K` | `5` | Chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.35` | Min cosine similarity (tuned for TMS docs) |
| `CONFIDENCE_THRESHOLD` | `0.30` | Min confidence to return an answer |
| `LLM_TIMEOUT` | `60` | API call timeout in seconds |

### Frontend (Streamlit Cloud Secrets / local shell)

| Variable | Default | Description |
|---|---|---|
| `BACKEND_URL` | `http://localhost:8000` | Points to the FastAPI backend |

---

## ── Chunking Strategy

**Algorithm**: Section-aware sliding window — purpose-built for TMS document structure.

1. **Section detection** — Scans for logistics domain headers (`pickup`, `consignee`, `bill of lading`, `rate breakdown`, etc.) using compiled regex
2. **Section-first split** — Text is divided at boundary lines to preserve semantic coherence
3. **Sliding window** for oversized sections (> 2400 chars / ~600 tokens) with 100-token overlap
4. **Tiny chunk merging** — Chunks under 150 chars merged into the previous chunk
5. **Page attribution** — Every chunk retains its source page number for citations

---

## ── Retrieval

| Property | Value |
|---|---|
| Index type | `faiss.IndexFlatIP` (exact cosine via inner product) |
| Normalization | L2-normalized on both sides — IP = cosine similarity |
| Document isolation | Each doc gets its own `storage/{doc_id}/` FAISS index |
| Query prefix | `"query: <question>"` (BGE instruction format) |
| Passage prefix | `"passage: <chunk>"` |
| Top-K | 5 chunks per query (configurable) |

---

## ── Guardrails

Four sequential checks prevent hallucinations and low-quality answers:

| # | Stage | When | Trigger | Action |
|---|---|---|---|---|
| 1 | Empty context | Pre-LLM | 0 chunks retrieved | Return "not found" immediately |
| 2 | Similarity threshold | Pre-LLM | Max similarity < 0.35 | Return "not found" — saves LLM API call |
| 3 | Answer grounding | Post-LLM | < 25% answer keywords in context | Refuse answer |
| 4 | Confidence threshold | Post-scoring | Confidence < 0.30 | Refuse answer |

---

## ── Confidence Scoring

```
confidence = 0.50 × max_similarity
           + 0.30 × avg_similarity
           + 0.20 × answer_coverage
```

| Tier | Score | Badge |
|---|---|---|
| High | ≥ 70% | Green |
| Medium | 45 – 69% | Amber |
| Low / Blocked | < 30% | Red (guardrail blocks) |

Scores are clamped to `[0.0, 1.0]`. Guardrail-refused answers return `confidence = 0.0`.

> Thresholds are tuned lower than generic RAG defaults because logistics documents  
> use repetitive, form-like language that naturally produces moderate embedding similarity.

---

## ── LLM Routing

```
Groq  →  OpenAI  →  Gemini  →  Ollama
```

- Missing key → provider silently skipped, logged as `⚠ skipped`
- API error / timeout → logged as `✗ failed`, next provider tried
- Success → logged as `✓ responded in Xs`, result returned immediately
- All fail → `503 Service Unavailable` with full error log

No caching — providers are retried fresh on every request.

---

## ── Structured Extraction

Extracts 11 fields using a **hybrid regex + LLM** approach:

| Field | Method |
|---|---|
| `rate`, `currency`, `weight`, `shipment_id`, `mode` | Regex (deterministic, high precision) |
| `shipper`, `consignee`, `carrier_name`, `equipment_type`, dates | LLM (handles varied phrasing) |

Regex results take priority for numeric/structured fields where LLMs may paraphrase.

---

## ── API Reference

### `GET /health`

```json
{
  "status": "ok",
  "embedding_location": "client-side (Streamlit)",
  "embedding_model": "BAAI/bge-small-en",
  "documents_indexed": 3
}
```

### `POST /upload_embeddings`

```json
// Request
{
  "doc_id": "uuid-generated-by-client",
  "filename": "bol.pdf",
  "chunks": [{"text": "...", "page": 1, "chunk_index": 0}],
  "embeddings": [[0.12, -0.05, ...]]
}

// Response
{ "doc_id": "uuid", "filename": "bol.pdf", "chunks_count": 14 }
```

### `POST /ask`

```json
// Request
{
  "doc_id": "uuid",
  "question": "Who is the consignee?",
  "query_embedding": [[0.08, -0.12, ...]]
}

// Response
{
  "answer": "ABC Corp",
  "confidence": 0.87,
  "sources": [{"text": "...", "page": 1, "chunk_index": 3, "similarity": 0.91}],
  "provider": "Groq",
  "logs": ["✓ Groq responded in 1.2s"],
  "guardrail_triggered": false
}
```

### `POST /extract`

```json
// Request
{ "doc_id": "uuid" }

// Response
{
  "doc_id": "uuid",
  "data": {
    "shipment_id": "LD53657",
    "shipper": "XYZ Logistics",
    "consignee": "ABC Manufacturing",
    "pickup_datetime": "2024-01-15 08:00",
    "delivery_datetime": "2024-01-16 14:00",
    "equipment_type": "53' Dry Van",
    "mode": "FTL",
    "rate": 1850.00,
    "currency": "USD",
    "weight": "42000 lbs",
    "carrier_name": "Fast Freight Inc"
  },
  "provider": "Groq",
  "confidence": 0.82
}
```

---

## ── Error Reference

| Scenario | Response |
|---|---|
| Off-topic question (low similarity) | Guardrail — "Not found in document" |
| Hallucinated answer (low grounding) | Post-LLM guardrail — refused |
| Confidence too low | Confidence guardrail — refused |
| All LLMs unavailable | `503` with full provider error log |
| Missing API key | Provider silently skipped, next tried |
| Unknown file type | `400` — supported: PDF, DOCX, TXT |
| Empty / unreadable document | `422` with clear message |
| Document not found | `404` — upload first |

---

## ── Supported Groq Models (April 2026)

| Model | Notes |
|---|---|
| `llama-3.3-70b-versatile` | **Recommended** — best results |
| `llama-3.1-8b-instant` | Faster, lower quality |
| `mixtral-8x7b-32768` | Large context window |
| `gemma2-9b-it` | Google-based alternative |

> After changing `GROQ_MODEL` in `.env`, always restart uvicorn — settings are read at startup only.

---

## ── Roadmap

- Multi-document queries across an entire shipment library
- Cross-encoder re-ranking after FAISS retrieval
- Streaming LLM token output to UI
- Persistent document metadata (SQLite)
- Scanned PDF support (Tesseract / PaddleOCR)
- API key authentication for production
- Swap FAISS for Pinecone / Weaviate for multi-user scale

---

<div align="center">

Built by **Manvik Siddhpura** &nbsp;·&nbsp; 2025  
Ultra Doc-Intelligence &nbsp;·&nbsp; AI-Powered Document Analysis for Transportation Management

</div>
