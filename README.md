# 🚛 Ultra Doc-Intelligence

> **AI-powered document intelligence assistant for Transportation Management Systems (TMS)**  
> Upload logistics documents — query them with natural language — extract structured shipment data.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                  Streamlit Frontend  (heavy ML here)              │
│                                                                    │
│  upload()                                                          │
│  ├── Parse PDF/DOCX/TXT                                           │
│  ├── Clean + chunk text (section-aware sliding window)            │
│  ├── BGE-small-en.encode(passages, normalize=True)  ◄── PyTorch  │
│  └── POST /upload_embeddings  { chunks, embeddings }              │
│                                                                    │
│  ask()                                                             │
│  ├── BGE-small-en.encode("query: …", normalize=True)             │
│  └── POST /ask  { doc_id, question, query_embedding }             │
└──────────────────────────┬───────────────────────────────────────┘
                           │ JSON (float32 embeddings)
┌──────────────────────────▼───────────────────────────────────────┐
│              FastAPI Backend  (no PyTorch — ~50 MB RAM)           │
│                                                                    │
│  POST /upload_embeddings                                           │
│  ├── np.array(embeddings, dtype=float32)                          │
│  └── FAISS IndexFlatIP  →  storage/{doc_id}/                      │
│                                                                    │
│  POST /ask                                                         │
│  ├── FAISS search (client embedding)                               │
│  ├── Guardrails (similarity + grounding + confidence)             │
│  └── LLM Router: Groq → OpenAI → Gemini → Ollama                 │
│                                                                    │
│  POST /extract  (regex + LLM hybrid, no embeddings)               │
└──────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Clone & set up environment

```bash
cd "Ultra Doc-Intelligence/backend"
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Start FastAPI backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 4. Start Streamlit frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

### 5. Open the app

- **UI**: http://localhost:8501  
- **API Docs**: http://localhost:8000/docs

---

## 📂 Project Structure

```
Ultra Doc-Intelligence/
├── backend/
│   ├── main.py                  # FastAPI app — /upload_embeddings, /ask, /extract
│   ├── config.py                # Settings (pydantic-settings)
│   ├── models.py                # Pydantic schemas (v2: UploadEmbeddingsRequest + query_embedding)
│   ├── pipeline/
│   │   ├── embedder.py          # Stub only — normalization helper, no model loading
│   │   ├── vector_store.py      # FAISS index management
│   │   ├── retriever.py         # Similarity retrieval (accepts pre-computed vector)
│   │   ├── guardrails.py        # Hallucination prevention
│   │   ├── confidence.py        # Confidence scoring
│   │   ├── llm_router.py        # Multi-provider LLM fallback
│   │   ├── ingestor.py          # (kept for /extract raw text reading)
│   │   └── extractor.py         # Structured data extraction
│   ├── storage/                 # FAISS indexes (auto-created)
│   └── requirements.txt         # No sentence-transformers / PyTorch
├── frontend/
│   ├── app.py                   # Streamlit UI + full embedding pipeline
│   └── requirements.txt         # sentence-transformers + parsers live here
├── .env.example
├── .gitignore
└── README.md
```

---

## 🔍 Chunking Strategy

**Algorithm**: Semantic-boundary-aware sliding window

1. **Section detection**: Scans for TMS-domain section headers (Pickup, Drop, Consignee, Rate Breakdown, Special Instructions, Bill of Lading, etc.) using regex
2. **Section split**: Text is split at detected section boundaries first — preserving semantic coherence
3. **Sliding window within oversized sections**: For sections exceeding ~2400 chars (600 tokens × 4 chars/token), a sliding window with 100-token overlap is applied
4. **Tiny chunk merging**: Chunks under 150 chars are merged into the previous chunk
5. **Page tracking**: Each chunk retains its page number for source citations

**Parameters** (configurable via `.env`):
| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 600 tokens | Target chunk size |
| `CHUNK_OVERLAP` | 100 tokens | Sliding overlap |

---

## 🔎 Retrieval Method

- **Index type**: `faiss.IndexFlatIP` (inner product = exact cosine similarity on L2-normalized vectors)
- **Per-document isolation**: Each document gets its own FAISS index stored at `storage/{doc_id}/`
- **Query embedding**: `"query: <user question>"` prefix (bge instruction format)
- **Passage embedding**: `"passage: <chunk text>"` prefix
- **Top-K**: 4 chunks retrieved per query (configurable)
- **Similarity metric**: Cosine similarity via inner product of L2-normalized vectors

---

## 🔒 Guardrails

Four guardrails applied in sequence to prevent hallucinations:

| # | Guardrail | When | Action |
|---|-----------|------|--------|
| 1 | **Empty context** | Pre-LLM | Return "Not found" if 0 chunks retrieved |
| 2 | **Similarity threshold** | Pre-LLM | Return "Not found" if max similarity < 0.35 |
| 3 | **Answer grounding** | Post-LLM | Check if ≥25% of answer keywords exist in context |
| 4 | **Confidence refusal** | Post-scoring | Return "Not found" if confidence score < 0.30 |

**Pre-LLM checks** save API calls when there's clearly no relevant content.  
**Post-LLM checks** catch cases where the LLM may have generated information beyond the context.

---

## 📊 Confidence Scoring

```
confidence = 0.5 × max_similarity
           + 0.3 × avg_similarity
           + 0.2 × answer_coverage
```

| Component | Weight | Description |
|-----------|--------|-------------|
| `max_similarity` | 50% | Highest cosine similarity among retrieved chunks |
| `avg_similarity` | 30% | Mean cosine similarity across top-4 chunks |
| `answer_coverage` | 20% | Fraction of answer keywords found in retrieved context |

Score is clamped to `[0.0, 1.0]`. "Not found" answers get `confidence = 0.0`.

**Interpretation**:
- 🟢 `≥ 70%` — High confidence
- 🟡 `45–69%` — Medium confidence
- 🔴 `< 30%` — Low (guardrail blocks this answer)

> **Note:** Thresholds are tuned lower than naive defaults because logistics documents often have moderate embedding similarity scores due to their structured, repetitive, form-like text.

---

## 🔁 LLM Routing Strategy

```
Groq  →  OpenAI  →  Gemini  →  Ollama
```

**Behavior**:
1. Each provider is attempted in priority order
2. If API key is missing → silently skip with log `⚠ Provider skipped`
3. If API call fails (timeout, error) → log `✗ Provider failed: <reason>` and try next
4. On success → log `✓ Provider responded in Xs` and return
5. If all fail → `503 Service Unavailable` with full log

Providers are tried fresh on every request. No caching of LLM responses.

---

## 🗂️ Structured Extraction

**Fields**: `shipment_id`, `shipper`, `consignee`, `pickup_datetime`, `delivery_datetime`, `equipment_type`, `mode`, `rate`, `currency`, `weight`, `carrier_name`

**Hybrid approach**:
- **Regex** (priority): `rate`, `currency`, `weight`, `shipment_id`, `mode` — deterministic patterns
- **LLM** (fills gaps): `shipper`, `consignee`, `carrier_name`, `equipment_type`, dates

The merge logic gives regex results priority for numeric/structured fields where LLMs may paraphrase incorrectly.

---

## ⚠️ Failure Cases & Handling

| Scenario | System Response |
|----------|----------------|
| Empty / unreadable document | 422 error with clear message |
| Low similarity (off-topic question) | Guardrail → "Not found in document" |
| All LLMs fail | 503 with provider error logs |
| Missing API key | Provider silently skipped; next provider tried |
| Hallucinated answer (low grounding) | Post-LLM guardrail → refused |
| Confidence too low | Confidence guardrail → refused |
| Unknown file type | 400 error with supported types listed |

---

## 🚀 API Reference

### `POST /upload_embeddings`
Receive pre-computed embeddings from the client and store in FAISS.

**Request** (`application/json`):
```json
{
  "doc_id": "uuid-generated-by-client",
  "filename": "bol.pdf",
  "chunks":     [{"text": "...", "page": 1, "chunk_index": 0}, ...],
  "embeddings": [[0.12, -0.05, ...], ...]   // float32, normalized
}
```
**Response**:
```json
{ "doc_id": "uuid", "filename": "bol.pdf", "chunks_count": 14, "message": "..." }
```

---

### `POST /ask`
Ask a natural language question (client sends pre-embedded query).

**Request**:
```json
{
  "doc_id":         "uuid",
  "question":       "Who is the consignee?",
  "query_embedding": [[0.08, -0.12, ...]]   // float32, shape [[dim]]
}
```
**Response**:
```json
{
  "answer": "ABC Corp",
  "confidence": 0.87,
  "sources": [{"text": "...", "page": 1, "chunk_index": 3, "similarity": 0.91}],
  "provider": "Groq",
  "logs": ["✓ Groq responded in 1.2s"],
  "guardrail_triggered": false
}
```

---

### `POST /extract`
Extract structured shipment data.

**Request**:
```json
{ "doc_id": "uuid" }
```
**Response**:
```json
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
  "logs": ["✓ Groq responded"],
  "confidence": 0.82
}
```

---

## 🧠 Memory Optimization Strategy

Ultra Doc-Intelligence v2 moves **all embedding computation to the Streamlit frontend**,
keeping the FastAPI backend lightweight and deployable on constrained environments like
Render's free tier (512 MB RAM).

### Problem

Loading `sentence-transformers` (with PyTorch) requires ~400–600 MB of RAM at startup,
causing OOM crashes on Render free tier or any sub-1 GB server.

### Solution

| Layer | Responsibility | RAM Impact |
|-------|---------------|------------|
| **Streamlit frontend** | Parse, chunk, embed (BAAI/bge-small-en) | ~300 MB (runs on user's machine or a separate process) |
| **FastAPI backend** | Receive float32 arrays, run FAISS, call LLM APIs | ~50–80 MB (no PyTorch) |

### How it works

1. **On document upload** — Streamlit parses the file locally, chunks the text, encodes
   every chunk with `BGE-small`, and POSTs the resulting `float32` embedding matrix to
   `/upload_embeddings`. The backend just calls `faiss.IndexFlatIP.add()`.

2. **On every question** — Streamlit encodes the query string into a single vector and
   sends it inside the `/ask` JSON body. The backend calls `index.search()` with the
   provided vector — no model needed.

3. **Normalization** — Both sides use L2-normalized vectors so inner-product = cosine
   similarity. The backend applies a defensive re-normalization to catch any drift.

### Benefits

- ✅ Backend stays under 100 MB RAM  
- ✅ No PyTorch / CUDA on the server  
- ✅ Same embedding quality (same model, same normalization)  
- ✅ Supports local execution and hosted deployment simultaneously  
- ✅ Frontend model load is cached across uploads (`@st.cache_resource`)  

---

## 🔮 Future Improvements

1. **Multi-document queries** — answer questions across an entire document library
2. **Re-ranking** — add cross-encoder re-ranking step after FAISS retrieval
3. **Streaming responses** — stream LLM tokens to UI for faster perceived response
4. **Persistent metadata** — store doc_id ↔ filename mapping in SQLite
5. **OCR support** — scanned PDF support via Tesseract/PaddleOCR
6. **Evaluation harness** — automated RAG accuracy benchmarking with labeled QA pairs
7. **Authentication** — API key auth for production deployment
8. **Cloud vector DB** — swap FAISS for Pinecone/Weaviate for multi-user scale

---

## 🛠️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key (primary LLM) |
| `OPENAI_API_KEY` | — | OpenAI API key (fallback) |
| `GEMINI_API_KEY` | — | Google Gemini API key (fallback) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `TOP_K` | `5` | Chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.35` | Min similarity to proceed (tuned for TMS docs) |
| `CONFIDENCE_THRESHOLD` | `0.30` | Min confidence to return answer |
| `CHUNK_SIZE` | `600` | Target chunk size (tokens) — frontend only |
| `CHUNK_OVERLAP` | `100` | Chunk overlap (tokens) — frontend only |

> **Note:** `EMBEDDING_MODEL` has been removed from backend config. The model
> `BAAI/bge-small-en` is hardcoded in the Streamlit frontend and is not
> configurable via the backend `.env`.
