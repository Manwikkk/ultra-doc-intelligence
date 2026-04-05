"""
app.py — Streamlit frontend for Ultra Doc-Intelligence (v2).

v2 Architecture:
  - Document parsing, chunking, and embedding (BAAI/bge-small-en) run here.
  - Backend (FastAPI) only does FAISS storage + LLM inference — no PyTorch.
  - Enables backend deployment on Render free tier (512 MB RAM).

UI v3:
  - Persistent chat history — Q&A accumulates across interactions.
  - Extraction results shown in a separate tab — never clears chat.
"""
from __future__ import annotations

import io
import os
import re
import uuid
import base64
import logging
import datetime
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)


# ── Favicon helper ────────────────────────────────────────────────────────────
def get_favicon():
    icon_path = Path(__file__).parent / "Document.png"
    if icon_path.exists():
        with open(icon_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{data}"
    return "📄"


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ultra Doc-Intelligence",
    page_icon=get_favicon(),
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .stApp { background: #0a0f1a; }
    .stApp > header { background: transparent !important; }
    #MainMenu, footer { visibility: hidden; }
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    section[data-testid="stSidebar"] { display: none !important; }

    /* ── Header ── */
    .site-header {
        background: rgba(10, 15, 26, 0.95);
        border-bottom: 1px solid rgba(255,255,255,0.06);
        padding: 0 0 1rem 0;
        margin-bottom: 2rem;
    }
    .header-inner {
        max-width: 1200px;
        margin: 0 auto;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 0;
    }
    .header-brand { display: flex; flex-direction: column; }
    .header-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.4px;
        line-height: 1.2;
    }
    .header-subtitle {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.3);
        font-weight: 400;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-top: 3px;
    }
    .header-status {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 0.75rem;
        color: rgba(255,255,255,0.45);
        font-weight: 500;
    }
    .dot-green {
        width: 7px; height: 7px; border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 8px rgba(34,197,94,0.6);
        display: inline-block; flex-shrink: 0;
    }
    .dot-red {
        width: 7px; height: 7px; border-radius: 50%;
        background: #ef4444;
        box-shadow: 0 0 8px rgba(239,68,68,0.5);
        display: inline-block; flex-shrink: 0;
    }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.67rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.25);
        margin-bottom: 0.6rem;
        margin-top: 0.1rem;
    }

    /* ── Cards ── */
    .card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.85rem;
    }
    .card-warning {
        background: rgba(245,158,11,0.06);
        border: 1px solid rgba(245,158,11,0.2);
        border-left: 3px solid #f59e0b;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }

    /* ── Chat message bubbles ── */
    .chat-question-row {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 0.5rem;
    }
    .chat-question-bubble {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: #ffffff;
        border-radius: 16px 16px 4px 16px;
        padding: 0.7rem 1rem;
        max-width: 85%;
        font-size: 0.88rem;
        line-height: 1.5;
        font-weight: 500;
    }
    .chat-answer-row {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 1.25rem;
    }
    .chat-answer-bubble {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 4px 16px 16px 16px;
        padding: 0.9rem 1.1rem;
        max-width: 92%;
        font-size: 0.875rem;
        color: rgba(255,255,255,0.85);
        line-height: 1.8;
    }
    .chat-answer-guardrail {
        background: rgba(239,68,68,0.07);
        border: 1px solid rgba(239,68,68,0.2);
        border-left: 3px solid #ef4444;
        border-radius: 4px 16px 16px 16px;
        padding: 0.9rem 1.1rem;
        max-width: 92%;
        font-size: 0.875rem;
        color: #fca5a5;
        line-height: 1.7;
    }
    .chat-meta {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 0.55rem;
        flex-wrap: wrap;
    }
    .chat-timestamp {
        font-size: 0.65rem;
        color: rgba(255,255,255,0.2);
        font-family: 'JetBrains Mono', monospace;
    }
    .chat-conf-badge {
        font-size: 0.67rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 999px;
        letter-spacing: 0.3px;
    }
    .conf-high   { background: rgba(34,197,94,0.12);  color: #22c55e;  border: 1px solid rgba(34,197,94,0.25); }
    .conf-medium { background: rgba(245,158,11,0.12); color: #f59e0b;  border: 1px solid rgba(245,158,11,0.25); }
    .conf-low    { background: rgba(239,68,68,0.12);  color: #ef4444;  border: 1px solid rgba(239,68,68,0.25); }

    /* ── Provider badge ── */
    .provider-pill {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 2px 10px; border-radius: 999px;
        font-size: 0.68rem; font-weight: 600; letter-spacing: 0.2px;
    }
    .pill-groq   { background: rgba(0,212,170,0.1);  color: #00d4aa; border: 1px solid rgba(0,212,170,0.25); }
    .pill-openai { background: rgba(16,163,127,0.1); color: #10a37f; border: 1px solid rgba(16,163,127,0.25); }
    .pill-gemini { background: rgba(66,133,244,0.1); color: #4285f4; border: 1px solid rgba(66,133,244,0.25); }
    .pill-ollama { background: rgba(255,107,53,0.1); color: #ff6b35; border: 1px solid rgba(255,107,53,0.25); }
    .pill-none   { background: rgba(239,68,68,0.1);  color: #ef4444; border: 1px solid rgba(239,68,68,0.25); }

    /* ── Chat empty state ── */
    .chat-empty {
        text-align: center;
        padding: 3.5rem 2rem;
        color: rgba(255,255,255,0.18);
    }
    .chat-empty-icon { font-size: 2rem; margin-bottom: 0.75rem; opacity: 0.3; }
    .chat-empty-text { font-size: 0.875rem; line-height: 1.7; }

    /* ── Source chunk ── */
    .source-chunk {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        margin: 0.4rem 0;
        font-size: 0.82rem;
        color: rgba(255,255,255,0.55);
        line-height: 1.7;
    }
    .chunk-meta {
        font-size: 0.63rem;
        color: rgba(255,255,255,0.25);
        margin-bottom: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Field rows (extraction) ── */
    .field-row {
        display: flex; justify-content: space-between;
        align-items: flex-start;
        padding: 9px 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        gap: 1.5rem;
    }
    .field-row:last-child { border-bottom: none; }
    .field-name  { color: rgba(255,255,255,0.35); font-size: 0.8rem; font-weight: 500; flex-shrink: 0; min-width: 165px; }
    .field-value { color: #e2e8f0; font-size: 0.85rem; font-weight: 500; text-align: right; word-break: break-word; }
    .field-null  { color: rgba(255,255,255,0.18); font-size: 0.8rem; font-style: italic; text-align: right; }

    /* ── Stats row ── */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.85rem;
        margin-bottom: 1.25rem;
    }
    .stat-box {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .stat-val { font-size: 1.35rem; font-weight: 700; color: #fff; letter-spacing: -0.5px; }
    .stat-key { font-size: 0.65rem; color: rgba(255,255,255,0.28); text-transform: uppercase; letter-spacing: 1px; margin-top: 3px; font-weight: 600; }

    /* ── Log ── */
    .log-line {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: rgba(255,255,255,0.35);
        padding: 2px 0;
        line-height: 1.65;
    }

    /* ── Doc chip ── */
    .doc-chip {
        display: flex; align-items: flex-start; flex-direction: column; gap: 3px;
        background: rgba(99,102,241,0.07);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 10px;
        padding: 9px 13px;
        font-size: 0.82rem;
        color: #a5b4fc;
        font-weight: 500;
        margin-top: 0.5rem;
        width: 100%;
    }
    .doc-chip-id {
        font-size: 0.65rem;
        font-family: 'JetBrains Mono', monospace;
        opacity: 0.5;
        color: rgba(165,180,252,0.65);
    }

    /* ── Divider ── */
    .section-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin: 1.5rem 0;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.02) !important;
        border-radius: 10px !important;
        padding: 4px !important;
        gap: 2px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        color: rgba(255,255,255,0.4) !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        padding: 6px 14px !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99,102,241,0.2) !important;
        color: #a5b4fc !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem !important;
    }

    /* ── Streamlit form ── */
    [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    [data-testid="InputInstructions"] { display: none !important; }

    /* ── Streamlit widget overrides ── */
    .stTextArea textarea {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.09) !important;
        border-radius: 10px !important;
        color: rgba(255,255,255,0.85) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        padding: 0.8rem 1rem !important;
        line-height: 1.5 !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(99,102,241,0.45) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.08) !important;
    }
    .stButton > button, .stFormSubmitButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all 0.15s ease !important;
        letter-spacing: 0.1px !important;
    }
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primaryFormSubmit"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border: none !important;
        font-size: 0.88rem !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button[kind="primaryFormSubmit"]:hover {
        opacity: 0.88 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(99,102,241,0.3) !important;
    }
    .stButton > button[kind="secondary"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: rgba(255,255,255,0.6) !important;
        font-size: 0.85rem !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(255,255,255,0.07) !important;
        border-color: rgba(255,255,255,0.18) !important;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
        border-radius: 4px !important;
    }
    .stProgress { background: rgba(255,255,255,0.05) !important; border-radius: 4px; }
    div[data-testid="stFileUploader"] {
        background: rgba(99,102,241,0.03) !important;
        border: 1.5px dashed rgba(99,102,241,0.18) !important;
        border-radius: 12px !important;
    }
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
        font-size: 0.8rem !important;
        color: rgba(255,255,255,0.4) !important;
    }
    hr { border-color: rgba(255,255,255,0.05) !important; }
    .stSuccess { background: rgba(34,197,94,0.07) !important; border: 1px solid rgba(34,197,94,0.18) !important; border-radius: 10px !important; color: #86efac !important; }
    .stError   { background: rgba(239,68,68,0.07) !important; border: 1px solid rgba(239,68,68,0.18) !important; border-radius: 10px !important; color: #fca5a5 !important; }
    .stInfo    { background: rgba(99,102,241,0.07) !important; border: 1px solid rgba(99,102,241,0.18) !important; border-radius: 10px !important; color: #a5b4fc !important; }

    /* ── Chat counter badge ── */
    .tab-badge {
        display: inline-flex; align-items: center; gap: 5px;
    }
    .badge-count {
        background: rgba(99,102,241,0.25);
        color: #a5b4fc;
        font-size: 0.62rem;
        font-weight: 700;
        padding: 1px 6px;
        border-radius: 999px;
        border: 1px solid rgba(99,102,241,0.3);
    }
    .badge-new {
        background: rgba(34,197,94,0.2);
        color: #22c55e;
        font-size: 0.62rem;
        font-weight: 700;
        padding: 1px 6px;
        border-radius: 999px;
        border: 1px solid rgba(34,197,94,0.3);
    }

    /* ── Extraction empty ── */
    .extract-empty {
        text-align: center;
        padding: 3rem 2rem;
        color: rgba(255,255,255,0.18);
        font-size: 0.85rem;
        line-height: 1.7;
    }

    /* ── Footer ── */
    .site-footer {
        margin-top: 3rem;
        padding: 1.5rem 0 1rem 0;
        border-top: 1px solid rgba(255,255,255,0.04);
        text-align: center;
    }
    .footer-text {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.18);
        letter-spacing: 0.3px;
        line-height: 1.8;
    }
    .footer-name { color: rgba(255,255,255,0.3); font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  CLIENT-SIDE EMBEDDING PIPELINE
# ════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading embedding model (BAAI/bge-small-en)…")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en")


SECTION_KEYWORDS = [
    r"pickup", r"pick[\s\-]?up", r"drop[\s\-]?off", r"delivery",
    r"consignee", r"shipper", r"rate breakdown", r"freight charges",
    r"special instructions", r"bill of lading", r"pro number",
    r"carrier", r"equipment", r"weight", r"hazmat", r"commodity",
]
SECTION_PATTERN = re.compile(
    r"(?i)^.*(" + "|".join(SECTION_KEYWORDS) + r").*$",
    re.MULTILINE,
)
CHARS_PER_TOKEN = 4
MAX_CHUNKS = 100


def _parse_pdf(file_bytes: bytes) -> str:
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append(f"[Page {i + 1}]\n{text}")
        doc.close()
        full = "\n\n".join(pages)
        if len(full.strip()) > 50:
            return full
        raise ValueError("Insufficient text from PyMuPDF")
    except Exception:
        import pdfplumber
        pages = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"[Page {i + 1}]\n{text}")
        return "\n\n".join(pages)


def _parse_docx(file_bytes: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(
                cell.text.strip() for cell in row.cells if cell.text.strip()
            )
            if row_text:
                parts.append(row_text)
    return "\n".join(parts)


def _clean_text(text: str) -> str:
    text = re.sub(r"[^\x09\x0a\x0d\x20-\x7e\u00a0-\ufffd]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(lines)


def _chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[dict]:
    chunk_size_chars = chunk_size * CHARS_PER_TOKEN
    overlap_chars = overlap * CHARS_PER_TOKEN

    page_map: dict[int, int] = {}
    page_pattern = re.compile(r"\[Page (\d+)\]")
    clean_parts, current_pos = [], 0
    for match in page_pattern.finditer(text):
        clean_parts.append(text[current_pos:match.start()])
        page_map[sum(len(p) for p in clean_parts)] = int(match.group(1))
        current_pos = match.end()
    clean_parts.append(text[current_pos:])
    clean_text_no_tags = "".join(clean_parts)

    def get_page(offset: int) -> int:
        page = 1
        for o, p in sorted(page_map.items()):
            if o <= offset:
                page = p
            else:
                break
        return page

    boundaries = [0]
    for m in SECTION_PATTERN.finditer(clean_text_no_tags):
        if m.start() - boundaries[-1] > 200:
            boundaries.append(m.start())
    boundaries.append(len(clean_text_no_tags))

    raw_sections = [
        (clean_text_no_tags[boundaries[i]:boundaries[i + 1]].strip(), boundaries[i])
        for i in range(len(boundaries) - 1)
        if clean_text_no_tags[boundaries[i]:boundaries[i + 1]].strip()
    ]

    final_chunks, chunk_idx = [], 0
    for section_text, section_start in raw_sections:
        if len(section_text) <= chunk_size_chars:
            final_chunks.append({"text": section_text, "chunk_index": chunk_idx, "page": get_page(section_start)})
            chunk_idx += 1
        else:
            start = 0
            while start < len(section_text):
                end = min(start + chunk_size_chars, len(section_text))
                chunk = section_text[start:end].strip()
                if chunk:
                    final_chunks.append({"text": chunk, "chunk_index": chunk_idx, "page": get_page(section_start + start)})
                    chunk_idx += 1
                if end == len(section_text):
                    break
                start += chunk_size_chars - overlap_chars

    MIN_CHUNK_CHARS = 150
    merged = []
    for chunk in final_chunks:
        if merged and len(chunk["text"]) < MIN_CHUNK_CHARS:
            merged[-1]["text"] += " " + chunk["text"]
        else:
            merged.append(chunk)
    for i, c in enumerate(merged):
        c["chunk_index"] = i
    return merged


def process_document(file_bytes: bytes, filename: str):
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        raw_text = _parse_pdf(file_bytes)
    elif suffix in (".docx", ".doc"):
        raw_text = _parse_docx(file_bytes)
    elif suffix == ".txt":
        raw_text = file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    cleaned = _clean_text(raw_text)
    chunks = _chunk_text(cleaned)

    if len(chunks) > MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]
        for i, c in enumerate(chunks):
            c["chunk_index"] = i

    model = load_embedding_model()
    prefixed = [f"passage: {c['text']}" for c in chunks]
    embeddings = model.encode(
        prefixed, batch_size=32, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)

    doc_id = str(uuid.uuid4())
    return doc_id, chunks, embeddings


def embed_query(question: str) -> np.ndarray:
    model = load_embedding_model()
    return model.encode(
        [f"query: {question}"],
        show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)


# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "doc_id":           None,
    "filename":         None,
    "chat_history":     [],        # list of chat turn dicts
    "extract_result":   None,
    "active_tab":       0,         # 0 = Chat, 1 = Extraction
    "backend_url":      os.getenv("BACKEND_URL", "http://localhost:8000"),
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_url() -> str:
    return st.session_state.backend_url.rstrip("/")


def backend_check() -> tuple[bool, str]:
    url = get_url()
    try:
        r = requests.get(f"{url}/health", timeout=5)
        if r.status_code == 200:
            return True, f"Connected · {url}"
        return False, f"HTTP {r.status_code} at {url}"
    except requests.exceptions.ConnectionError:
        return False, f"Connection refused at {url}"
    except requests.exceptions.Timeout:
        return False, f"Timed out at {url}"
    except Exception as e:
        return False, str(e)


def provider_badge(provider: str) -> str:
    p = provider.lower()
    css  = {"groq": "pill-groq", "openai": "pill-openai", "gemini": "pill-gemini", "ollama": "pill-ollama"}.get(p, "pill-none")
    icon = {"groq": "⚡", "openai": "◆", "gemini": "✦", "ollama": "🦙"}.get(p, "•")
    return f'<span class="provider-pill {css}">{icon} {provider}</span>'


def conf_badge(score: float) -> str:
    if score >= 0.70:
        cls, label = "conf-high", f"{score:.0%} High"
    elif score >= 0.45:
        cls, label = "conf-medium", f"{score:.0%} Med"
    else:
        cls, label = "conf-low", f"{score:.0%} Low"
    return f'<span class="chat-conf-badge {cls}">{label}</span>'


def fmt_time(iso: str) -> str:
    try:
        dt = datetime.datetime.fromisoformat(iso)
        return dt.strftime("%H:%M")
    except Exception:
        return ""


# ── Backend status ────────────────────────────────────────────────────────────
alive, alive_detail = backend_check()

# ── Header ───────────────────────────────────────────────────────────────────
dot        = '<span class="dot-green"></span>' if alive else '<span class="dot-red"></span>'
status_txt = "API Online" if alive else "API Offline"

st.markdown(f"""
<div class="site-header">
  <div class="header-inner">
    <div class="header-brand">
      <div class="header-title">Ultra Doc-Intelligence</div>
      <div class="header-subtitle">Transportation Management System · Document Analysis</div>
    </div>
    <div class="header-status">{dot} {status_txt}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Offline warning ───────────────────────────────────────────────────────────
if not alive:
    st.markdown(f"""
<div class="card-warning">
  <div style="font-size:0.9rem;color:#fde68a;font-weight:600;margin-bottom:6px">Backend Offline</div>
  <div style="font-size:0.82rem;color:rgba(255,255,255,0.5);line-height:1.8">
    <b>Detail:</b> {alive_detail}<br><br>
    <b>Start the backend:</b><br>
    <code style="background:rgba(0,0,0,0.35);padding:2px 10px;border-radius:4px;color:#fde68a;font-family:'JetBrains Mono',monospace;font-size:0.8rem">cd backend &amp;&amp; uvicorn main:app --host 127.0.0.1 --port 8000 --reload</code>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Two-column layout ─────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.65], gap="large")


# ════════════════════════════════════════════════════════════
#  LEFT PANEL — Upload · Ask · Extract
# ════════════════════════════════════════════════════════════
with left_col:

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Document</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload",
        type=["pdf", "docx", "txt"],
        help="Bill of Lading, Rate Confirmation, Carrier RC, or any logistics document",
        label_visibility="collapsed",
        disabled=not alive,
    )

    if uploaded_file and st.session_state.filename != uploaded_file.name:
        with st.spinner(f"Parsing & embedding **{uploaded_file.name}**…"):
            try:
                file_bytes = uploaded_file.getvalue()
                doc_id, chunks, embeddings = process_document(file_bytes, uploaded_file.name)

                payload = {
                    "doc_id":     doc_id,
                    "filename":   uploaded_file.name,
                    "chunks":     chunks,
                    "embeddings": embeddings.tolist(),
                }
                resp = requests.post(f"{get_url()}/upload_embeddings", json=payload, timeout=120)

                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.doc_id         = data["doc_id"]
                    st.session_state.filename       = uploaded_file.name
                    # Clear history when new doc is uploaded
                    st.session_state.chat_history   = []
                    st.session_state.extract_result = None
                    st.session_state.active_tab     = 0
                    st.success(f"**{uploaded_file.name}** — {data['chunks_count']} chunks indexed")
                else:
                    st.error(f"Upload failed: {resp.json().get('detail', resp.text)}")
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach backend at {get_url()}.")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.doc_id:
        st.markdown(
            f'<div class="doc-chip">'
            f'<span>📄 {st.session_state.filename}</span>'
            f'<span class="doc-chip-id">{st.session_state.doc_id}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Ask ───────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Ask a Question</div>', unsafe_allow_html=True)

    no_doc = st.session_state.doc_id is None or not alive

    with st.form(key="ask_form", clear_on_submit=True):
        question = st.text_area(
            "Question",
            placeholder="e.g. Who is the consignee?\nWhat is the freight rate?",
            disabled=no_doc,
            label_visibility="collapsed",
            height=100,
        )
        ask_btn = st.form_submit_button(
            "Search Document",
            disabled=no_doc,
            use_container_width=True,
            type="primary",
        )

    # Enter to submit
    components.html("""
        <script>
        const doc = window.parent.document;
        const textareas = doc.querySelectorAll('textarea');
        textareas.forEach(ta => {
            if (ta.dataset.hasEnterListener) return;
            ta.dataset.hasEnterListener = "true";
            ta.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    const btn = doc.querySelector('button[kind="primaryFormSubmit"]');
                    if (btn) btn.click();
                }
            });
        });
        </script>
    """, height=0, width=0)

    if ask_btn and question.strip():
        with st.spinner("Embedding query & retrieving answer…"):
            try:
                query_vec = embed_query(question.strip())
                resp = requests.post(
                    f"{get_url()}/ask",
                    json={
                        "doc_id":          st.session_state.doc_id,
                        "question":        question.strip(),
                        "query_embedding": query_vec.tolist(),
                    },
                    timeout=90,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    # Append to persistent chat history
                    st.session_state.chat_history.append({
                        "question":           question.strip(),
                        "answer":             result.get("answer", ""),
                        "confidence":         result.get("confidence", 0.0),
                        "sources":            result.get("sources", []),
                        "provider":           result.get("provider", "None"),
                        "logs":               result.get("logs", []),
                        "guardrail_triggered": result.get("guardrail_triggered", False),
                        "timestamp":          datetime.datetime.now().isoformat(),
                    })
                    st.session_state.active_tab = 0   # stay on Chat tab
                else:
                    st.error(f"Error: {resp.json().get('detail', resp.text)}")
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach backend at {get_url()}.")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Structured Extraction ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">Structured Extraction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.8rem;color:rgba(255,255,255,0.28);margin-bottom:0.7rem;line-height:1.6">'
        'Extract key shipment fields using hybrid regex + LLM analysis.'
        '</div>',
        unsafe_allow_html=True,
    )

    extract_btn = st.button(
        "Extract Shipment Data",
        disabled=(st.session_state.doc_id is None or not alive),
        use_container_width=True,
    )

    if extract_btn:
        with st.spinner("Running extraction (regex + LLM)…"):
            try:
                resp = requests.post(
                    f"{get_url()}/extract",
                    json={"doc_id": st.session_state.doc_id},
                    timeout=120,
                )
                if resp.status_code == 200:
                    st.session_state.extract_result = resp.json()
                    st.session_state.active_tab     = 1   # switch to Extraction tab
                else:
                    st.error(f"Error: {resp.json().get('detail', resp.text)}")
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach backend at {get_url()}.")
            except Exception as e:
                st.error(f"Error: {e}")

    # ── How it works ──────────────────────────────────────────────────────────
    if not st.session_state.doc_id:
        st.markdown("""
<div class="card" style="margin-top:1.25rem">
  <div style="font-size:0.78rem;color:rgba(255,255,255,0.4);font-weight:600;margin-bottom:0.6rem">How it works</div>
  <div style="font-size:0.78rem;color:rgba(255,255,255,0.25);line-height:2">
    1. Upload a PDF, DOCX, or TXT logistics document<br>
    2. Document is parsed &amp; embedded in your session<br>
    3. Ask any number of questions — full chat history is kept<br>
    4. Run structured extraction at any time — chat stays intact<br><br>
    <span style="color:rgba(255,255,255,0.18)">Supports: Bill of Lading · Rate Confirmation · Carrier RC</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Clear Chat button (only when there's history) ─────────────────────────
    if st.session_state.chat_history:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        if st.button("🗑 Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.active_tab   = 0
            st.rerun()


# ════════════════════════════════════════════════════════════
#  RIGHT PANEL — Tabbed: Chat History | Extraction Results
# ════════════════════════════════════════════════════════════
with right_col:

    n_msgs     = len(st.session_state.chat_history)
    has_extract = st.session_state.extract_result is not None

    # Build tab labels with live counts
    chat_label    = f"💬 Chat  ({n_msgs})" if n_msgs else "💬 Chat"
    extract_label = f"📋 Extraction  ✓" if has_extract else "📋 Extraction"

    tab_chat, tab_extract = st.tabs([chat_label, extract_label])


    # ── TAB 1: Chat History ───────────────────────────────────────────────────
    with tab_chat:

        if not st.session_state.chat_history:
            st.markdown("""
<div class="chat-empty">
  <div class="chat-empty-icon">💬</div>
  <div class="chat-empty-text">
    No questions yet.<br>Upload a document and start asking questions —<br>your full conversation will appear here.
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            for i, turn in enumerate(st.session_state.chat_history):
                q         = turn["question"]
                a         = turn["answer"]
                conf      = turn["confidence"]
                provider  = turn["provider"]
                sources   = turn["sources"]
                logs      = turn["logs"]
                guardrail = turn["guardrail_triggered"]
                ts        = fmt_time(turn.get("timestamp", ""))

                # ── Question bubble ───────────────────────────────────────────
                st.markdown(f"""
<div class="chat-question-row">
  <div class="chat-question-bubble">{q}</div>
</div>
""", unsafe_allow_html=True)

                # ── Answer bubble ─────────────────────────────────────────────
                if guardrail:
                    bubble_class = "chat-answer-guardrail"
                    icon = "⚠️ "
                else:
                    bubble_class = "chat-answer-bubble"
                    icon = ""

                meta_html = f"""
<div class="chat-meta">
  <span class="chat-timestamp">{ts}</span>
  {conf_badge(conf)}
  {provider_badge(provider)}
</div>
"""
                st.markdown(f"""
<div class="chat-answer-row">
  <div class="{bubble_class}">
    {icon}{a}
    {meta_html}
  </div>
</div>
""", unsafe_allow_html=True)

                # ── Expandable sources + logs ─────────────────────────────────
                if sources or logs:
                    with st.expander(
                        f"Sources ({len(sources)})  ·  Msg #{i + 1}",
                        expanded=False,
                    ):
                        for src in sources:
                            page_info = f"Page {src['page']}" if src.get("page") else "—"
                            st.markdown(
                                f'<div class="source-chunk">'
                                f'<div class="chunk-meta">Chunk #{src["chunk_index"]} · {page_info} · {src["similarity"]:.0%} similarity</div>'
                                f'{src["text"][:500]}{"…" if len(src["text"]) > 500 else ""}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        if logs:
                            st.markdown(
                                '<div style="margin-top:0.5rem;margin-bottom:0.25rem;font-size:0.65rem;'
                                'color:rgba(255,255,255,0.2);letter-spacing:1px;text-transform:uppercase">Logs</div>',
                                unsafe_allow_html=True,
                            )
                            for log in logs:
                                st.markdown(f'<div class="log-line">{log}</div>', unsafe_allow_html=True)

                # Thin divider between turns (skip after last)
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown(
                        '<hr style="border-color:rgba(255,255,255,0.04);margin:0.25rem 0 1rem 0">',
                        unsafe_allow_html=True,
                    )


    # ── TAB 2: Extraction Results ─────────────────────────────────────────────
    with tab_extract:

        if not has_extract:
            st.markdown("""
<div class="extract-empty">
  <div style="font-size:2rem;opacity:0.25;margin-bottom:0.75rem">📋</div>
  Click <b>Extract Shipment Data</b> in the left panel<br>
  to automatically pull key shipment fields from the document.<br><br>
  <span style="color:rgba(255,255,255,0.12)">Chat history is preserved while you run extraction.</span>
</div>
""", unsafe_allow_html=True)

        else:
            extract      = st.session_state.extract_result
            data         = extract.get("data", {})
            provider     = extract.get("provider", "None")
            ext_logs     = extract.get("logs", [])
            non_null     = sum(1 for v in data.values() if v is not None)
            total        = len(data)
            completeness = non_null / total if total else 0

            st.markdown(f"""
<div class="stats-grid">
  <div class="stat-box">
    <div class="stat-val" style="color:{'#22c55e' if completeness>=0.7 else '#f59e0b' if completeness>=0.45 else '#ef4444'}">{completeness:.0%}</div>
    <div class="stat-key">Completeness</div>
  </div>
  <div class="stat-box">
    <div style="margin-top:4px">{provider_badge(provider)}</div>
    <div class="stat-key" style="margin-top:6px">LLM Provider</div>
  </div>
  <div class="stat-box">
    <div class="stat-val">{non_null}/{total}</div>
    <div class="stat-key">Fields Found</div>
  </div>
</div>
""", unsafe_allow_html=True)

            st.progress(completeness)
            st.markdown('<div class="section-label" style="margin-top:0.5rem">Extracted Fields</div>', unsafe_allow_html=True)

            FIELD_LABELS = {
                "shipment_id":       "Shipment / BOL ID",
                "shipper":           "Shipper",
                "consignee":         "Consignee",
                "pickup_datetime":   "Pickup Date / Time",
                "delivery_datetime": "Delivery Date / Time",
                "equipment_type":    "Equipment Type",
                "mode":              "Mode",
                "rate":              "Rate",
                "currency":          "Currency",
                "weight":            "Weight",
                "carrier_name":      "Carrier Name",
            }

            rows_html = ""
            for field, label in FIELD_LABELS.items():
                val = data.get(field)
                if val is not None:
                    display = f"${val:,.2f}" if field == "rate" and isinstance(val, (int, float)) else str(val)
                    rows_html += (
                        f'<div class="field-row">'
                        f'<span class="field-name">{label}</span>'
                        f'<span class="field-value">{display}</span>'
                        f'</div>'
                    )
                else:
                    rows_html += (
                        f'<div class="field-row">'
                        f'<span class="field-name">{label}</span>'
                        f'<span class="field-null">not found</span>'
                        f'</div>'
                    )

            st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)

            with st.expander("Raw JSON", expanded=False):
                st.json(data)

            if ext_logs:
                with st.expander("Extraction Logs", expanded=False):
                    for log in ext_logs:
                        st.markdown(f'<div class="log-line">{log}</div>', unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-footer">
  <div class="footer-text">
    Ultra Doc-Intelligence &nbsp;&middot;&nbsp; AI-Powered Document Analysis for Transportation Management<br>
    &copy; 2025 <span class="footer-name">Manvik Siddhpura</span>. All Rights Reserved.
  </div>
</div>
""", unsafe_allow_html=True)
