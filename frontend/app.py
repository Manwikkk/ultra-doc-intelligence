"""
app.py — Streamlit frontend for Ultra Doc-Intelligence (v3).
Professional enterprise UI — persistent chat, tabbed results.
"""
from __future__ import annotations

import io, os, re, uuid, base64, logging, datetime
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)


def get_favicon():
    p = Path(__file__).parent / "Document.png"
    if p.exists():
        return f"data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"
    return "📄"


st.set_page_config(
    page_title="Ultra Doc-Intelligence",
    page_icon=get_favicon(),
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
#  DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif !important; }
.stApp { background: #0d1117 !important; }
.stApp > header { display: none !important; }
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="InputInstructions"] { display: none !important; }

/* ── App Shell ── */
.app-shell {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    background: #0d1117;
}

/* ── Top Navigation Bar ── */
.topbar {
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(13,17,23,0.92);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 0 2rem;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.topbar-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.topbar-logo {
    width: 28px; height: 28px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; color: #fff; font-weight: 700;
}
.topbar-name {
    font-size: 0.9rem;
    font-weight: 700;
    color: #f0f6fc;
    letter-spacing: -0.3px;
}
.topbar-divider {
    width: 1px; height: 18px;
    background: rgba(255,255,255,0.1);
}
.topbar-sub {
    font-size: 0.73rem;
    color: rgba(255,255,255,0.25);
    font-weight: 400;
    letter-spacing: 0.5px;
}
.status-chip {
    display: flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-weight: 500;
    color: rgba(255,255,255,0.4);
}
.dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.dot-green { background: #22c55e; box-shadow: 0 0 6px rgba(34,197,94,0.5); }
.dot-red   { background: #ef4444; box-shadow: 0 0 6px rgba(239,68,68,0.4); }

/* ── Main content area ── */
.main-content {
    display: flex;
    flex: 1;
    gap: 0;
    padding: 0;
}

/* ── Left Panel ── */
.left-panel {
    width: 360px;
    min-width: 360px;
    border-right: 1px solid rgba(255,255,255,0.06);
    padding: 1.5rem 1.25rem;
    position: sticky;
    top: 56px;
    height: calc(100vh - 56px);
    overflow-y: auto;
    background: #0d1117;
}

/* ── Right Panel ── */
.right-panel {
    flex: 1;
    padding: 1.5rem 2rem;
    overflow-y: auto;
}

/* ── Panel title ── */
.panel-title {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.22);
    margin-bottom: 0.75rem;
}

/* ── Upload zone ── */
div[data-testid="stFileUploader"] {
    background: rgba(99,102,241,0.03) !important;
    border: 1.5px dashed rgba(99,102,241,0.18) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.35) !important;
}

/* ── Doc chip ── */
.doc-chip {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(34,197,94,0.05);
    border: 1px solid rgba(34,197,94,0.15);
    border-radius: 10px;
    padding: 10px 13px;
    margin-top: 0.6rem;
}
.doc-chip-icon {
    font-size: 1.1rem;
    flex-shrink: 0;
}
.doc-chip-body { flex: 1; min-width: 0; }
.doc-chip-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: #86efac;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.doc-chip-id {
    font-size: 0.62rem;
    color: rgba(134,239,172,0.45);
    font-family: 'JetBrains Mono', monospace;
    margin-top: 2px;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin: 1.25rem 0;
}

/* ── Form area ── */
[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: rgba(255,255,255,0.88) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    padding: 0.8rem 1rem !important;
    line-height: 1.55 !important;
    resize: none !important;
    transition: border-color 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.07) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder { color: rgba(255,255,255,0.2) !important; }

/* ── Buttons ── */
.stButton > button,
.stFormSubmitButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 9px !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.1px !important;
    font-size: 0.85rem !important;
}
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primaryFormSubmit"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    color: #fff !important;
}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primaryFormSubmit"]:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.35) !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: rgba(255,255,255,0.55) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.18) !important;
    color: rgba(255,255,255,0.75) !important;
}

/* ── Extract button special ── */
.extract-btn .stButton > button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: rgba(255,255,255,0.7) !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    border-radius: 4px !important;
}
.stProgress { background: rgba(255,255,255,0.05) !important; border-radius: 4px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
    gap: 0 !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    color: rgba(255,255,255,0.35) !important;
    font-size: 0.83rem !important;
    font-weight: 600 !important;
    padding: 10px 18px !important;
    margin-right: 4px !important;
    transition: color 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: rgba(255,255,255,0.65) !important;
    background: rgba(255,255,255,0.03) !important;
}
.stTabs [aria-selected="true"] {
    color: #a5b4fc !important;
    border-bottom-color: #6366f1 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 1.25rem 0 0 0 !important;
    background: transparent !important;
}

/* ── Q&A Thread ── */
.qa-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.qa-card:hover { border-color: rgba(255,255,255,0.12); }
.qa-question-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 1rem;
}
.qa-q-icon {
    width: 26px; height: 26px; border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.65rem; font-weight: 700; color: #fff;
    flex-shrink: 0; margin-top: 1px;
}
.qa-question-text {
    font-size: 0.88rem;
    font-weight: 600;
    color: rgba(255,255,255,0.88);
    line-height: 1.5;
    flex: 1;
}
.qa-turn-time {
    font-size: 0.65rem;
    color: rgba(255,255,255,0.2);
    font-family: 'JetBrains Mono', monospace;
    flex-shrink: 0;
    margin-top: 4px;
}

/* ── Metrics strip ── */
.metrics-strip {
    display: flex;
    gap: 0;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 1rem;
}
.metric-cell {
    flex: 1;
    padding: 10px 14px;
    border-right: 1px solid rgba(255,255,255,0.06);
    text-align: center;
}
.metric-cell:last-child { border-right: none; }
.metric-label {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 700;
    color: rgba(255,255,255,0.22);
    margin-bottom: 5px;
}
.metric-value {
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: -0.3px;
    line-height: 1;
}
.conf-high   { color: #22c55e; }
.conf-medium { color: #f59e0b; }
.conf-low    { color: #ef4444; }

/* ── Provider pill ── */
.provider-pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.2px;
}
.pill-groq   { background: rgba(0,212,170,0.12);  color: #00d4aa; border: 1px solid rgba(0,212,170,0.25); }
.pill-openai { background: rgba(16,163,127,0.12); color: #10a37f; border: 1px solid rgba(16,163,127,0.25); }
.pill-gemini { background: rgba(66,133,244,0.12); color: #4285f4; border: 1px solid rgba(66,133,244,0.25); }
.pill-ollama { background: rgba(255,107,53,0.12); color: #ff6b35; border: 1px solid rgba(255,107,53,0.25); }
.pill-none   { background: rgba(239,68,68,0.10);  color: #ef4444; border: 1px solid rgba(239,68,68,0.25); }

/* ── Answer text ── */
.qa-answer-label {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
    color: rgba(99,102,241,0.6);
    margin-bottom: 0.5rem;
}
.qa-answer-text {
    font-size: 0.9rem;
    line-height: 1.85;
    color: rgba(255,255,255,0.82);
}
.qa-guardrail {
    background: rgba(239,68,68,0.06);
    border: 1px solid rgba(239,68,68,0.2);
    border-left: 3px solid #ef4444;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    font-size: 0.875rem;
    color: #fca5a5;
    line-height: 1.65;
}

/* ── Source chunks ── */
.source-item {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.5);
    line-height: 1.7;
}
.source-meta {
    font-size: 0.63rem;
    color: rgba(255,255,255,0.22);
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.4rem;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
    color: rgba(255,255,255,0.38) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}

/* ── Extraction ── */
.ext-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.ext-stat {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.ext-stat-val { font-size: 1.4rem; font-weight: 700; color: #fff; letter-spacing: -0.5px; }
.ext-stat-key {
    font-size: 0.63rem; color: rgba(255,255,255,0.25);
    text-transform: uppercase; letter-spacing: 1px;
    margin-top: 4px; font-weight: 600;
}
.ext-field-row {
    display: flex; justify-content: space-between;
    align-items: flex-start;
    padding: 9px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    gap: 1.5rem;
}
.ext-field-row:last-child { border-bottom: none; }
.ext-field-label { color: rgba(255,255,255,0.32); font-size: 0.82rem; font-weight: 500; flex-shrink: 0; min-width: 170px; }
.ext-field-val   { color: #e2e8f0; font-size: 0.85rem; font-weight: 600; text-align: right; word-break: break-word; }
.ext-field-null  { color: rgba(255,255,255,0.15); font-size: 0.8rem; font-style: italic; text-align: right; }
.ext-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
}

/* ── Empty states ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: rgba(255,255,255,0.18);
}
.empty-icon { font-size: 2.5rem; margin-bottom: 0.85rem; opacity: 0.25; }
.empty-text { font-size: 0.875rem; line-height: 1.75; }
.empty-hint { font-size: 0.78rem; color: rgba(255,255,255,0.12); margin-top: 0.5rem; }

/* ── Warning banner ── */
.offline-banner {
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.2);
    border-left: 3px solid #f59e0b;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
    font-size: 0.84rem;
    color: rgba(255,255,255,0.65);
    line-height: 1.75;
}

/* ── Log lines ── */
.log-entry {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: rgba(255,255,255,0.3);
    padding: 2px 0;
    line-height: 1.6;
}

/* ── Streamlit alerts ── */
.stSuccess { background: rgba(34,197,94,0.07) !important; border: 1px solid rgba(34,197,94,0.18) !important; border-radius: 10px !important; color: #86efac !important; font-size: 0.85rem !important; }
.stError   { background: rgba(239,68,68,0.07) !important; border: 1px solid rgba(239,68,68,0.18) !important; border-radius: 10px !important; color: #fca5a5 !important; font-size: 0.85rem !important; }

/* ── Info card ── */
.info-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-top: 1rem;
}
.info-card-title {
    font-size: 0.72rem; font-weight: 700;
    color: rgba(255,255,255,0.3); margin-bottom: 0.5rem;
    text-transform: uppercase; letter-spacing: 0.8px;
}
.info-card-body {
    font-size: 0.78rem; color: rgba(255,255,255,0.22);
    line-height: 1.9;
}

/* ── Footer ── */
.app-footer {
    border-top: 1px solid rgba(255,255,255,0.05);
    padding: 1rem 2rem;
    text-align: center;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.15);
    letter-spacing: 0.2px;
}
.app-footer span { color: rgba(255,255,255,0.25); font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  EMBEDDING PIPELINE  (client-side)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en")


SECTION_KEYWORDS = [
    r"pickup", r"pick[\s\-]?up", r"drop[\s\-]?off", r"delivery",
    r"consignee", r"shipper", r"rate breakdown", r"freight charges",
    r"special instructions", r"bill of lading", r"pro number",
    r"carrier", r"equipment", r"weight", r"hazmat", r"commodity",
]
_SECTION_RE = re.compile(r"(?i)^.*(" + "|".join(SECTION_KEYWORDS) + r").*$", re.MULTILINE)
CHARS_PER_TOKEN = 4
MAX_CHUNKS = 100


def _parse_pdf(b: bytes) -> str:
    try:
        import fitz
        doc = fitz.open(stream=b, filetype="pdf")
        pages = [f"[Page {i+1}]\n{p.get_text('text')}" for i, p in enumerate(doc) if p.get_text("text").strip()]
        doc.close()
        full = "\n\n".join(pages)
        if len(full.strip()) > 50:
            return full
        raise ValueError()
    except Exception:
        import pdfplumber
        pages = []
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            for i, page in enumerate(pdf.pages):
                t = page.extract_text() or ""
                if t.strip():
                    pages.append(f"[Page {i+1}]\n{t}")
        return "\n\n".join(pages)


def _parse_docx(b: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(b))
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            r = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
            if r:
                parts.append(r)
    return "\n".join(parts)


def _clean(text: str) -> str:
    text = re.sub(r"[^\x09\x0a\x0d\x20-\x7e\u00a0-\ufffd]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(re.sub(r"[ \t]+", " ", l).strip() for l in text.splitlines())


def _chunk(text: str, size=600, overlap=100) -> list[dict]:
    cs, ov = size * CHARS_PER_TOKEN, overlap * CHARS_PER_TOKEN
    page_map: dict[int, int] = {}
    parts, pos = [], 0
    for m in re.finditer(r"\[Page (\d+)\]", text):
        parts.append(text[pos:m.start()])
        page_map[sum(len(p) for p in parts)] = int(m.group(1))
        pos = m.end()
    parts.append(text[pos:])
    clean = "".join(parts)

    def pg(off):
        p = 1
        for o, v in sorted(page_map.items()):
            if o <= off: p = v
            else: break
        return p

    bounds = [0]
    for m in _SECTION_RE.finditer(clean):
        if m.start() - bounds[-1] > 200:
            bounds.append(m.start())
    bounds.append(len(clean))

    sections = [(clean[bounds[i]:bounds[i+1]].strip(), bounds[i])
                for i in range(len(bounds)-1)
                if clean[bounds[i]:bounds[i+1]].strip()]

    chunks, idx = [], 0
    for sec, ss in sections:
        if len(sec) <= cs:
            chunks.append({"text": sec, "chunk_index": idx, "page": pg(ss)}); idx += 1
        else:
            s = 0
            while s < len(sec):
                e = min(s + cs, len(sec))
                c = sec[s:e].strip()
                if c:
                    chunks.append({"text": c, "chunk_index": idx, "page": pg(ss+s)}); idx += 1
                if e == len(sec): break
                s += cs - ov

    merged = []
    for c in chunks:
        if merged and len(c["text"]) < 150:
            merged[-1]["text"] += " " + c["text"]
        else:
            merged.append(c)
    for i, c in enumerate(merged):
        c["chunk_index"] = i
    return merged


def process_document(file_bytes: bytes, filename: str):
    suf = Path(filename).suffix.lower()
    if suf == ".pdf":          raw = _parse_pdf(file_bytes)
    elif suf in (".docx",".doc"): raw = _parse_docx(file_bytes)
    elif suf == ".txt":        raw = file_bytes.decode("utf-8", errors="replace")
    else: raise ValueError(f"Unsupported: {suf}")

    chunks = _chunk(_clean(raw))
    if len(chunks) > MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]
        for i, c in enumerate(chunks): c["chunk_index"] = i

    model = load_embedding_model()
    embs = model.encode(
        [f"passage: {c['text']}" for c in chunks],
        batch_size=32, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    return str(uuid.uuid4()), chunks, embs


def embed_query(q: str) -> np.ndarray:
    return load_embedding_model().encode(
        [f"query: {q}"], show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────
_defaults = {
    "doc_id":         None,
    "filename":       None,
    "chat_history":   [],        # [{question, answer, confidence, sources, provider, logs, guardrail_triggered, timestamp}]
    "extract_result": None,
    "backend_url":    os.getenv("BACKEND_URL", "http://localhost:8000"),
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def get_url() -> str:
    return st.session_state.backend_url.rstrip("/")


def backend_check():
    try:
        r = requests.get(f"{get_url()}/health", timeout=5)
        return r.status_code == 200, f"Connected · {get_url()}"
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to {get_url()}"
    except Exception as e:
        return False, str(e)


def provider_pill(p: str) -> str:
    p = p or "None"
    lo = p.lower()
    icons = {"groq": "⚡", "openai": "◆", "gemini": "✦", "ollama": "🦙"}
    css   = {"groq": "pill-groq", "openai": "pill-openai", "gemini": "pill-gemini", "ollama": "pill-ollama"}
    return (f'<span class="provider-pill {css.get(lo,"pill-none")}">'
            f'{icons.get(lo,"•")} {p}</span>')


def conf_color(s: float) -> str:
    return "conf-high" if s >= 0.70 else "conf-medium" if s >= 0.45 else "conf-low"


def fmt_time(iso: str) -> str:
    try:
        return datetime.datetime.fromisoformat(iso).strftime("%H:%M")
    except:
        return ""


# ─────────────────────────────────────────────────────────────
#  TOP BAR
# ─────────────────────────────────────────────────────────────
alive, alive_detail = backend_check()
dot_cls = "dot-green" if alive else "dot-red"
status_label = "Backend Online" if alive else "Backend Offline"

st.markdown(f"""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-logo">AI</div>
    <span class="topbar-name">Ultra Doc-Intelligence</span>
    <div class="topbar-divider"></div>
    <span class="topbar-sub">Transportation Management System</span>
  </div>
  <div class="status-chip">
    <span class="dot {dot_cls}"></span>
    {status_label}
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  TWO-COLUMN LAYOUT
# ─────────────────────────────────────────────────────────────
left_col, right_col = st.columns([5, 9], gap="large")


# ═══════════════════════════════════════════════
#  LEFT PANEL
# ═══════════════════════════════════════════════
with left_col:

    if not alive:
        st.markdown(f"""
<div class="offline-banner">
  <strong style="color:#fde68a">Backend is offline</strong><br>
  {alive_detail}<br><br>
  Start it with:<br>
  <code style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#fde68a">
    cd backend &amp;&amp; uvicorn main:app --port 8000 --reload
  </code>
</div>
""", unsafe_allow_html=True)

    # ── Document ──────────────────────────────────────────────
    st.markdown('<div class="panel-title">Document</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "drop_file",
        type=["pdf", "docx", "txt"],
        help="Upload a Bill of Lading, Rate Confirmation, Carrier RC, or any logistics document",
        label_visibility="collapsed",
        disabled=not alive,
    )

    if uploaded and st.session_state.filename != uploaded.name:
        with st.spinner(f"Processing **{uploaded.name}**…"):
            try:
                doc_id, chunks, embs = process_document(uploaded.getvalue(), uploaded.name)
                r = requests.post(f"{get_url()}/upload_embeddings", json={
                    "doc_id": doc_id, "filename": uploaded.name,
                    "chunks": chunks, "embeddings": embs.tolist(),
                }, timeout=120)
                if r.status_code == 200:
                    d = r.json()
                    st.session_state.doc_id         = d["doc_id"]
                    st.session_state.filename       = uploaded.name
                    st.session_state.chat_history   = []
                    st.session_state.extract_result = None
                    st.success(f"{uploaded.name} · {d['chunks_count']} chunks indexed")
                else:
                    st.error(r.json().get("detail", r.text))
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach backend.")
            except Exception as e:
                st.error(str(e))

    if st.session_state.doc_id:
        fname = st.session_state.filename or ""
        did   = st.session_state.doc_id or ""
        st.markdown(f"""
<div class="doc-chip">
  <div class="doc-chip-icon">📄</div>
  <div class="doc-chip-body">
    <div class="doc-chip-name">{fname}</div>
    <div class="doc-chip-id">{did}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Ask ───────────────────────────────────────────────────
    st.markdown('<div class="panel-title">Ask a Question</div>', unsafe_allow_html=True)

    no_doc = not st.session_state.doc_id or not alive

    with st.form("ask_form", clear_on_submit=True):
        question = st.text_area(
            "q", placeholder="e.g. Who is the consignee?\nWhat is the freight rate?\nWhen is pickup scheduled?",
            disabled=no_doc, label_visibility="collapsed", height=105,
        )
        ask_btn = st.form_submit_button(
            "Ask Question", disabled=no_doc,
            use_container_width=True, type="primary",
        )

    components.html("""<script>
    const d = window.parent.document;
    d.querySelectorAll('textarea').forEach(ta => {
        if (ta.dataset.el) return; ta.dataset.el = '1';
        ta.addEventListener('keydown', e => {
            if (e.key==='Enter' && !e.shiftKey) {
                e.preventDefault();
                d.querySelector('button[kind="primaryFormSubmit"]')?.click();
            }
        });
    });
    </script>""", height=0, width=0)

    if ask_btn and question.strip():
        with st.spinner("Retrieving answer…"):
            try:
                qvec = embed_query(question.strip())
                r = requests.post(f"{get_url()}/ask", json={
                    "doc_id": st.session_state.doc_id,
                    "question": question.strip(),
                    "query_embedding": qvec.tolist(),
                }, timeout=90)
                if r.status_code == 200:
                    res = r.json()
                    st.session_state.chat_history.append({
                        "question":            question.strip(),
                        "answer":              res.get("answer", ""),
                        "confidence":          res.get("confidence", 0.0),
                        "sources":             res.get("sources", []),
                        "provider":            res.get("provider", "None"),
                        "logs":                res.get("logs", []),
                        "guardrail_triggered": res.get("guardrail_triggered", False),
                        "timestamp":           datetime.datetime.now().isoformat(),
                    })
                    st.rerun()
                else:
                    st.error(r.json().get("detail", r.text))
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach backend.")
            except Exception as e:
                st.error(str(e))

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Extraction ────────────────────────────────────────────
    st.markdown('<div class="panel-title">Structured Extraction</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.78rem;color:rgba(255,255,255,0.22);margin-bottom:0.6rem;line-height:1.6">'
        'Extract key shipment fields using regex + LLM analysis.</p>',
        unsafe_allow_html=True,
    )
    if st.button("Extract Shipment Data", disabled=no_doc, use_container_width=True):
        with st.spinner("Extracting structured data…"):
            try:
                r = requests.post(f"{get_url()}/extract",
                                  json={"doc_id": st.session_state.doc_id}, timeout=120)
                if r.status_code == 200:
                    st.session_state.extract_result = r.json()
                    st.rerun()
                else:
                    st.error(r.json().get("detail", r.text))
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach backend.")
            except Exception as e:
                st.error(str(e))

    # ── How it works (no doc yet) ─────────────────────────────
    if not st.session_state.doc_id:
        st.markdown("""
<div class="info-card">
  <div class="info-card-title">How it works</div>
  <div class="info-card-body">
    1. Upload a PDF, DOCX, or TXT document<br>
    2. Ask questions in plain English<br>
    3. Full conversation history is preserved<br>
    4. Run extraction anytime — chat stays intact
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Clear chat ────────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        if st.button("🗑  Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ═══════════════════════════════════════════════
#  RIGHT PANEL — Tabs
# ═══════════════════════════════════════════════
with right_col:

    n       = len(st.session_state.chat_history)
    has_ext = st.session_state.extract_result is not None

    chat_lbl = f"💬  Conversation  ({n})" if n else "💬  Conversation"
    ext_lbl  = "📋  Extraction  ✓" if has_ext else "📋  Extraction"

    tab_chat, tab_ext = st.tabs([chat_lbl, ext_lbl])

    # ── Tab: Conversation ─────────────────────────────────────
    with tab_chat:
        if not st.session_state.chat_history:
            st.markdown("""
<div class="empty-state">
  <div class="empty-icon">💬</div>
  <div class="empty-text">No conversation yet.<br>Upload a document and ask your first question.</div>
  <div class="empty-hint">Supports Bill of Lading · Rate Confirmation · Carrier RC · Invoices</div>
</div>
""", unsafe_allow_html=True)
        else:
            # Newest first toggle — show oldest first (chronological)
            for i, turn in enumerate(st.session_state.chat_history):
                q         = turn["question"]
                answer    = turn["answer"]
                conf      = turn["confidence"]
                provider  = turn["provider"]
                sources   = turn["sources"]
                logs      = turn["logs"]
                guardrail = turn["guardrail_triggered"]
                ts        = fmt_time(turn.get("timestamp", ""))

                cc = conf_color(conf)

                # ── Question row ──────────────────────────────
                st.markdown(f"""
<div class="qa-card">
  <div class="qa-question-row">
    <div class="qa-q-icon">Q</div>
    <div class="qa-question-text">{q}</div>
    <div class="qa-turn-time">{ts}</div>
  </div>

  <!-- Metrics strip -->
  <div class="metrics-strip">
    <div class="metric-cell">
      <div class="metric-label">Confidence Rate</div>
      <div class="metric-value {cc}">{conf:.0%}</div>
    </div>
    <div class="metric-cell">
      <div class="metric-label">LLM Source</div>
      <div class="metric-value" style="font-size:0.85rem;margin-top:2px">{provider_pill(provider)}</div>
    </div>
    <div class="metric-cell">
      <div class="metric-label">Sources Used</div>
      <div class="metric-value">{len(sources)}</div>
    </div>
  </div>

  <!-- Answer -->
  <div class="qa-answer-label">Answer</div>
""", unsafe_allow_html=True)

                if guardrail:
                    st.markdown(f'<div class="qa-guardrail">⚠&nbsp; {answer}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="qa-answer-text">{answer}</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # ── Sources & Logs expander ───────────────────
                if sources:
                    with st.expander(f"View {len(sources)} source chunk{'s' if len(sources)!=1 else ''}  ·  Q{i+1}", expanded=False):
                        for src in sources:
                            pg = f"Page {src['page']}" if src.get("page") else "—"
                            st.markdown(
                                f'<div class="source-item">'
                                f'<div class="source-meta">Chunk #{src["chunk_index"]} · {pg} · {src["similarity"]:.0%} similarity</div>'
                                f'{src["text"][:500]}{"…" if len(src["text"])>500 else ""}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        if logs:
                            st.markdown('<div style="margin-top:0.6rem"></div>', unsafe_allow_html=True)
                            for log in logs:
                                st.markdown(f'<div class="log-entry">{log}</div>', unsafe_allow_html=True)

    # ── Tab: Extraction ───────────────────────────────────────
    with tab_ext:
        if not has_ext:
            st.markdown("""
<div class="empty-state">
  <div class="empty-icon">📋</div>
  <div class="empty-text">No extraction run yet.<br>Click <strong>Extract Shipment Data</strong> in the left panel.</div>
  <div class="empty-hint">Chat history is preserved while extraction runs.</div>
</div>
""", unsafe_allow_html=True)
        else:
            ext  = st.session_state.extract_result
            data = ext.get("data", {})
            prov = ext.get("provider", "None")
            elogs= ext.get("logs", [])
            nn   = sum(1 for v in data.values() if v is not None)
            tot  = len(data)
            comp = nn / tot if tot else 0

            cc = "conf-high" if comp >= 0.7 else "conf-medium" if comp >= 0.45 else "conf-low"

            st.markdown(f"""
<div class="ext-stats">
  <div class="ext-stat">
    <div class="ext-stat-val {cc}">{comp:.0%}</div>
    <div class="ext-stat-key">Completeness</div>
  </div>
  <div class="ext-stat">
    <div style="margin-top:2px">{provider_pill(prov)}</div>
    <div class="ext-stat-key" style="margin-top:6px">LLM Source</div>
  </div>
  <div class="ext-stat">
    <div class="ext-stat-val">{nn}/{tot}</div>
    <div class="ext-stat-key">Fields Found</div>
  </div>
</div>
""", unsafe_allow_html=True)

            st.progress(comp)

            LABELS = {
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
            rows = ""
            for field, label in LABELS.items():
                val = data.get(field)
                if val is not None:
                    disp = f"${val:,.2f}" if field == "rate" and isinstance(val, (int, float)) else str(val)
                    rows += (f'<div class="ext-field-row">'
                             f'<span class="ext-field-label">{label}</span>'
                             f'<span class="ext-field-val">{disp}</span></div>')
                else:
                    rows += (f'<div class="ext-field-row">'
                             f'<span class="ext-field-label">{label}</span>'
                             f'<span class="ext-field-null">not found</span></div>')

            st.markdown(f'<div class="ext-card">{rows}</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                with st.expander("Raw JSON", expanded=False):
                    st.json(data)
            with col2:
                if elogs:
                    with st.expander("Extraction Logs", expanded=False):
                        for log in elogs:
                            st.markdown(f'<div class="log-entry">{log}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  Ultra Doc-Intelligence &nbsp;·&nbsp; AI-Powered Document Analysis for Transportation Management
  &nbsp;·&nbsp; &copy; 2025 <span>Manvik Siddhpura</span>
</div>
""", unsafe_allow_html=True)
