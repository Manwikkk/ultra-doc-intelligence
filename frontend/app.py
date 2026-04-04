"""
app.py — Streamlit frontend for Ultra Doc-Intelligence.
Professional dark UI with full-width layout, structured results, and footer.
"""
from __future__ import annotations

import os
import base64
from pathlib import Path
import json
import requests
import streamlit as st
import streamlit.components.v1 as components

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
        padding-top: 0 !important;
        padding-bottom: 2rem !important;
        max-width: 1100px !important;
        margin: 0 auto !important;
    }
    section[data-testid="stSidebar"] { display: none !important; }

    /* ── Header ── */
    .site-header {
        background: rgba(10, 15, 26, 0.95);
        border-bottom: 1px solid rgba(255,255,255,0.06);
        padding: 0;
        margin-bottom: 2.5rem;
        position: sticky;
        top: 0;
        z-index: 100;
        backdrop-filter: blur(12px);
    }
    .header-inner {
        max-width: 1100px;
        margin: 0 auto;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 18px 0;
    }
    .header-brand { display: flex; flex-direction: column; }
    .header-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.4px;
        line-height: 1.2;
    }
    .header-subtitle {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.32);
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
        padding: 7px 16px;
        font-size: 0.78rem;
        color: rgba(255,255,255,0.5);
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
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.28);
        margin-bottom: 0.65rem;
        margin-top: 0.15rem;
    }

    /* ── Cards ── */
    .card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .card-highlight {
        background: linear-gradient(135deg, rgba(99,102,241,0.06), rgba(139,92,246,0.04));
        border: 1px solid rgba(99,102,241,0.18);
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .card-warning {
        background: rgba(245,158,11,0.06);
        border: 1px solid rgba(245,158,11,0.2);
        border-left: 3px solid #f59e0b;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }

    /* ── Answer ── */
    .answer-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: rgba(99,102,241,0.7);
        margin-bottom: 0.9rem;
    }
    .answer-text {
        font-size: 0.975rem;
        line-height: 1.9;
        color: rgba(255,255,255,0.88);
    }

    /* ── Confidence ── */
    .conf-label {
        font-size: 0.68rem;
        color: rgba(255,255,255,0.35);
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .conf-value-high   { font-size: 1.6rem; font-weight: 700; color: #22c55e; letter-spacing: -0.5px; }
    .conf-value-medium { font-size: 1.6rem; font-weight: 700; color: #f59e0b; letter-spacing: -0.5px; }
    .conf-value-low    { font-size: 1.6rem; font-weight: 700; color: #ef4444; letter-spacing: -0.5px; }

    /* ── Provider badge ── */
    .provider-pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 14px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.2px;
    }
    .pill-groq   { background: rgba(0,212,170,0.1);  color: #00d4aa; border: 1px solid rgba(0,212,170,0.25); }
    .pill-openai { background: rgba(16,163,127,0.1); color: #10a37f; border: 1px solid rgba(16,163,127,0.25); }
    .pill-gemini { background: rgba(66,133,244,0.1); color: #4285f4; border: 1px solid rgba(66,133,244,0.25); }
    .pill-ollama { background: rgba(255,107,53,0.1); color: #ff6b35; border: 1px solid rgba(255,107,53,0.25); }
    .pill-none   { background: rgba(239,68,68,0.1);  color: #ef4444; border: 1px solid rgba(239,68,68,0.25); }

    /* ── Guardrail ── */
    .guardrail-alert {
        background: rgba(239,68,68,0.07);
        border: 1px solid rgba(239,68,68,0.22);
        border-left: 3px solid #ef4444;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        color: #fca5a5;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 1rem;
        line-height: 1.6;
    }

    /* ── Source chunk ── */
    .source-chunk {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin: 0.5rem 0;
        font-size: 0.84rem;
        color: rgba(255,255,255,0.62);
        line-height: 1.7;
    }
    .chunk-meta {
        font-size: 0.66rem;
        color: rgba(255,255,255,0.28);
        margin-bottom: 0.45rem;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Field rows ── */
    .field-row {
        display: flex; justify-content: space-between;
        align-items: flex-start;
        padding: 10px 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        gap: 1.5rem;
    }
    .field-row:last-child { border-bottom: none; }
    .field-name  { color: rgba(255,255,255,0.36); font-size: 0.82rem; font-weight: 500; flex-shrink: 0; min-width: 175px; }
    .field-value { color: #e2e8f0; font-size: 0.87rem; font-weight: 500; text-align: right; word-break: break-word; }
    .field-null  { color: rgba(255,255,255,0.18); font-size: 0.82rem; font-style: italic; text-align: right; }

    /* ── Log ── */
    .log-line {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: rgba(255,255,255,0.38);
        padding: 2px 0;
        line-height: 1.65;
    }

    /* ── Doc chip ── */
    .doc-chip {
        display: flex; align-items: flex-start; flex-direction: column; gap: 3px;
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.22);
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 0.84rem;
        color: #a5b4fc;
        font-weight: 500;
        margin-top: 0.5rem;
        width: 100%;
    }
    .doc-chip-id {
        font-size: 0.68rem;
        font-family: 'JetBrains Mono', monospace;
        opacity: 0.55;
        color: rgba(165,180,252,0.7);
    }

    /* ── Placeholder ── */
    .placeholder-card {
        background: rgba(255,255,255,0.02);
        border: 1px dashed rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 4rem 2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .placeholder-icon { font-size: 2.2rem; margin-bottom: 0.85rem; opacity: 0.35; }
    .placeholder-text { color: rgba(255,255,255,0.22); font-size: 0.9rem; line-height: 1.7; }

    /* ── Stats row ── */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .stat-box {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .stat-val { font-size: 1.5rem; font-weight: 700; color: #fff; letter-spacing: -0.5px; }
    .stat-key { font-size: 0.68rem; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 1px; margin-top: 3px; font-weight: 600; }

    /* ── Divider ── */
    .section-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin: 2rem 0;
    }

    /* ── Streamlit form — strip default border box ── */
    [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    /* Hide "Press Enter to submit form" helper text */
    [data-testid="InputInstructions"] {
        display: none !important;
    }

    /* ── Streamlit widget overrides ── */
    .stTextArea textarea {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.09) !important;
        border-radius: 10px !important;
        color: rgba(255,255,255,0.85) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.92rem !important;
        padding: 0.85rem 1rem !important;
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
        font-size: 0.9rem !important;
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
        color: rgba(255,255,255,0.65) !important;
        font-size: 0.88rem !important;
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
        border: 1.5px dashed rgba(99,102,241,0.2) !important;
        border-radius: 12px !important;
    }
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.025) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 8px !important;
        font-size: 0.83rem !important;
        color: rgba(255,255,255,0.48) !important;
    }
    hr { border-color: rgba(255,255,255,0.05) !important; }
    .stSuccess { background: rgba(34,197,94,0.07) !important; border: 1px solid rgba(34,197,94,0.18) !important; border-radius: 10px !important; color: #86efac !important; }
    .stError   { background: rgba(239,68,68,0.07) !important; border: 1px solid rgba(239,68,68,0.18) !important; border-radius: 10px !important; color: #fca5a5 !important; }
    .stInfo    { background: rgba(99,102,241,0.07) !important; border: 1px solid rgba(99,102,241,0.18) !important; border-radius: 10px !important; color: #a5b4fc !important; }

    /* ── Footer ── */
    .site-footer {
        margin-top: 4rem;
        padding: 2rem 0 1.5rem 0;
        border-top: 1px solid rgba(255,255,255,0.05);
        text-align: center;
    }
    .footer-text {
        font-size: 0.78rem;
        color: rgba(255,255,255,0.2);
        letter-spacing: 0.3px;
        line-height: 1.8;
    }
    .footer-name {
        color: rgba(255,255,255,0.35);
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "doc_id":         None,
    "filename":       None,
    "ask_result":     None,
    "extract_result": None,
    "backend_url":    os.getenv("BACKEND_URL", "http://localhost:8000"),
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


def conf_class(score: float) -> str:
    if score >= 0.70: return "conf-value-high"
    if score >= 0.45: return "conf-value-medium"
    return "conf-value-low"


def provider_badge(provider: str) -> str:
    p = provider.lower()
    css  = {"groq": "pill-groq", "openai": "pill-openai", "gemini": "pill-gemini", "ollama": "pill-ollama"}.get(p, "pill-none")
    icon = {"groq": "⚡", "openai": "◆", "gemini": "✦", "ollama": "🦙"}.get(p, "•")
    return f'<span class="provider-pill {css}">{icon} {provider}</span>'


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
  <div style="font-size:0.9rem;color:#fde68a;font-weight:600;margin-bottom:6px">
    Backend Offline
  </div>
  <div style="font-size:0.82rem;color:rgba(255,255,255,0.5);line-height:1.8">
    <b>Detail:</b> {alive_detail}<br><br>
    <b>Start the backend:</b><br>
    <code style="background:rgba(0,0,0,0.35);padding:2px 10px;border-radius:4px;color:#fde68a;font-family:'JetBrains Mono',monospace;font-size:0.8rem">cd backend &amp;&amp; uvicorn main:app --host 127.0.0.1 --port 8000 --reload</code>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Two-column layout ─────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.55], gap="large")


# ════════════════════════════════════════════════════════════
#  LEFT PANEL — Document upload + Query + Extract
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
        with st.spinner(f"Ingesting **{uploaded_file.name}**…"):
            try:
                resp = requests.post(
                    f"{get_url()}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.doc_id         = data["doc_id"]
                    st.session_state.filename       = uploaded_file.name
                    st.session_state.ask_result     = None
                    st.session_state.extract_result = None
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

    # ── Ask (Enter to submit) ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">Ask a Question</div>', unsafe_allow_html=True)

    no_doc = st.session_state.doc_id is None or not alive

    with st.form(key="ask_form", clear_on_submit=False):
        question = st.text_area(
            "Question",
            placeholder="e.g. Who is the consignee?\nWhat is the freight rate?",
            disabled=no_doc,
            label_visibility="collapsed",
            height=110,
        )
        ask_btn = st.form_submit_button(
            "Search Document",
            disabled=no_doc,
            use_container_width=True,
            type="primary",
        )

    # Allow Enter to submit the form (Shift+Enter for newline)
    components.html(
        """
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
        """,
        height=0,
        width=0,
    )

    if ask_btn and question.strip():
        with st.spinner("Retrieving answer…"):
            try:
                resp = requests.post(
                    f"{get_url()}/ask",
                    json={"doc_id": st.session_state.doc_id, "question": question},
                    timeout=90,
                )
                if resp.status_code == 200:
                    st.session_state.ask_result     = resp.json()
                    st.session_state.extract_result = None
                else:
                    st.error(f"Error: {resp.json().get('detail', resp.text)}")
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach backend at {get_url()}.")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Extract ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Structured Extraction</div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:0.82rem;color:rgba(255,255,255,0.3);margin-bottom:0.75rem;line-height:1.6">'
        'Automatically extract key shipment fields using hybrid regex + LLM analysis.'
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
                    st.session_state.ask_result     = None
                else:
                    st.error(f"Error: {resp.json().get('detail', resp.text)}")
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach backend at {get_url()}.")
            except Exception as e:
                st.error(f"Error: {e}")

    # ── How it works ─────────────────────────────────────────────────────────
    if not st.session_state.doc_id:
        st.markdown("""
<div class="card" style="margin-top:1.5rem">
  <div style="font-size:0.8rem;color:rgba(255,255,255,0.45);font-weight:600;margin-bottom:0.65rem;letter-spacing:0.3px">How it works</div>
  <div style="font-size:0.8rem;color:rgba(255,255,255,0.28);line-height:2;letter-spacing:0.1px">
    1. Upload a PDF, DOCX, or TXT logistics document<br>
    2. Ask questions in plain English via the query panel<br>
    3. Use structured extraction to pull key shipment fields<br><br>
    <span style="color:rgba(255,255,255,0.2)">Supports: Bill of Lading · Rate Confirmation · Carrier RC · Invoices</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  RIGHT PANEL — Results
# ════════════════════════════════════════════════════════════
with right_col:

    # ── Ask Result ────────────────────────────────────────────────────────────
    if st.session_state.ask_result:
        result = st.session_state.ask_result

        score    = result.get("confidence", 0.0)
        provider = result.get("provider", "None")
        sources  = result.get("sources", [])

        st.markdown(f"""
<div class="stats-grid">
  <div class="stat-box">
    <div class="stat-val" style="color:{'#22c55e' if score>=0.7 else '#f59e0b' if score>=0.45 else '#ef4444'}">{score:.0%}</div>
    <div class="stat-key">Confidence</div>
  </div>
  <div class="stat-box">
    <div style="margin-top:4px">{provider_badge(provider)}</div>
    <div class="stat-key" style="margin-top:6px">LLM Provider</div>
  </div>
  <div class="stat-box">
    <div class="stat-val">{len(sources)}</div>
    <div class="stat-key">Chunks Retrieved</div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)

        if result.get("guardrail_triggered"):
            st.markdown(
                f'<div class="guardrail-alert">Guardrail triggered — {result["answer"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="card-highlight">'
                f'<div class="answer-label">Response</div>'
                f'<div class="answer-text">{result["answer"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.progress(score)

        if sources:
            with st.expander(f"Source Chunks  ·  {len(sources)} retrieved", expanded=False):
                for src in sources:
                    page_info = f"Page {src['page']}" if src.get("page") else "—"
                    st.markdown(
                        f'<div class="source-chunk">'
                        f'<div class="chunk-meta">Chunk #{src["chunk_index"]} · {page_info} · Similarity {src["similarity"]:.0%}</div>'
                        f'{src["text"][:520]}{"…" if len(src["text"]) > 520 else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        logs = result.get("logs", [])
        if logs:
            with st.expander("System Logs", expanded=False):
                for log in logs:
                    st.markdown(f'<div class="log-line">{log}</div>', unsafe_allow_html=True)


    # ── Extract Result ────────────────────────────────────────────────────────
    if st.session_state.extract_result:
        extract = st.session_state.extract_result
        data    = extract.get("data", {})

        non_null     = sum(1 for v in data.values() if v is not None)
        total        = len(data)
        completeness = non_null / total if total else 0
        provider     = extract.get("provider", "None")

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
        st.markdown('<div class="section-label">Extracted Fields</div>', unsafe_allow_html=True)

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

        ext_logs = extract.get("logs", [])
        if ext_logs:
            with st.expander("Extraction Logs", expanded=False):
                for log in ext_logs:
                    st.markdown(f'<div class="log-line">{log}</div>', unsafe_allow_html=True)


    # ── Placeholder ───────────────────────────────────────────────────────────
    if not st.session_state.ask_result and not st.session_state.extract_result:
        st.markdown("""
<div class="placeholder-card">
  <div class="placeholder-icon">&#x2B21;</div>
  <div class="placeholder-text">
    Upload a document and ask a question,<br>or run structured extraction to see results here.<br><br>
    <span style="font-size:0.78rem;opacity:0.55">Bill of Lading · Rate Confirmation · Carrier RC</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-footer">
  <div class="footer-text">
    Ultra Doc-Intelligence &nbsp;&middot;&nbsp; AI-Powered Document Analysis for Transportation Management<br>
    &copy; 2025 <span class="footer-name">Manvik Siddhpura</span>. All Rights Reserved.
  </div>
</div>
""", unsafe_allow_html=True)
