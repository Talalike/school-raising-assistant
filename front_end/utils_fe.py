# front_end/utils_fe.py
import os
import pathlib
import requests
import streamlit as st
from dotenv import load_dotenv

# Load .env for local dev (repo root)
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)

# ------------------------
# CATEGORY OPTIONS
# ------------------------
CATEGORIES = [
    "Technology",
    "Travels",
    "Art and entertainment",
    "Social Activism",
    "Environment",
    "School libraries",
    "Space and structures",
]


# ------------------------
# BRAND / DESIGN TOKENS (SchoolRaising)
# ------------------------
GLOBAL_CSS = """
<style>
:root{
  --sr-blue:#6F8CFF;        /* brand primary (balloon) */
  --sr-gold:#FFB02A;        /* brand accent (book) */
  --sr-ink:#151515;         /* primary text */
  --sr-slate:#6B7280;       /* secondary text */
  --sr-mist:#F6F8FF;        /* soft bg */
  --sr-line:#E8ECFF;        /* light blue border */
  --sr-success:#16A34A;
  --sr-warning:#F59E0B;
  --sr-error:#DC2626;
}

/* app bg + base */
html, body, [data-testid="stAppViewContainer"]{
  background:#FFFFFF;
  color:var(--sr-ink);
}

/* hero + cards */
.sr-hero{
  padding:4.0rem 1.5rem;
  border-radius:24px;
  background:
    radial-gradient(1200px 600px at 8% 12%, rgba(111,140,255,0.10), transparent),
    linear-gradient(180deg, rgba(255,176,42,0.10), rgba(255,176,42,0.00));
  border:1px solid var(--sr-line);
}
.sr-card{
  background:#FFFFFF;
  border:1px solid var(--sr-line);
  border-radius:16px;
  padding:1rem 1.1rem;
}
.sr-muted{ color:var(--sr-slate); }
.sr-hr{ height:1px; background:var(--sr-line); margin:1.25rem 0; }

/* badge */
.sr-badge{
  display:inline-flex; gap:.5rem; align-items:center;
  padding:.35rem .65rem; border-radius:999px;
  background:rgba(111,140,255,.10);
  color:#27304C; font-weight:600; font-size:.85rem;
  border:1px solid rgba(111,140,255,.25);
}

/* progress (for later steps) */
.sr-progress-wrap{ width:100%; background:rgba(111,140,255,.15); border-radius:999px; height:10px; }
.sr-progress{ height:10px; border-radius:999px; background:var(--sr-blue); width:0%; transition:width .25s ease; }
.sr-steps{ color:var(--sr-slate); font-weight:600; font-size:.9rem; }

/* make Streamlit primary buttons match brand */
button[kind="primary"]{
  background-color: var(--sr-blue) !important;
  color: #fff !important;
  border: 1px solid rgba(0,0,0,0.04) !important;
}
button[kind="primary"]:hover{ filter:brightness(0.96); }

</style>
"""


def inject_global_styles():
    """Inject global CSS once per page."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# Progress / step header helper


def step_header(step: int, total: int, title: str):
    """
    Renders a consistent step header with a progress bar.
    Usage: step_header(step=2, total=5, title="Project Basics")
    """
    # simple guard
    step = max(1, min(step, total))
    pct = int((step / total) * 100)

    # ensure global CSS is injected
    try:
        st.markdown  # sanity check
    except Exception:
        pass
    else:
        st.markdown("""
        <style>
        .sr-progress-wrap{ width:100%; background:rgba(111,140,255,.15); border-radius:999px; height:10px; }
        .sr-progress{ height:10px; border-radius:999px; background:var(--sr-blue); width:0%; transition:width .25s ease; }
        .sr-steps{ color:var(--sr-slate); font-weight:600; font-size:.9rem; margin-top:.35rem; }
        </style>
        """, unsafe_allow_html=True)

    st.markdown(f"##### Step {step} of {total}")
    st.title(title)
    st.markdown(
        f'<div class="sr-progress-wrap"><div class="sr-progress" style="width:{pct}%"></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sr-steps">Progress: {pct}%</div>', unsafe_allow_html=True)


# ------------------------
# API HELPERS
# ------------------------


def api_base() -> str:
    # prefer env var; fall back to st.secrets if set there
    base = os.getenv("ALBA_API_BASE", "").rstrip("/")
    if not base:
        base = str(st.secrets.get("ALBA_API_BASE", "")).rstrip("/")
    if not base:
        raise RuntimeError(
            "Environment variable ALBA_API_BASE is not set. "
            "Add it to your .env for local dev or to Streamlit Cloud secrets."
        )
    return base


def generate_campaign(payload: dict, timeout: int = 180) -> dict:
    url = f"{api_base()}/generate_campaign"
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def ensure_state(st):
    defaults = {
        "school_name": "",
        "project_category": "",
        "user_input_1": "",
        "user_input_2": "",
        "user_input_3": "",
        "draft": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ------------------------
# BRAND ASSET HELPERS
# ------------------------
_ASSETS_DIR = pathlib.Path(__file__).parent / "assets"


def brand_paths() -> dict:
    """Return default asset paths (override if you rename)."""
    return {
        "wordmark": str(_ASSETS_DIR / "sr_wordmark.png"),
        "mark": str(_ASSETS_DIR / "sr_mark.png"),
    }


def image_available(path: str) -> bool:
    try:
        p = pathlib.Path(path)
        return p.exists() and p.is_file()
    except Exception:
        return False


def render_wordmark(width: int = 200):
    """Render the full logo if available, else fallback text."""
    paths = brand_paths()
    if image_available(paths["wordmark"]):
        st.image(paths["wordmark"], width=width)
    else:
        st.markdown("### **SchoolRaising**")
