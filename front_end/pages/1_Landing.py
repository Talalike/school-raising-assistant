# front_end/pages/1_Landing.py
import streamlit as st
from utils_fe import inject_global_styles, brand_paths, image_available

st.set_page_config(
    page_title="Alba — SchoolRaising’s Virtual Assistant",
    page_icon="🎒",
    layout="wide",
)

inject_global_styles()

# --- Gentle top offset so header/logo never looks cropped ---
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# --- Visual system: display font, step cards, gradients, motion-like polish ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&display=swap');

:root {
  --alba-ink: var(--sr-ink);
  --alba-muted: var(--sr-slate);
  --alba-bg: #ffffff;
  --alba-mist: var(--sr-mist);
  --alba-line: var(--sr-line);
  --alba-step1: linear-gradient(180deg, #FFF4EC 0%, #FFFFFF 100%); /* warm peach */
  --alba-step2: linear-gradient(180deg, #EEF5FF 0%, #FFFFFF 100%); /* sky */
  --alba-step3: linear-gradient(180deg, #F6EDFF 0%, #FFFFFF 100%); /* lilac */
}

.block-container { max-width: 980px !important; }
[data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, var(--alba-mist) 0%, #FFFFFF 520px);
}

/* Hero */
.alba-hero {
  background: linear-gradient(180deg, rgba(255,255,255,.75) 0%, #FFFFFF 100%);
  border: 1px solid var(--alba-line);
  box-shadow: 0 8px 24px rgba(18, 24, 40, 0.06);
  border-radius: 18px;
  padding: 24px 22px;
}
.alba-title{
  font-family:'Poppins', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  font-weight:800; letter-spacing:-.015em; font-size:2.35rem; line-height:1.15;
  color:var(--alba-ink); margin:0 0 .5rem 0;
}
.alba-sub{ color:var(--alba-muted); font-size:1.05rem; margin:0; }
.sr-badge { display:inline-flex; align-items:center; gap:8px; font-size:.85rem;
  border:1px solid var(--alba-line); padding:.25rem .5rem; border-radius:999px; background:#fff; }

/* Separators */
.alba-hr{ height:1px; background:var(--alba-line); margin:1.25rem 0; }

/* How it works — bold cards */
.alba-steps{
  display:grid; grid-template-columns: 1fr 1fr 1fr; gap:16px;
}
@media (max-width: 900px){ .alba-steps{ grid-template-columns: 1fr; } }

.alba-step{
  border:1px solid var(--alba-line);
  border-radius:16px;
  padding:18px 16px;
  box-shadow: 0 8px 22px rgba(18, 24, 40, 0.05);
  transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
}
.alba-step:hover{ transform: translateY(-2px); box-shadow: 0 10px 28px rgba(18,24,40,.08); filter: saturate(1.02); }

.alba-step h3{
  font-family:'Poppins', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  font-size:1.15rem; letter-spacing:-.01em; margin:.25rem 0 .35rem 0;
}
.alba-step .muted{ color:var(--alba-muted); font-size:.95rem; margin:0; }
.alba-num{
  display:inline-flex; align-items:center; justify-content:center;
  width:28px; height:28px; border-radius:999px; font-weight:700;
  background:#111827; color:#fff; font-size:.95rem;
  box-shadow: 0 4px 10px rgba(17,24,39,0.18);
}
.alba-icon{
  display:inline-flex; align-items:center; justify-content:center;
  width:34px; height:34px; border-radius:10px; margin-left:8px;
  background:#ffffffaa; border:1px solid var(--alba-line);
}

/* CTA area */
.alba-cta-wrap{ display:flex; justify-content:center; margin-top:10px; }
</style>
""", unsafe_allow_html=True)

# --- BRAND: left-aligned wordmark ONLY ---
paths = brand_paths()
wordmark_path = paths.get("wordmark", "assets/sr_wordmark.png")
if image_available(wordmark_path):
    st.image(wordmark_path, width=200)
else:
    st.markdown("### **SchoolRaising**")
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# --- HERO ---
st.markdown(
    """
<div class="alba-hero">
  <div class="sr-badge">🎒 Alba • SchoolRaising Assistant</div>
  <div style="height:10px;"></div>
  <div class="alba-title">Meet Alba — your school’s storytelling companion.</div>
  <p class="alba-sub">
    Alba helps teachers, parents, and students craft <b>professional crowdfunding campaigns</b> in minutes.
    Answer a few simple questions and Alba turns your idea into a <b>clear, engaging story.</b>
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# --- VALUE PROPS ---
st.markdown("<div class='alba-hr'></div>", unsafe_allow_html=True)
st.subheader("Why schools use Alba")
st.markdown(
    """
<div class="alba-grid">
  <div class="sr-card">🪄 <b>Smart & fast</b><br/>Get a full draft in under two minutes.</div>
  <div class="sr-card">💡 <b>Easy & guided</b><br/>No writing experience needed.</div>
  <div class="sr-card">🎓 <b>Made for schools</b><br/>Tone and structure that fit your community.</div>
  <div class="sr-card">✍️ <b>Fully editable</b><br/>Tweak every section before you publish.</div>
</div>
<div class="sr-muted" style="margin-top:.35rem;">
Everything is <b>fully editable</b> — Alba is here to inspire, not replace, your voice.
</div>
""",
    unsafe_allow_html=True,
)

# --- HOW IT WORKS ---
st.markdown("<div class='alba-hr'></div>", unsafe_allow_html=True)
st.subheader("How it works")

st.markdown("""
<div class="alba-steps">
  <div class="alba-step" style="background: var(--alba-step1);">
    <div><span class="alba-num">1</span><span class="alba-icon">🏫</span></div>
    <h3>Answer</h3>
    <p class="muted">Small prompts about your project basics.</p>
    <p class="muted">School name, category, and your idea in brief.</p>
  </div>

  <div class="alba-step" style="background: var(--alba-step2);">
    <div><span class="alba-num">2</span><span class="alba-icon">✏️</span></div>
    <h3>Draft</h3>
    <p class="muted">Alba turns your inputs into a full campaign.</p>
    <p class="muted">Title, sections, rewards, and impact narrative.</p>
  </div>

  <div class="alba-step" style="background: var(--alba-step3);">
    <div><span class="alba-num">3</span><span class="alba-icon">🚀</span></div>
    <h3>Edit & Publish</h3>
    <p class="muted">You keep control of every word.</p>
    <p class="muted">Refine tone, reorder sections, adjust rewards.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# --- TRUST NOTE + CTA (CTA after steps) ---
st.markdown(
    '<div class="sr-muted" style="margin-top:.6rem;">'
    'Made by <b>SchoolRaising</b> · Your draft stays private until you publish.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
st.markdown("<div class='alba-cta-wrap'>", unsafe_allow_html=True)
cta = st.button("✨ Start your draft with Alba", type="primary")
st.markdown("</div>", unsafe_allow_html=True)
if cta:
    st.switch_page("pages/2_Project_Basics.py")

# --- FOOTER ---
st.markdown("<div class='alba-hr'></div>", unsafe_allow_html=True)
st.caption(
    "SchoolRaising is the first crowdfunding platform dedicated to Italian schools")
st.caption("[Privacy](/#) · [Terms](/#) · [Contact](/#)")
