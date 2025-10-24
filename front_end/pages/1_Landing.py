# front_end/pages/1_Landing.py
import streamlit as st
from utils_fe import inject_global_styles, brand_paths, image_available, render_wordmark

st.set_page_config(
    page_title="Alba — SchoolRaising’s Virtual Assistant",
    page_icon="🎒",
    layout="wide",
)

inject_global_styles()

# ---- Page-level layout polish (max width + background) ----
st.markdown("""
<style>
/* soft page background */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, #F6F8FF 0%, #FFFFFF 420px);
}
/* center the main column and control width */
.block-container {
  max-width: 980px !important;
  padding-top: 1.5rem !important;
}
/* bigger, tighter headline */
.alba-h1 {
  font-size: 2.25rem; /* 36px */
  line-height: 1.2;
  letter-spacing: -0.02em;
  font-weight: 800;
  margin: 0.25rem 0 0.5rem 0;
  color: #151515;
}
/* supporting copy */
.alba-subtle { color: #6B7280; font-size: 1.05rem; }
.alba-section-title { font-weight: 800; font-size: 1.35rem; margin-top: 2.25rem; }
.alba-note { color: #6B7280; margin-top: .75rem; }
.alba-card {
  background:#FFF; border:1px solid #E8ECFF; border-radius:16px; padding:14px 16px;
}
.alba-grid { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
@media (max-width: 820px) {
  .alba-grid { grid-template-columns: 1fr; }
}
/* CTA row (icon + button) */
.alba-cta-row { display:flex; align-items:center; gap:12px; margin-top:16px; }
</style>
""", unsafe_allow_html=True)

# ---------- Wordmark (top-left) ----------
paths = brand_paths()
if image_available(paths["wordmark"]):
    st.image(paths["wordmark"], width=200)
else:
    st.markdown("### **SchoolRaising**")

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ---------- Title & supporting text ----------
st.markdown('<div class="alba-h1">Meet Alba — your school’s storytelling companion.</div>',
            unsafe_allow_html=True)

st.markdown(
    '<p class="alba-subtle">'
    "Alba helps teachers, parents, and students craft <b>professional crowdfunding campaigns</b> in minutes. "
    "Answer a few simple questions and Alba turns your idea into a <b>clear, engaging story</b> — with titles, "
    "sections, and supporter rewards."
    "</p>",
    unsafe_allow_html=True,
)

# ---------- Value props ----------
st.markdown('<div class="alba-section-title">Why schools use Alba</div>',
            unsafe_allow_html=True)
st.markdown(
    """
<div class="alba-grid">
  <div class="alba-card">🪄 <b>Smart & fast</b><br/>Get a full draft in under two minutes.</div>
  <div class="alba-card">💡 <b>Easy & guided</b><br/>No writing experience needed.</div>
  <div class="alba-card">🎯 <b>Made for schools</b><br/>Tone and structure that fit your community.</div>
  <div class="alba-card">✍️ <b>Fully editable</b><br/>Tweak every section before you publish.</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Editability reassurance ----------
st.markdown(
    '<p class="alba-note">Everything is <b>fully editable</b> — Alba is here to inspire, not replace, your voice.</p>',
    unsafe_allow_html=True,
)

# ---------- CTA (icon on the left, primary button) ----------
st.markdown('<div class="alba-cta-row">', unsafe_allow_html=True)
icon_rendered = False
if image_available(paths["mark"]):
    st.image(paths["mark"], width=28)
    icon_rendered = True
else:
    st.markdown("### 🎈")

# Keep the button directly beside the icon
go = st.button("Let Alba Create Your Campaign ✨", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

if go:
    st.switch_page("pages/2_Project_Basics.py")

# ---------- Footer ----------
st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
st.caption("Alba is a SchoolRaising initiative to help schools tell their stories and mobilize their communities.")
