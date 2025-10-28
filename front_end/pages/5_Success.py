# front_end/pages/5_Success.py
import json
import streamlit as st
from utils_fe import inject_global_styles, ensure_state, step_header

st.set_page_config(page_title="Alba — Success", page_icon="🎒", layout="wide")
inject_global_styles()
ensure_state(st)

# Guard
draft = st.session_state.get("draft")
if not draft:
    st.info("No finished draft found. Start from the beginning.")
    st.switch_page("pages/1_Landing.py")

# Celebrate only once per arrival on this page
if "success_balloons" not in st.session_state:
    st.session_state.success_balloons = True
    st.balloons()

# ---------- Progress / Title ----------
step_header(step=5, total=5, title="Your campaign draft is ready 🎉")

# ---------- Styles ----------
st.markdown("""
<style>
.alba-hero-card{
  background:#FFFFFF;
  border:1px solid #E8ECFF;
  border-radius:20px;
  padding:1.25rem 1.5rem;
  margin-top:1rem;
}
.alba-title{ font-size:1.5rem; font-weight:800; margin:0.25rem 0 0.25rem 0; color:#151515; }

.alba-two{ display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-top:.75rem;}
@media (max-width: 900px){ .alba-two{ grid-template-columns: 1fr; } }

.alba-card{ background:#FFF; border:1px solid #E8ECFF; border-radius:16px; padding:1rem 1.1rem; }
.alba-card h4{ margin:.25rem 0 .5rem 0; }
.alba-reward{
  background:#FFF; border:1px solid #E8ECFF; border-radius:14px; padding:12px; margin-top:10px;
}
.alba-actions{ display:flex; justify-content:center; margin:1.25rem 0 0.25rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Hero summary ----------
# st.markdown('<div class="alba-hero-card">', unsafe_allow_html=True)
st.markdown(
    f'<div class="alba-title">{draft.get("title","Untitled")}</div>', unsafe_allow_html=True)

cols = st.columns(2)
with cols[0]:
    st.caption("Alt title 1")
    st.write(draft.get("alt_title_1", "—"))
with cols[1]:
    st.caption("Alt title 2")
    st.write(draft.get("alt_title_2", "—"))
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Sections ----------
st.markdown('<div class="alba-two">', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="alba-card">', unsafe_allow_html=True)
    st.markdown("#### In Practice")        # <-- moved up
    st.write(draft.get("in_practice", ""))
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.markdown("#### Introduction")       # <-- moved below
    st.write(draft.get("introduction", ""))
    st.markdown('</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="alba-card">', unsafe_allow_html=True)
    st.markdown("#### Description")
    st.write(draft.get("description", ""))
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Rewards (read-only cards) ----------
st.markdown('<div class="alba-card" style="margin-top:16px;">',
            unsafe_allow_html=True)
st.markdown("#### Rewards")
rewards = draft.get("rewards", [])
if isinstance(rewards, list) and rewards:
    for i, r in enumerate(rewards, start=1):
        name = (r.get("name", "") if isinstance(
            r, dict) else str(r)) or "Reward"
        desc = (r.get("description", "") if isinstance(r, dict) else "")
        price = (r.get("price", "") if isinstance(r, dict) else "")
        st.markdown(
            f'<div class="alba-reward"><b>{i}. {name}</b><br/><span class="alba-subtle">{desc}</span><br/><span class="alba-kv">Price (€): <b>{price}</b></span></div>',
            unsafe_allow_html=True,
        )
else:
    st.caption("No rewards included.")

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Single CTA ----------
st.markdown('<div class="alba-actions">', unsafe_allow_html=True)
if st.button("Generate a new campaign", type="primary"):
    # reset state for a fresh start (keep draft None)
    for k in ("school_name", "project_category", "user_input_1", "user_input_2", "user_input_3"):
        st.session_state[k] = ""
    st.session_state["draft"] = None
    st.session_state["edit_draft"] = None
    st.session_state["success_balloons"] = False
    st.switch_page("pages/2_Project_Basics.py")
st.markdown('</div>', unsafe_allow_html=True)

# Optional: tiny footer line
st.caption(
    "Alba is a SchoolRaising initiative to help schools tell their stories and mobilize their communities."
)
