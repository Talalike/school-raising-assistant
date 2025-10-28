# front_end/pages/2_Project_Basics.py
import streamlit as st
from utils_fe import inject_global_styles, ensure_state, CATEGORIES, step_header

# ---------- Page setup ----------
st.set_page_config(page_title="Alba — Project Basics",
                   page_icon="🎒", layout="wide")
inject_global_styles()
ensure_state(st)

# ---------- Progress header ----------
step_header(step=2, total=5, title="Project Basics")

# ---------- Form card ----------
st.markdown("""
<style>
.alba-card {
  background:#FFF; border:1px solid #E8ECFF; border-radius:16px; 
  padding:2rem 2.25rem; margin-top:1.5rem;
}
.alba-label { font-weight:600; margin-bottom:6px; }
.alba-note { color:#6B7280; font-size:0.95rem; margin-top:4px; }
.alba-buttons { display:flex; justify-content:space-between; margin-top:1.75rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("### Tell us a few basics about your project")

# --- Inputs ---
st.session_state.school_name = st.text_input(
    "School name", value=st.session_state.school_name
)

# --- Project category (dropdown with placeholder, no default) ---
chosen = st.selectbox(
    "Project category",
    CATEGORIES,
    index=None,                              # ← no option pre-selected
    placeholder="Choose a project category",  # grey prompt text
)

# Persist the user’s choice (only if they picked something)
if chosen:
    st.session_state.project_category = chosen

# st.session_state.project_category = st.selectbox(
#    "Project category", CATEGORIES,
#    index=(
#        CATEGORIES.index(st.session_state.project_category)
#        if st.session_state.project_category in CATEGORIES else 0
#    )
# )

st.markdown(
    '<p class="alba-note">You can always edit these later before submitting your final campaign.</p>',
    unsafe_allow_html=True,
)

# --- Buttons row ---
st.markdown('<div class="alba-buttons">', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    if st.button("← Back", type="secondary"):
        st.switch_page("pages/1_Landing.py")
with col2:
    if st.button("Next →", type="primary"):
        if not st.session_state.school_name.strip():
            st.warning("Please enter your school name.")
        else:
            st.switch_page("pages/3_Project_Details.py")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
