# front_end/pages/3_Project_Details.py
import textwrap
import streamlit as st
from utils_fe import (
    inject_global_styles,
    ensure_state,
    step_header,
    generate_campaign,
)

# ---------- Page setup ----------
st.set_page_config(page_title="Alba — Project Details",
                   page_icon="🎒", layout="wide")
inject_global_styles()
ensure_state(st)

# Guard: if user skipped Step 2
if not st.session_state.school_name:
    st.info("Please fill in your project basics first.")
    st.switch_page("pages/2_Project_Basics.py")

# ---------- Progress header ----------
step_header(step=3, total=5, title="Project Details")

# ---------- Styles ----------
st.markdown(
    """
<style>
.alba-card { background:#FFF; border:1px solid #E8ECFF; border-radius:16px; padding:2rem 2.25rem; margin-top:1.5rem; }
.alba-help { color:#6B7280; font-size:.95rem; margin-top:.25rem; }
.alba-hint { color:#6B7280; font-size:.9rem; margin:.25rem 0 .75rem; }
.alba-buttons { display:flex; justify-content:space-between; margin-top:1.5rem; }
.alba-counter { color:#6B7280; font-size:.85rem; text-align:right; margin-top:.25rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Form card ----------
st.markdown('<div class="alba-card">', unsafe_allow_html=True)
st.markdown("### Tell us more about your idea")

# Soft char limits (guidance only)
SOFT_MIN, SOFT_MAX = 60, 600


def counter(text: str) -> str:
    n = len(text.strip())
    tip = ""
    if n and n < SOFT_MIN:
        tip = " · a bit short — consider adding details"
    elif n > SOFT_MAX:
        tip = " · quite long — consider tightening"
    return f"{n} characters{tip}"


# --- Q1 ---
st.session_state.user_input_1 = st.text_area(
    "What is your project about?",
    value=st.session_state.user_input_1,
    help=("Briefly describe the idea: what do you want to do, who is involved, and what makes it special? "
          "E.g. “We want to create a theater performance on bullying, involving 3 classes and a local artist.”"),
    height=130,
)
st.markdown(
    f'<div class="alba-counter">{counter(st.session_state.user_input_1)}</div>', unsafe_allow_html=True)

# --- Q2 ---
st.session_state.user_input_2 = st.text_area(
    "Why is this project important for your school or community?",
    value=st.session_state.user_input_2,
    help=("Tell us what motivated you. What need or dream are you responding to? "
          "E.g. “Our students often feel excluded and need new ways to express themselves.”"),
    height=130,
)
st.markdown(
    f'<div class="alba-counter">{counter(st.session_state.user_input_2)}</div>', unsafe_allow_html=True)

# --- Q3 ---
st.session_state.user_input_3 = st.text_area(
    "Would you like to offer something to the people who support your project?",
    value=st.session_state.user_input_3,
    help=("You can mention a small gift, a public thank you, or an invitation to join an activity. "
          "E.g. “A thank-you video made by students, or the chance to attend the final event.”"),
    height=110,
)
st.markdown(
    f'<div class="alba-counter">{counter(st.session_state.user_input_3)}</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="alba-hint">Tip: write naturally — Alba will polish the tone and structure for you.</div>',
    unsafe_allow_html=True,
)

# ---------- Buttons ----------
st.markdown('<div class="alba-buttons">', unsafe_allow_html=True)
col_back, col_next = st.columns(2)

with col_back:
    if st.button("← Back", type="secondary"):
        st.switch_page("pages/2_Project_Basics.py")

with col_next:
    if st.button("Generate draft ✨", type="primary"):
        # Validate locally
        fields = {
            "What is your project about?": st.session_state.user_input_1,
            "Why is this project important?": st.session_state.user_input_2,
            "Supporter offer (rewards idea)": st.session_state.user_input_3,
        }
        missing = [label for label,
                   val in fields.items() if not val or not val.strip()]
        if missing:
            st.warning("Please complete: " + ", ".join(missing))
            st.stop()

        payload = {
            "school_name": st.session_state.school_name.strip(),
            "project_category": st.session_state.project_category,
            "user_input_1": st.session_state.user_input_1.strip(),
            "user_input_2": st.session_state.user_input_2.strip(),
            "user_input_3": st.session_state.user_input_3.strip(),
        }

        with st.spinner("Generating your campaign… this usually takes under 2 minutes."):
            try:
                st.session_state.draft = generate_campaign(
                    payload, timeout=180)
            except Exception as e:
                # Make a friendly error message
                msg = str(e)
                # Trim very long HTTP bodies if present
                if len(msg) > 900:
                    msg = msg[:900] + "…"
                st.error("Generation failed. " + msg)
                st.stop()

        # Proceed to the editable draft page
        st.switch_page("pages/4_Draft_Editable.py")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
