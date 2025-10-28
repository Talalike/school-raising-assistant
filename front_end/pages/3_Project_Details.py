# front_end/pages/3_Project_Details.py
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

# Guard: if user skipped Step 2, send them back
if not st.session_state.get("school_name", "").strip():
    st.switch_page("pages/2_Project_Basics.py")

# ---------- Progress header ----------
step_header(step=3, total=5, title="Project Details")

# ---------- Page-local styles (scoped) ----------
st.markdown("""
<style>
/* A calmer, airier rhythm than default .alba-card stacks */
.alba-card.details-stack { padding: 1.2rem 1.3rem; }
.details-mini { 
  background: var(--alba-surface, #ffffff);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 16px;
  padding: 1rem 1rem 0.25rem 1rem;
  margin-bottom: 12px;
  transition: box-shadow .15s ease, border-color .15s ease;
}
.details-mini.invalid { 
  border-color: rgba(84,110,255,0.45);
  box-shadow: 0 0 0 3px rgba(84,110,255,0.12);
}
.details-label {
  font-weight: 600; 
  font-size: 0.95rem; 
  margin-bottom: 6px;
}
.details-divider {
  height: 1px; 
  background: rgba(0,0,0,0.05); 
  margin: 8px 0 2px 0;
}

/* Softer textarea visuals without touching global theme */
.details-mini .stTextArea textarea {
  border-radius: 12px !important;
  line-height: 1.45 !important;
  min-height: 130px !important;
}
.details-mini .stTextArea textarea:focus {
  box-shadow: 0 0 0 3px rgba(84,110,255,0.10) !important;
  border-color: rgba(84,110,255,0.35) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="alba-card details-stack">', unsafe_allow_html=True)

# ---------- Validation helpers ----------


def _is_blank(val: str) -> bool:
    return not str(val or "").strip()


invalid_map = {"f1": False, "f2": False, "f3": False}
scroll_target = None  # will become the first invalid field id on submit

# ---------- Fields (placeholders only, no counters) ----------
# Field 1

st.markdown('<div class="details-label">About your project</div>',
            unsafe_allow_html=True)
about_txt = st.text_area(
    label="About your project",
    label_visibility="collapsed",
    key="user_input_1",
    placeholder="Briefly describe what you’ll do, with who, and where. Mention activities, timeline, and the intended outcome.",
)
st.markdown('<div class="details-divider"></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Field 2

st.markdown('<div class="details-label">Why this matters</div>',
            unsafe_allow_html=True)
why_txt = st.text_area(
    label="Why this matters",
    label_visibility="collapsed",
    key="user_input_2",
    placeholder="Explain the need or problem, who benefits (students, teachers, community), and what changes once this project succeeds.",
)
st.markdown('<div class="details-divider"></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Field 3

st.markdown('<div class="details-label">What supporters get</div>',
            unsafe_allow_html=True)
perks_txt = st.text_area(
    label="What supporters get",
    label_visibility="collapsed",
    key="user_input_3",
    placeholder="List simple, meaningful ways to thank supporters (e.g., thank-you note, invite to a class demo, student-made postcard).",
)
st.markdown('<div class="details-divider"></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Buttons row ----------
st.markdown('<div class="alba-buttons">', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    if st.button("← Back", type="secondary"):
        st.switch_page("pages/2_Project_Basics.py")
with c2:
    if st.button("Generate draft ✨", type="primary"):
        # Validate minimal non-empty inputs
        invalid_map["f1"] = _is_blank(about_txt)
        invalid_map["f2"] = _is_blank(why_txt)
        invalid_map["f3"] = _is_blank(perks_txt)

        # Decide which field to scroll to first (if any)
        if invalid_map["f1"]:
            scroll_target = "field1"
        elif invalid_map["f2"]:
            scroll_target = "field2"
        elif invalid_map["f3"]:
            scroll_target = "field3"

        if any(invalid_map.values()):
            st.warning(
                "Please complete all three sections before generating your draft.")
            # Apply invalid class via a tiny script (best-effort; harmless if blocked)
            st.markdown(f"""
                <script>
                  const invalids = { [k for k,v in invalid_map.items() if v] };
                  invalids.forEach(k => {{
                    const el = document.querySelector(`[data-field="{{k}}"]`);
                    if (el) el.classList.add("invalid");
                  }});
                  const tgt = document.getElementById("{scroll_target or ''}");
                  if (tgt && tgt.scrollIntoView) {{
                      tgt.scrollIntoView({{ behavior: "smooth", block: "start" }});
                  }}
                </script>
            """, unsafe_allow_html=True)
        else:
            # Build payload strictly from session (names from ensure_state)
            payload = {
                "school_name": st.session_state.get("school_name", "").strip(),
                "project_category": st.session_state.get("project_category", "").strip(),
                "user_input_1": st.session_state.get("user_input_1", "").strip(),
                "user_input_2": st.session_state.get("user_input_2", "").strip(),
                "user_input_3": st.session_state.get("user_input_3", "").strip(),
            }

            # Spinner: retain time mention per your decision
            with st.spinner("Generating your campaign draft — this may take under 2 minutes…"):
                try:
                    resp = generate_campaign(payload)
                    # Persist and proceed
                    st.session_state.draft = resp
                    st.switch_page("pages/4_Draft_Editable.py")
                except Exception as e:
                    msg = str(e)
                    if len(msg) > 900:
                        msg = msg[:900] + "…"
                    st.error("Generation failed. " + msg)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
