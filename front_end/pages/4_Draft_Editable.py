# front_end/pages/4_Draft_Editable.py
import copy
import streamlit as st
from utils_fe import inject_global_styles, ensure_state, step_header

st.set_page_config(page_title="Alba — Review & Edit Draft",
                   page_icon="🎒", layout="wide")
inject_global_styles()
ensure_state(st)

# Guard: make sure a draft exists (user completed Step 3)
if not st.session_state.get("draft"):
    st.info("No draft available yet. Please generate your campaign first.")
    st.switch_page("pages/3_Project_Details.py")

# ---------- Init editable copy (one-time) ----------
# We keep a separate working copy so user edits don't mutate the raw draft unexpectedly
if "edit_draft" not in st.session_state or not st.session_state["edit_draft"]:
    # normalize rewards to list[dict]
    draft_copy = copy.deepcopy(st.session_state["draft"]) or {}
    rewards = draft_copy.get("rewards", [])
    fixed = []
    for r in rewards:
        if isinstance(r, dict):
            fixed.append({
                "name": r.get("name", ""),
                "description": r.get("description", ""),
                "price": r.get("price", ""),
            })
        else:
            # fallback if backend ever returns strings
            fixed.append({"name": str(r), "description": "", "price": ""})
    draft_copy["rewards"] = fixed
    # ensure keys
    for k in ["title", "alt_title_1", "alt_title_2", "introduction", "in_practice", "description"]:
        draft_copy.setdefault(k, "")
    st.session_state["edit_draft"] = draft_copy

edit = st.session_state["edit_draft"]

# ---------- Step header ----------
step_header(step=4, total=5, title="Review & Edit Your Draft")

# ---------- Styles ----------
st.markdown("""
<style>
.alba-card { background:#FFF; border:1px solid #E8ECFF; border-radius:16px; padding:1.25rem 1.35rem; margin-top:1rem; }
.alba-two { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
@media (max-width: 820px) { .alba-two { grid-template-columns: 1fr; } }
.alba-label { font-weight:600; margin-bottom:.25rem; }
.alba-actions { display:flex; justify-content:space-between; gap:12px; margin-top:1.25rem; }
.alba-row { display:flex; gap:10px; align-items:center; }
.alba-reward { background:#FFF; border:1px solid #E8ECFF; border-radius:14px; padding:12px; margin-top:10px; }
.alba-reward-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }
.alba-note { color:#6B7280; font-size:.92rem; margin-top:.25rem; }
.alba-add { margin-top:.5rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Titles block ----------
st.markdown("### Titles")
with st.container():
    with st.container():
        st.session_state.edit_draft["title"] = st.text_input(
            "Main title", value=edit.get("title", ""), key="ed_title")
    cols = st.columns(2)
    with cols[0]:
        st.session_state.edit_draft["alt_title_1"] = st.text_input(
            "Alt title 1", value=edit.get("alt_title_1", ""), key="ed_alt1")
    with cols[1]:
        st.session_state.edit_draft["alt_title_2"] = st.text_input(
            "Alt title 2", value=edit.get("alt_title_2", ""), key="ed_alt2")

st.markdown('<div class="alba-card">', unsafe_allow_html=True)
st.markdown("### Sections")

# ---------- Body sections (editable text areas) ----------
c1, c2 = st.columns(2)
with c1:
    st.session_state.edit_draft["introduction"] = st.text_area(
        "Introduction", value=edit.get("introduction", ""), height=180, key="ed_intro"
    )
    st.session_state.edit_draft["in_practice"] = st.text_area(
        "In Practice", value=edit.get("in_practice", ""), height=180, key="ed_practice"
    )
with c2:
    st.session_state.edit_draft["description"] = st.text_area(
        "Description", value=edit.get("description", ""), height=370, key="ed_desc"
    )
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Rewards (editable cards, not a table) ----------
st.markdown("### Rewards")
st.markdown('<div class="alba-card">', unsafe_allow_html=True)


def ensure_rewards():
    if not isinstance(st.session_state.edit_draft.get("rewards"), list):
        st.session_state.edit_draft["rewards"] = []


ensure_rewards()

# Add reward button
if st.button("➕ Add reward", help="Add a new reward level", use_container_width=False):
    st.session_state.edit_draft["rewards"].append(
        {"name": "", "description": "", "price": ""})
    st.rerun()

# Render reward blocks
to_delete_index = None
for i, r in enumerate(st.session_state.edit_draft["rewards"]):
    st.markdown('<div class="alba-reward">', unsafe_allow_html=True)
    head_cols = st.columns([6, 1])
    with head_cols[0]:
        st.markdown(f"**Reward {i+1}**", unsafe_allow_html=True)
    with head_cols[1]:
        if st.button("🗑️", key=f"del_{i}", help="Remove this reward"):
            to_delete_index = i

    # Editable fields
    st.session_state.edit_draft["rewards"][i]["name"] = st.text_input(
        "Name", value=r.get("name", ""), key=f"r_name_{i}"
    )
    st.session_state.edit_draft["rewards"][i]["description"] = st.text_area(
        "Description", value=r.get("description", ""), height=100, key=f"r_desc_{i}"
    )
    # price as text or number — keeping text to avoid locale issues; you can validate later
    st.session_state.edit_draft["rewards"][i]["price"] = st.text_input(
        "Price (€)", value=str(r.get("price", "")), key=f"r_price_{i}"
    )

    st.markdown('</div>', unsafe_allow_html=True)

# Apply deletion after loop to avoid layout glitches
if to_delete_index is not None:
    del st.session_state.edit_draft["rewards"][to_delete_index]
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Actions ----------
st.markdown('<div class="alba-actions">', unsafe_allow_html=True)
col_back, col_next = st.columns(2)

with col_back:
    if st.button("← Back", type="secondary"):
        st.switch_page("pages/3_Project_Details.py")

with col_next:
    if st.button("Save & Continue →", type="primary"):
        # Lightweight normalization before saving
        ed = st.session_state.edit_draft

        # Coerce rewards to trimmed dicts
        clean_rewards = []
        for r in ed.get("rewards", []):
            if not isinstance(r, dict):
                continue
            name = (r.get("name", "") or "").strip()
            desc = (r.get("description", "") or "").strip()
            price = (r.get("price", "") or "").strip()
            # keep even if blank; user might want to finish later
            clean_rewards.append(
                {"name": name, "description": desc, "price": price})

        # Save back to the main draft
        st.session_state.draft = {
            "title": (ed.get("title", "") or "").strip(),
            "alt_title_1": (ed.get("alt_title_1", "") or "").strip(),
            "alt_title_2": (ed.get("alt_title_2", "") or "").strip(),
            "introduction": ed.get("introduction", "") or "",
            "in_practice": ed.get("in_practice", "") or "",
            "description": ed.get("description", "") or "",
            "rewards": clean_rewards,
        }

        st.switch_page("pages/5_Success.py")

st.markdown('</div>', unsafe_allow_html=True)
