# front_end/pages/4_Draft_Editable.py
import copy
import streamlit as st
from utils_fe import inject_global_styles, ensure_state, step_header

# ------------------------------------------------------------------
#  Page config, global styles, state guards
# ------------------------------------------------------------------
st.set_page_config(page_title="Alba — Review & Edit Draft",
                   page_icon="🎒", layout="wide")
inject_global_styles()
ensure_state(st)

# Make sure a draft exists (user finished Step 3)
if not st.session_state.get("draft"):
    st.info("No draft available yet. Please generate your campaign first.")
    st.switch_page("pages/3_Project_Details.py")

# ------------------------------------------------------------------
#  One-time editable working copy
# ------------------------------------------------------------------
if "edit_draft" not in st.session_state or not st.session_state["edit_draft"]:
    draft_copy = copy.deepcopy(st.session_state["draft"]) or {}

    # Normalise rewards to list[dict]
    fixed = []
    for r in draft_copy.get("rewards", []):
        if isinstance(r, dict):
            fixed.append({
                "name": r.get("name", ""),
                "description": r.get("description", ""),
                "price": r.get("price", ""),
            })
        else:
            fixed.append({"name": str(r), "description": "", "price": ""})
    draft_copy["rewards"] = fixed

    # Ensure text keys exist
    for k in ["title", "alt_title_1", "alt_title_2",
              "introduction", "in_practice", "description"]:
        draft_copy.setdefault(k, "")

    st.session_state["edit_draft"] = draft_copy

edit = st.session_state["edit_draft"]

# ------------------------------------------------------------------
#  Header
# ------------------------------------------------------------------
step_header(step=4, total=5, title="Review & Edit Your Draft")

# ------------------------------------------------------------------
#  Titles block (unchanged)
# ------------------------------------------------------------------
st.markdown("### Titles")
st.session_state.edit_draft["title"] = st.text_input(
    "Main title", value=edit.get("title", ""), key="ed_title")

cols_alt = st.columns(2)
with cols_alt[0]:
    st.session_state.edit_draft["alt_title_1"] = st.text_input(
        "Alt title 1", value=edit.get("alt_title_1", ""), key="ed_alt1")
with cols_alt[1]:
    st.session_state.edit_draft["alt_title_2"] = st.text_input(
        "Alt title 2", value=edit.get("alt_title_2", ""), key="ed_alt2")

# ------------------------------------------------------------------
#  Sections – accordion style, ordered: In Practice → Introduction → Description
# ------------------------------------------------------------------
st.markdown('<div class="alba-card">', unsafe_allow_html=True)
st.markdown("### Sections")

with st.expander("In Practice", expanded=True):
    st.session_state.edit_draft["in_practice"] = st.text_area(
        "In Practice", value=edit.get("in_practice", ""), height=180, key="ed_practice")

with st.expander("Introduction"):
    st.session_state.edit_draft["introduction"] = st.text_area(
        "Introduction", value=edit.get("introduction", ""), height=180, key="ed_intro")

with st.expander("Description"):
    st.session_state.edit_draft["description"] = st.text_area(
        "Description", value=edit.get("description", ""), height=220, key="ed_desc")

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
#  Rewards – collapsible cards with Up/Down re-ordering
# ------------------------------------------------------------------
st.markdown("### Rewards")
st.markdown('<div class="alba-card">', unsafe_allow_html=True)


def ensure_rewards():
    if not isinstance(st.session_state.edit_draft.get("rewards"), list):
        st.session_state.edit_draft["rewards"] = []


ensure_rewards()

# Add reward
if st.button("➕ Add reward", help="Add a new reward level"):
    st.session_state.edit_draft["rewards"].append(
        {"name": "", "description": "", "price": ""})
    st.rerun()

# Render reward blocks
to_delete, move_up, move_down = None, None, None
rewards = st.session_state.edit_draft["rewards"]

for i, r in enumerate(rewards):
    name_preview = r.get("name", "").strip() or f"Reward {i+1}"
    price_preview = str(r.get("price", "")).strip()
    header = f"{name_preview} — €{price_preview}" if price_preview else name_preview

    with st.expander(header, expanded=True):
        # Re-order & delete controls
        row = st.columns([1, 1, 1, 6])
        with row[0]:
            if i > 0 and st.button("⬆️", key=f"up_{i}", help="Move up"):
                move_up = i
        with row[1]:
            if i < len(rewards) - 1 and st.button("⬇️", key=f"down_{i}", help="Move down"):
                move_down = i
        with row[2]:
            if st.button("🗑️", key=f"delete_{i}", help="Remove"):
                to_delete = i
        # Editable fields
        st.session_state.edit_draft["rewards"][i]["name"] = st.text_input(
            "Name", value=r.get("name", ""), key=f"r_name_{i}")
        st.session_state.edit_draft["rewards"][i]["description"] = st.text_area(
            "Description", value=r.get("description", ""), height=100, key=f"r_desc_{i}")
        st.session_state.edit_draft["rewards"][i]["price"] = st.text_input(
            "Price (€)", value=str(r.get("price", "")), key=f"r_price_{i}")

# Apply deletions or movements outside the loop
if to_delete is not None:
    del rewards[to_delete]
    st.rerun()

if move_up is not None:
    rewards[move_up - 1], rewards[move_up] = rewards[move_up], rewards[move_up - 1]
    st.rerun()

if move_down is not None:
    rewards[move_down +
            1], rewards[move_down] = rewards[move_down], rewards[move_down + 1]
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
#  Footer actions
# ------------------------------------------------------------------
st.markdown('<div class="alba-actions">', unsafe_allow_html=True)
col_back, col_publish = st.columns(2)

with col_back:
    if st.button("← Back", type="secondary"):
        st.switch_page("pages/3_Project_Details.py")

with col_publish:
    if st.button("Publish 🚀", type="primary"):
        # Clean rewards
        clean_rewards = []
        for r in rewards:
            clean_rewards.append({
                "name": (r.get("name") or "").strip(),
                "description": (r.get("description") or "").strip(),
                "price": (r.get("price") or "").strip(),
            })

        st.session_state.draft = {
            "title": (edit.get("title") or "").strip(),
            "alt_title_1": (edit.get("alt_title_1") or "").strip(),
            "alt_title_2": (edit.get("alt_title_2") or "").strip(),
            "introduction": edit.get("introduction") or "",
            "in_practice": edit.get("in_practice") or "",
            "description": edit.get("description") or "",
            "rewards": clean_rewards,
        }

        st.switch_page("pages/5_Success.py")

st.markdown('</div>', unsafe_allow_html=True)
