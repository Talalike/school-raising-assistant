import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv

# Load .env for local dev
load_dotenv()

API_BASE = os.getenv("ALBA_API_BASE", "").rstrip("/")
GENERATE_URL = f"{API_BASE}/generate_campaign" if API_BASE else None

st.set_page_config(
    page_title="Alba Assistant — Draft Generator", layout="wide")
st.title("🎒 Alba — Campaign Draft Generator")

if not GENERATE_URL:
    st.error("ALBA_API_BASE is not set. Put it in your .env or export it.")
    st.stop()

with st.form("campaign_form", clear_on_submit=False):
    school_name = st.text_input("School name", placeholder="Liceo Manzoni")
    project_category = st.selectbox(
        "Project category",
        ["Technology", "Travels", "Art and entertainment", "Social Activism",
         "Environment", "School libraries", "Space and structures"]
    )

    user_input_1 = st.text_area(
        "What is your project about?",
        help="Briefly describe the idea: what do you want to do, who is involved, and what makes it special? "
             "(E.g. “We want to create a theater performance on bullying, involving 3 classes and a local artist.”)",
        height=120,
    )

    user_input_2 = st.text_area(
        "Why is this project important for your school or community?",
        help="Tell us what motivated you. What need or dream are you responding to? "
             "(E.g. “Our students often feel excluded and need new ways to express themselves.”)",
        height=120,
    )

    user_input_3 = st.text_area(
        "Would you like to offer something to the people who support your project?",
        help="You can mention a small gift, a public thank you, or an invitation to join an activity. "
             "(E.g. “A thank-you video made by students, or the chance to attend the final event.”)",
        height=100,
    )

    submitted = st.form_submit_button("Generate draft ✨")


def _validate():
    missing = []
    for k, v in {
        "School name": school_name,
        "Project category": project_category,
        "What is your project about?": user_input_1,
        "Why is this project important…": user_input_2,
        "Supporter offer": user_input_3,
    }.items():
        if not v or (isinstance(v, str) and not v.strip()):
            missing.append(k)
    return missing


if submitted:
    missing = _validate()
    if missing:
        st.warning("Please fill all fields: " + ", ".join(missing))
        st.stop()

    payload = {
        "school_name": school_name.strip(),
        "project_category": project_category,
        "user_input_1": user_input_1.strip(),
        "user_input_2": user_input_2.strip(),
        "user_input_3": user_input_3.strip(),
    }

    with st.spinner("Generating draft…"):
        try:
            r = requests.post(GENERATE_URL, json=payload, timeout=180)
            r.raise_for_status()
            data = r.json()
        except requests.HTTPError as e:
            st.error(
                f"API error: {e}\n{getattr(e.response, 'text', '')[:800]}")
            st.stop()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    st.success("Draft generated ✅")

    # Titles
    st.header(data.get("title", "Untitled"))
    alt_cols = st.columns(2)
    with alt_cols[0]:
        st.caption("Alt title 1")
        st.write(data.get("alt_title_1", "—"))
    with alt_cols[1]:
        st.caption("Alt title 2")
        st.write(data.get("alt_title_2", "—"))

    # Body sections
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Introduction")
        st.write(data.get("introduction", ""))
        st.subheader("In Practice")
        st.write(data.get("in_practice", ""))
    with c2:
        st.subheader("Description")
        st.write(data.get("description", ""))

    # Rewards (list of objects {name, description, price})
    st.subheader("Rewards")
    rewards = data.get("rewards", [])
    if isinstance(rewards, list) and rewards:
        # Normalize objects
        norm = []
        for rwd in rewards:
            if isinstance(rwd, dict):
                norm.append({
                    "Name": rwd.get("name", ""),
                    "Description": rwd.get("description", ""),
                    "Price (€)": rwd.get("price", "")
                })
            else:
                # fallback if backend ever returns strings
                norm.append(
                    {"Name": str(rwd), "Description": "", "Price (€)": ""})
        st.dataframe(norm, use_container_width=True)
    else:
        st.info("No rewards suggested.")

    # Copy/export helpers
    with st.expander("Raw JSON"):
        st.code(json.dumps(data, indent=2))
