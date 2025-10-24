# front_end/main_fe.py
import streamlit as st
from utils_fe import brand_paths, image_available

# Get logo path for the page icon (fallback to emoji if missing)
paths = brand_paths()
icon_path = paths["mark"] if image_available(paths["mark"]) else "🎒"

st.set_page_config(
    page_title="Alba — SchoolRaising’s Virtual Assistant",
    page_icon=icon_path,
    layout="wide",
)

# Instantly redirect to the landing page
st.switch_page("pages/1_Landing.py")
