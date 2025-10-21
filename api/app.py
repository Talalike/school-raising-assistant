import streamlit as st
from pipeline.rag_pipeline import run_pipeline
import io
import re
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Alba Assistant", layout="centered")
st.title("📚 Alba | School Campaign Assistant")

st.markdown("Answer the questions below to generate your school crowdfunding campaign draft. You can edit each section before saving it.")

# INPUT FORM
with st.form("campaign_form"):
    school_name = st.text_input("🏫 School name")
    project_category = st.selectbox("📂 Project category", [
                                    "Spazio e Strutture", "Tecnologia", "Arte e Cultura", "Musica", "Sport", "Eventi", "Ambiente", "Inclusione", "Altro"])
    user_input_1 = st.text_area("📌 What is your project about?", height=80)
    user_input_2 = st.text_area("💡 Why is this project important?", height=80)
    user_input_3 = st.text_area(
        "🎁 Would you like to offer rewards?", height=80)
    submitted = st.form_submit_button("🚀 Generate Draft")

if submitted:
    with st.spinner("Generating your campaign draft. Please wait..."):
        output = run_pipeline(school_name, project_category,
                              user_input_1, user_input_2, user_input_3)

    # --- PARSE SECTIONS FROM TEXT ---
    def extract_section(label, text):
        pattern = rf"{label}[:\*]*.*?\n(.+?)(?=\n\n|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    title = extract_section("Title", output)
    alt1 = extract_section("Alternative Title 1", output)
    alt2 = extract_section("Alternative Title 2", output)
    in_practice = extract_section("In Practice", output)
    intro = extract_section("Introduction", output)
    desc = extract_section("Description", output)

    rewards = []
    reward_blocks = re.findall(r"\*\*(.*?)\*\*\s*(.*?)\s*€(\d+)", output)
    for name, desc_reward, amount in reward_blocks:
        rewards.append(
            {"name": name.strip(), "desc": desc_reward.strip(), "amount": amount.strip()})

    st.subheader("📝 Edit your draft")

    title = st.text_input("📌 Title", value=title)
    alt1 = st.text_input("🎯 Alternative Title 1", value=alt1)
    alt2 = st.text_input("🎯 Alternative Title 2", value=alt2)
    in_practice = st.text_input("🧠 In Practice", value=in_practice)
    intro = st.text_area("📖 Introduction", value=intro, height=200)
    desc = st.text_area("📝 Description", value=desc, height=200)

    reward_inputs = []
    st.markdown("🎁 **Rewards**")
    for i, r in enumerate(rewards):
        name = st.text_input(f"🎁 Reward {i+1} - Name", value=r["name"])
        description = st.text_area(
            f"✏️ Reward {i+1} - Description", value=r["desc"], height=80)
        amount = st.text_input(
            f"💶 Reward {i+1} - Amount (€)", value=r["amount"])
        reward_inputs.append(
            {"name": name, "desc": description, "amount": amount})

    # --- GENERATE PDF ---
    def generate_pdf(sections):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        story = []

        for label, content in sections.items():
            story.append(Paragraph(f"<b>{label}</b>", styles["Heading1"]))
            story.append(Paragraph(content.replace(
                "\n", "<br/>"), styles["BodyText"]))
            story.append(Spacer(1, 12))

        if reward_inputs:
            story.append(Paragraph("<b>Rewards</b>", styles["Heading1"]))
            for r in reward_inputs:
                reward_text = f"<b>{r['name']}</b><br/>{r['desc']}<br/><i>€{r['amount']}</i>"
                story.append(Paragraph(reward_text, styles["BodyText"]))
                story.append(Spacer(1, 6))

        doc.build(story)
        buffer.seek(0)
        return buffer

    final_sections = {
        "Title": title,
        "Alternative Title 1": alt1,
        "Alternative Title 2": alt2,
        "In Practice": in_practice,
        "Introduction": intro,
        "Description": desc,
    }

    pdf = generate_pdf(final_sections)
    st.download_button("📥 Download as PDF", pdf,
                       file_name="campaign_draft.pdf")
    st.success("✅ Your campaign draft is ready! You can download it as a PDF.")
