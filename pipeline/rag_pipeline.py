import os
from dotenv import load_dotenv
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import re
from pathlib import Path

# 🔐 Env + Load
load_dotenv()
db_path = "embeddings/vector_db"
vectorstore_path = Path(f"{db_path}/index.faiss")

# 🔍 Check if FAISS vectorstore exists before loading
if not vectorstore_path.exists():
    print("🚧 Vectorstore not found. Running embedding pipeline...")
    from pipeline.generate_embeddings import generate_embeddings
    generate_embeddings()
    print("✅ Embeddings generated successfully.\n")

# ✅ Ora possiamo caricare il vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.load_local(
    db_path,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# 🧠 Jaccard Similarity

def jaccard_similarity(str1, str2):
    tokens1 = set(str1.lower().split())
    tokens2 = set(str2.lower().split())
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union) if union else 0

# 📊 Validate Output Sections

def validate_output_sections(text):
    import re

    results = {}

    # 🎯 Titolo principale
    title_match = re.search(r"\*{0,2}\s*Title:\s*(.+)", text)
    if title_match:
        title = title_match.group(1).strip()
        results["title_length"] = len(title)
        results["title"] = title

    # 🎯 Titolo alternativo 1
    alt1_match = re.search(r"(?:🎯|\*{0,2})\s*Alternative Title 1:\s*(.+)", text)
    if alt1_match:
        alt1 = alt1_match.group(1).strip()
        results["alt_title_1"] = alt1
        results["alt_title_1_length"] = len(alt1)
    else:
        print("❌ Alternative Title 1 not found.")

    # 🎯 Titolo alternativo 2
    alt2_match = re.search(r"(?:🎯|\*{0,2})\s*Alternative Title 2:\s*(.+)", text)
    if alt2_match:
        alt2 = alt2_match.group(1).strip()
        results["alt_title_2"] = alt2
        results["alt_title_2_length"] = len(alt2)
    else:
        print("❌ Alternative Title 2 not found.")

    # 📖 Introduzione
    intro_match = re.search(r"\*{0,2}\s*Introduction:\*{0,2}\s*(.+?)\n+\*{0,2}\s*Description:", text, re.DOTALL)
    if intro_match:
        intro = intro_match.group(1).strip()
        results["introduction_length"] = len(intro)
        results["introduction"] = intro

    # 📝 Descrizione
    desc_match = re.search(r"\*{0,2}\s*Description:\*{0,2}\s*(.+?)\n+\*{0,2}\s*Rewards:", text, re.DOTALL)
    if desc_match:
        desc = desc_match.group(1).strip()
        results["description_length"] = len(desc)
        results["description"] = desc

    # 🧾 Log finale per verifica
    print("\n📏 Post-check – Sezione per sezione:")

    if "title_length" in results:
        print(f"🔠 Title length: {results['title_length']} characters")
        if results["title_length"] > 50:
            print("❌ Title too long! (>50)")

    if "alt_title_1_length" in results:
        print(f"🎯 Alt Title 1 length: {results['alt_title_1_length']} characters")
        if results["alt_title_1_length"] > 50:
            print("❌ Alt Title 1 too long! (>50)")

    if "alt_title_2_length" in results:
        print(f"🎯 Alt Title 2 length: {results['alt_title_2_length']} characters")
        if results["alt_title_2_length"] > 50:
            print("❌ Alt Title 2 too long! (>50)")

    if "introduction_length" in results:
        print(f"📚 Introduction length: {results['introduction_length']} characters")
        if results["introduction_length"] < 800:
            print("❌ Introduction too short! (<800)")

    if "description_length" in results:
        print(f"📜 Description length: {results['description_length']} characters")
        if results["description_length"] < 1000:
            print("❌ Description too short! (<1000)")

    print("\n🧪 Post-check complete.\n")

# 🔹 Custom Retriever

def custom_category_retriever(query: str, category: str, k: int = 5, k_base: int = 30):
    all_docs = vectorstore.similarity_search_with_score(query, k=k_base)
    matching = [doc for doc, _ in all_docs if doc.metadata.get("category") == category.lower()]
    others = [doc for doc, _ in all_docs if doc.metadata.get("category") != category.lower()]
    selected = (matching + others)[:k]

    print("\n📂 Retrieved Documents by Category Priority:\n")
    for i, doc in enumerate(selected, 1):
        cat = doc.metadata.get("category", "N/A")
        print(f"📄 Document {i}: Category = {cat}")

    return selected

# ⚖️ Core Function

def run_pipeline(school_name, project_category, user_input_1, user_input_2, user_input_3, k=5):
    retrieved_docs = custom_category_retriever(user_input_1, project_category, k=k, k_base=30)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = PromptTemplate.from_template("""
You are Alba, a virtual assistant that helps teachers and school communities in Italy create
compelling crowdfunding campaigns on the School Raising platform.

Your task is to generate a complete campaign draft in English based on:

1. The user's short inputs
2. A selection of relevant past school campaigns retrieved below

The retrieved examples reflect the typical style of successful Italian school campaigns — warm,
inclusive, educational, and community-driven. Use them as inspiration for tone, structure,
length, and content — but do not copy.

Based on the user's input, generate the following sections:

1. Title (main + 2 alternatives, each max 50 characters – strict limit)  
→ Generate three title options for the campaign:  
   - **Title**: the main, most natural choice  
   - **Alternative Title 1**: another clear and inspiring option  
   - **Alternative Title 2**: a different version with the same clarity and tone  
→ All three must be clear, educational, and inspiring.  
→ Avoid slogans, acronyms, or overly generic titles.  
→ Each title must stay within 50 characters (strict limit).  
→ Double-check the character count before output.  
→ Format them exactly like this:  

Title: [main title]  
Alternative Title 1: [first alternative]  
Alternative Title 2: [second alternative]  
                                      
2. In Practice (1 sentence, max 160 characters)  
→ Describe what the project will do, for whom, and why.  
→ Format: [specific action] + [target group] + [purpose or value].

3. Introduction (min 800, max 1000 characters)  
→ Explain the motivation for the project: What problem or opportunity does it address? Why now?  
→ What benefits will it bring to students, the school, or the community?  
→ Use a warm, empathetic tone.  
→ The school name must appear in the first sentence.  
→ DO NOT start the section with “At {school_name}”, “At our school”, or any equivalent formula.  
→ Start with a different sentence structure that includes the school name, but in a natural way.  
→ Example of correct start: “Students at Scuola XYZ are taking their first steps into coding…”  
→ If your first sentence starts with “At…”, you MUST regenerate it before returning the output.
→ You MUST ensure this section is between **800 and 1000 characters.**  
→ If the content exceeds 1000 characters, you MUST shorten it before returning the final result.  
→ If it is under 800, you MUST expand it before returning the final result.
→ If the Introduction exceeds 1000 characters, shorten it before returning.  
→ The output will be considered invalid if it does not fall within this range.  
→ Do NOT return any Introduction longer than 1000 characters.

4. Description (min 1000 characters)  
→ Write a discursive text, divided into clear, short paragraphs.  
→ Do not use bullet points unless strictly necessary.  
→ Clearly describe:
  - What activities will take place (based on the user inputs)
  - Who will be involved (students, teachers, partners)
  - What impact is expected (skills, inclusion, participation, etc.)
→ End with a warm, inclusive call to action inviting readers to support the project.  
→ You MUST ensure this section contains **at least 1000 characters**.  
→ If too short, expand it naturally with real, meaningful content (no filler, no repetition).  
→ Never return a Description shorter than 1000 characters.
                                      
5. Rewards (5–6 tiers)  
→ Always prioritize intangible rewards (e.g. thank-you video, invite to final event, digital updates).  
→ If no input is provided, generate relevant default rewards.  
→ Tangible rewards may follow as optional extras.  
→ Each reward must be written on three separate lines:
   Line 1: **Reward Name**
   Line 2: One-line description
   Line 3: Price (e.g., €5, €10, etc.)
→ Never place the description or price on the same line as the reward name.
→ Separate rewards with one blank line.
→ Make the list visually easy to scan and consistent across all rewards.
→ Do not suggest the reward “donazione libera”.


Global style rules:
Language: simple, accessible, non-bureaucratic English
Voice: inclusive “we”, involving families and community
Avoid: superlatives (“the best...”), marketing tone, acronyms, or technical terms
Ensure inclusivity: mention benefits for all students; avoid any barriers
Use short sentences, compact paragraphs, and reader-friendly formatting
No Politics, no sexism, no racism.

User input:
1. School name → {school_name}
2. Project category → {project_category}
3. What is your project about? → {user_input_1}
4. Why is this project important? → {user_input_2}
5. Would you like to offer rewards? → {user_input_3}

Retrieved example campaigns:
{context}

Now write the full campaign draft as specified above.
""")

    llm = ChatOpenAI(model="gpt-4o", temperature=0.3, openai_api_key=os.getenv("OPENAI_API_KEY"))
    prompt_input = {
        "context": context,
        "school_name": school_name,
        "project_category": project_category,
        "user_input_1": user_input_1,
        "user_input_2": user_input_2,
        "user_input_3": user_input_3
    }

    max_attempts = 3
    similarity_threshold = 0.8
    attempt = 0
    final_response = ""

    while attempt < max_attempts:
        attempt += 1
        print(f"\n🔄 Attempt {attempt}...")
        chain = prompt | llm
        response = chain.invoke(prompt_input)
        final_response = response.content if hasattr(response, "content") else response

        too_similar = any(
            jaccard_similarity(final_response, doc.page_content) > similarity_threshold
            for doc in retrieved_docs
        )

        if not too_similar:
            print("✅ Response seems original.")
            break
        else:
            print(f"⚠️ Similarity above threshold. Regenerating...")

    if too_similar:
        print("🚫 Max attempts reached. Final response may still be too similar.")

    print("\n📣 Final Campaign Draft:\n") # comment this line in case you want output only in JSON format and not text format
    print(final_response)                  # comment this line in case you want output only in JSON format and not text format
    validate_output_sections(final_response)

    parsed_output = parse_generated_output(final_response)
    print("📦 Parsed output to return to FastAPI:")
    print(parsed_output)
    return parsed_output

    # 👇 Funzione di parsing incorporata
def parse_generated_output(text):
    import re
    import logging

    parsed = {
        "title": "[MISSING]",
        "alt_title_1": "[MISSING]",
        "alt_title_2": "[MISSING]",
        "in_practice": "[MISSING]",
        "introduction": "[MISSING]",
        "description": "[MISSING]",
        "rewards": []
    }

    try:
        parsed["title"] = re.search(r"\*{0,2}\s*Title:\s*(.*)", text).group(1).strip()
        parsed["alt_title_1"] = re.search(r"\*{0,2}\s*Alternative Title 1:\s*(.*)", text).group(1).strip()
        parsed["alt_title_2"] = re.search(r"\*{0,2}\s*Alternative Title 2:\s*(.*)", text).group(1).strip()
    except Exception as e:
        logging.warning("⚠️ Title block parsing failed: %s", e)

    try:
        parsed["in_practice"] = re.search(r"\*{0,2}\s*In Practice:\s*(.*)", text).group(1).strip()
    except Exception as e:
        logging.warning("⚠️ In Practice parsing failed: %s", e)

    try:
        intro_match = re.search(r"\*{0,2}\s*Introduction:\*{0,2}\s*(.+?)\n+\*{0,2}\s*Description:", text, re.DOTALL)
        if intro_match:
            parsed["introduction"] = intro_match.group(1).strip()
        else:
            logging.warning("⚠️ Introduction section not matched.")
    except Exception as e:
        logging.warning("⚠️ Introduction parsing failed: %s", e)

    try:
        desc_match = re.search(r"\*{0,2}\s*Description:\*{0,2}\s*(.+?)\n+\*{0,2}\s*Rewards:", text, re.DOTALL)
        if desc_match:
            parsed["description"] = desc_match.group(1).strip()
        else:
            logging.warning("⚠️ Description section not matched.")
    except Exception as e:
        logging.warning("⚠️ Description parsing failed: %s", e)

    try:
        rewards_match = re.search(r"\*{0,2}\s*Rewards:\*{0,2}\s*(.+)", text, re.DOTALL)
        if rewards_match:
            rewards_raw = rewards_match.group(1).strip()
            reward_blocks = re.split(r"\n{2,}", rewards_raw)
            for block in reward_blocks:
                lines = block.strip().split("\n")
                if len(lines) >= 3:
                    name = lines[0].replace("**", "").strip()
                    desc = lines[1].strip()
                    price = re.sub(r"[^\d]", "", lines[2])
                    try:
                        parsed["rewards"].append({
                            "name": name,
                            "description": desc,
                            "price": int(price)
                        })
                    except ValueError:
                        continue
    except Exception as e:
        logging.warning("⚠️ Rewards parsing failed: %s", e)

    # Fallback
    for key in parsed:
        if key != "rewards" and (not parsed[key] or parsed[key] == "[MISSING]"):
            parsed[key] = f"[Auto-filled content for missing section: {key}]"
    if not parsed["rewards"]:
        parsed["rewards"] = [{
            "name": "Supporter",
            "description": "Thank you for supporting our school project!",
            "price": 10
        }]

    return parsed