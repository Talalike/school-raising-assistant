import os
from dotenv import load_dotenv
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import re

# 🧠 Jaccard Similarity

def jaccard_similarity(str1, str2):
    tokens1 = set(str1.lower().split())
    tokens2 = set(str2.lower().split())
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union) if union else 0

# 📊 Validate Output Sections

def validate_output_sections(text):
    results = {}
    title_match = re.search(r"Title[:\*]*.*?\s*(.+)", text)
    intro_match = re.search(r"Introduction[:\*]*.*?\n(.+?)(?:\n\n|\Z)", text, re.DOTALL)
    desc_match = re.search(r"Description[:\*]*.*?\n+(.+?)(?=\n\n(?:\*\*|Rewards|\ud83d\udcda|\Z))", text, re.DOTALL)

    if title_match:
        title = title_match.group(1).strip()
        results["title_length"] = len(title)
        results["title"] = title

    if intro_match:
        intro = intro_match.group(1).strip()
        results["introduction_length"] = len(intro)
        results["introduction"] = intro

    if desc_match:
        desc = desc_match.group(1).strip()
        results["description_length"] = len(desc)
        results["description"] = desc

    print("\n📏 Post-check – Sezione per sezione:")
    if "title_length" in results:
        print(f"🔠 Title length: {results['title_length']} characters")
        if results["title_length"] > 50:
            print("❌ Title too long! (>50)")

    if "introduction_length" in results:
        print(f"📚 Introduction length: {results['introduction_length']} characters")
        if results["introduction_length"] < 800:
            print("❌ Introduction too short! (<800)")

    if "description_length" in results:
        print(f"📜 Description length: {results['description_length']} characters")
        if results["description_length"] < 1000:
            print("❌ Description too short! (<1000)")

    print("\n🧪 Post-check complete.\n")

# 🔐 Env + Load
load_dotenv()
db_path = "embeddings/vector_db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.load_local(db_path, embeddings=embedding_model, allow_dangerous_deserialization=True)

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

📌 Based on the user's input, generate the following sections:

1. Title (max 50 characters – strict limit)  
→ Must be clear, educational, and inspiring.  
→ Avoid advertising slogans or obscure acronyms.  
→ If the title exceeds 50 characters, you MUST regenerate it before returning the final answer.  
→ Do not include any title longer than 50 characters in the final output.
→ Double-check the character count before output.
                                      
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


🟩 Global style rules:
Language: simple, accessible, non-bureaucratic English
Voice: inclusive “we”, involving families and community
Avoid: superlatives (“the best...”), marketing tone, acronyms, or technical terms
Ensure inclusivity: mention benefits for all students; avoid any barriers
Use short sentences, compact paragraphs, and reader-friendly formatting
No Politics, no sexism, no racism.

🧑‍🏫 User input:
1. School name → {school_name}
2. Project category → {project_category}
3. What is your project about? → {user_input_1}
4. Why is this project important? → {user_input_2}
5. Would you like to offer rewards? → {user_input_3}

📚 Retrieved example campaigns:
{context}

✏ Now write the full campaign draft as specified above.
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

    print("\n📣 Final Campaign Draft:\n")
    print(final_response)
    validate_output_sections(final_response)
    return final_response
