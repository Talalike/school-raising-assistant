import pandas as pd
import re
import os
from typing import List
from langchain_core.documents import Document

def clean_text(text: str) -> str:
    """Clean text by removing HTML entities and trimming spaces."""
    if pd.isna(text):
        return ""
    return re.sub(r"&[a-z]+;", "", str(text)).strip()

def load_schoolraising_documents() -> List[Document]:
    """
    Load and preprocess School Raising campaign data from the Excel file
    located in the 'data/' folder, and return a list of LangChain Documents.
    """
    # Costruisce un path assoluto dinamico, valido da qualsiasi posizione
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "..", "data", "SchoolRaising_Dataset.xlsx")

    df = pd.read_excel(path)

    # Clean relevant text fields
    text_fields = ["Title", "In Practice", "Introduction", "Description", "rewards"]
    for field in text_fields:
        df[field] = df[field].apply(clean_text)

    documents = []
    for _, row in df.iterrows():
        full_text = f"""📌 Title: {row['Title']}
✅ In Practice: {row['In Practice']}
📖 Introduction: {row['Introduction']}
📝 Description: {row['Description']}
🎁 Rewards: {row['rewards']}"""
    
# ⛔ Filtro: ignora documenti troppo brevi
        if len(full_text.split()) < 30:
           continue

        metadata = {
            "category": str(row["Category"]).lower().strip(),
            "school": row["School"],
            "id": int(row["ID Campagna"])
        }

        documents.append(Document(page_content=full_text, metadata=metadata))

 # ✅ DEBUG: stampa i primi 3 documenti per verifica
        for d in documents[:3]:
           print("\n🔍 Documento di esempio:")
           print("→ Categoria:", d.metadata.get("category"))
           print("→ Scuola:", d.metadata.get("school"))
           print("→ ID:", d.metadata.get("id"))
           print("→ Inizio contenuto:", d.page_content[:200], "...\n")

    return documents

if __name__ == "__main__":
    print("🔧 Running data preprocessing test...\n")
    load_schoolraising_documents()