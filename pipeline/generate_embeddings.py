from utils.preprocess_dataset import load_schoolraising_documents
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import csv  # 👈 nuovo import

def generate_embeddings():
    # Step 1: Load preprocessed documents
    documents = load_schoolraising_documents()
    print(f"✅ Loaded {len(documents)} documents for embedding.")

    # Step 2: Initialize the embeddings model
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs={"batch_size": 32})

    print(f"🔍 Using embedding model: {model_name}")

    # Step 3: Generate the FAISS vectorstore from documents
    vectorstore = FAISS.from_documents(documents, embeddings)
    print("✅ Embeddings generated.")

    # Step 4: Save the vectorstore locally
    vectorstore.save_local("embeddings/vector_db")
    print("💾 Vectorstore saved to embeddings/vector_db")

    # Step 5: Save debug info to CSV
    with open("embeddings/debug_vector_log.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Length", "Preview", "Category", "School", "Project ID"])
        for i, doc in enumerate(documents):
            writer.writerow([
                i,
                len(doc.page_content.split()),
                doc.page_content[:30].replace("\n", " ") + "...",
                doc.metadata.get("category", ""),
                doc.metadata.get("school", ""),
                doc.metadata.get("id", "")
            ])
    print("🧾 Log CSV salvato in embeddings/debug_vector_log.csv")

if __name__ == "__main__":
    generate_embeddings()


