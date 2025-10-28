# 🎓 School Raising Assistant (MVP)

**Alba** is an AI-powered virtual assistant designed to help teachers and school project creators write compelling crowdfunding campaigns in just a few minutes, based on a few simple inputs.

## 🚀 Objective

Build a functional MVP (internal demo) in 3 weeks that:
- Transforms 5 user inputs into a campaign draft
- Suggests title, description, rewards, CTA, and tags
- Is inspired by ~150 past school campaigns (CSV + FAISS)
- Uses a hosted LLM (OpenAI GPT-4o) and lightweight RAG pipeline

## 🧰 Tech Stack

- Python 3.12.4
- FastAPI
- FAISS + SentenceTransformers
- OpenAI API (GPT-4o)
- Pandas
- Local vector database

## 📦 Setup

### 1. Clone the repository

```bash
git clone git@github.com:talalike/school-raising-assistant.git
cd school-raising-assistant

Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

Install dependencies
pip install -r requirements.txt

Please run first this command
--> python -m pipeline.generate_embeddings ( in order to generate your embeddings) 
after that you can run 
--> python interactive_assistant.py


