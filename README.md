# 🎓 School Raising Assistant (MVP)

**Alba** is an AI-powered virtual assistant designed to help teachers and school project creators write compelling crowdfunding campaigns in just a few minutes, based on a few simple inputs.

## 🚀 Objective

Build a functional MVP (internal demo) in 3 weeks that:
- Transforms 4–6 user inputs into a campaign draft
- Suggests title, description, rewards, CTA, and tags
- Is inspired by ~150 past school campaigns (CSV + FAISS)
- Uses a hosted LLM (OpenAI GPT-4o) and lightweight RAG pipeline

## 🧰 Tech Stack

- Python 3.12.4
- FastAPI
- FAISS + SentenceTransformers
- OpenAI API (GPT-4o)
- Pandas, Loguru
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

Project Structure
school-raising-assistant/
├── app.py                # FastAPI entrypoint
├── requirements.txt      # Python dependencies
├── data/                 # CSV dataset (150 school campaigns)
├── embeddings/           # FAISS vector DB + embeddings
├── pipeline/             # RAG pipeline and prompt templates
├── utils/                # Helper scripts (e.g. data loader)
├── notebooks/            # Jupyter testing notebooks
├── frontend/             # Optional basic UI form

