#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
School-Raising RAG pipeline  ✦  2025-10-15

Usage examples
--------------

# 1) One-time ingest  (adds _EN columns if --pretranslate)
python sr_rag.py ingest \
  --csv data/campaigns.csv \
  --db  vectorstore/sr_db \
  --pretranslate

# 2) Ask a corpus question
python sr_rag.py query \
  --db  vectorstore/sr_db \
  --q   "What makes a strong rewards section?"

# 3) Generate a brand-new campaign draft from teacher inputs
python sr_rag.py draft \
  --db         vectorstore/sr_db \
  --category   "Technology" \
  --school     "I.C. Galileo Galilei" \
  --goal       "Set up a robotics club with Lego Spike kits" \
  --importance "Hands-on STEM boosts critical thinking & inclusion" \
  --who        "Grade-7 students, STEM teachers, local mentors" \
  --rewards    "Thank-you video, invite to demo day, personalised e-certificate" \
  --k 3
"""

from transformers import MarianMTModel, MarianTokenizer
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import argparse
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

# ── Embeddings & Vector DB ──────────────────────────────────────────

# ── LLM (Groq → Llama 3) ────────────────────────────────────────────

# ── Optional IT→EN translation ──────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

REQUIRED_COLS = [
    "Category", "School", "Title",
    "Introduction", "In Practice", "Description", "rewards",
    "Goal (€)", "Totale Raccolto (€)", "Stato"
]
TEXT_COLS = ["Title", "Introduction", "In Practice", "Description", "rewards"]

# ── Prompt templates ────────────────────────────────────────────────
SYSTEM_CAMPAIGN_PROMPT = """
You are an assistant that drafts **English crowdfunding campaigns for Italian K-12 schools**.
Follow these non-negotiable rules:

• Structure & length:
  1) Title……………… ≤ 50 chars
  2) In Practice…… 1 sentence, ≤ 160 chars
  3) Introduction… ≤ 700 chars
  4) Description… ≤ 1000 chars, bullet-points allowed, finish with a gentle CTA
  5) Rewards……… 5–6 tiers: short title + one-line desc, intangible first, never “donazione libera”

• Tone & style:
  – Warm, inclusive “we / together”, simple language, no superlatives, no jargon.
  – School-community vocabulary (workshop, inclusion, citizenship…) but no copy-paste.
  – Mention benefits for **all** students; avoid barriers, politics, or sensitive content.

Never break these caps.  If a section would exceed its limit, shorten it.
"""
USER_CAMPAIGN_TEMPLATE = """
## User inputs
Category: {category}
School: {school}

Q1 Goal: {goal}
Q2 Importance: {importance}
Q3 Who is involved: {who}
Q4 Reward ideas: {rewards}

## Inspiration snippets (do NOT copy; take tone & structure cues only)
{context}

✦ Draft a full campaign in the required 5 sections, strictly observing all length and style rules.
"""

# ── Helpers ─────────────────────────────────────────────────────────


def get_llm(temp: float = 0.25):
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=temp,
        max_tokens=450  #  guard-rail
    )


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )


@dataclass
class Doc:
    page_content: str
    metadata: Dict[str, Any]


def ensure_cols(df: pd.DataFrame):
    miss = [c for c in REQUIRED_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

# ----- Optional offline translator -----


def translate_it_en(texts: List[str], model="Helsinki-NLP/opus-mt-it-en", batch=8):
    tok = MarianTokenizer.from_pretrained(model)
    mdl = MarianMTModel.from_pretrained(model)
    out = []
    for i in range(0, len(texts), batch):
        sub = [t if isinstance(t, str) else "" for t in texts[i:i+batch]]
        gen = mdl.generate(**tok(sub, return_tensors="pt",
                           padding=True, truncation=True, max_length=512))
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out

# ----- Build FAISS -----


def build_docs(df: pd.DataFrame) -> List[Doc]:
    docs = []
    for _, row in df.iterrows():
        blob = "\n".join(f"{c}: {str(row.get(c,''))}" for c in TEXT_COLS if str(
            row.get(c, "")).strip())
        docs.append(Doc(page_content=blob, metadata={
                    "Category": row["Category"]}))
    return docs


def save_faiss(docs: List[Doc], db_path: str):
    vs = FAISS.from_texts(
        texts=[d.page_content for d in docs],
        embedding=get_embeddings(),
        metadatas=[d.metadata for d in docs],
        distance_strategy=DistanceStrategy.COSINE
    )
    os.makedirs(db_path, exist_ok=True)
    vs.save_local(db_path)
    print(f"✅  Saved FAISS → {db_path}")


def load_faiss(db_path: str) -> FAISS:
    return FAISS.load_local(
        folder_path=db_path,
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.COSINE
    )


def nearest_neighbors(vs: FAISS, query: str, cat: str, k: int):
    """Filter by Category, then semantic search."""
    cat_docs = vs.similarity_search(
        query=query,
        k=20  # rough cut
    )
    same_cat = [d for d in cat_docs if d.metadata.get(
        "Category", "").lower() == cat.lower()]
    return same_cat[:k]


def trim(txt: str, max_chars=400):
    txt = txt.strip()
    return txt[:max_chars] + ("…" if len(txt) > max_chars else "")

# ── Commands ────────────────────────────────────────────────────────


def cmd_ingest(a):
    df = pd.read_csv(a.csv)
    ensure_cols(df)

    if a.pretranslate:
        print("⇢ Translating IT→EN…")
        for c in TEXT_COLS:
            df[f"{c}_EN"] = translate_it_en(
                df[c].astype(str).fillna("").tolist())
        out_csv = a.csv.replace(".csv", "_with_EN.csv")
        df.to_csv(out_csv, index=False)
        print(f"✅  Saved translated CSV → {out_csv}")

    docs = build_docs(df)
    save_faiss(docs, a.db)


def cmd_query(a):
    vs = load_faiss(a.db)
    ctx = nearest_neighbors(vs, a.q, "", a.k)
    blob = "\n\n---\n\n".join(trim(d.page_content, 500) for d in ctx)

    llm = get_llm(0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer in English, grounded in the snippets. If unsure, say so."),
        ("user", f"Question: {a.q}\n\nSnippets:\n{blob}")
    ])
    answer = (prompt | llm).invoke({})
    print("\n🟢 Answer:\n", answer.content)


def cmd_draft(a):
    vs = load_faiss(a.db)
    user_context = f"{a.goal} {a.importance} {a.who} {a.rewards}"
    neighbors = nearest_neighbors(vs, user_context, a.category, a.k)
    ctx_blob = "\n\n---\n\n".join(trim(d.page_content) for d in neighbors)

    llm = get_llm(0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_CAMPAIGN_PROMPT),
        ("user", USER_CAMPAIGN_TEMPLATE)
    ])
    filled = {
        "category": a.category,
        "school":   a.school,
        "goal":     a.goal,
        "importance": a.importance,
        "who":      a.who,
        "rewards":  a.rewards or "–",   # fallback
        "context":  ctx_blob or " "
    }
    draft = (prompt | llm).invoke(filled)
    print("\n📄  Generated campaign draft\n" + "-"*40 + "\n")
    print(draft.content)

# ── CLI wiring ──────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="School-Raising RAG CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ingest
    pin = sub.add_parser("ingest", help="Build FAISS DB from CSV")
    pin.add_argument("--csv", required=True)
    pin.add_argument("--db",  required=True)
    pin.add_argument("--pretranslate", action="store_true")
    pin.set_defaults(func=cmd_ingest)

    # query
    pq = sub.add_parser("query", help="Q&A over past campaigns")
    pq.add_argument("--db", required=True)
    pq.add_argument("--q",  required=True)
    pq.add_argument("--k", type=int, default=4)
    pq.set_defaults(func=cmd_query)

    # draft
    pdft = sub.add_parser("draft", help="Generate a new campaign draft")
    pdft.add_argument("--db",       required=True)
    pdft.add_argument("--category", required=True)
    pdft.add_argument("--school",   required=True)
    pdft.add_argument("--goal",     required=True)
    pdft.add_argument("--importance", required=True)
    pdft.add_argument("--who",      required=True)
    pdft.add_argument("--rewards",  default="")
    pdft.add_argument("--k", type=int, default=3,
                      help="top-K example snippets")
    pdft.set_defaults(func=cmd_draft)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
