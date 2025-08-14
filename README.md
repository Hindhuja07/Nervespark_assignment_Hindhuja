📄 Code Docs RAG — Minimal Demo

A minimal Retrieval-Augmented Generation (RAG) pipeline for querying code documentation using Streamlit.
This project lets you ingest documents, parse them, and interactively query them using a retriever + LLM pipeline.

🚀 Features

Document ingestion from multiple sources.

Custom parsers to preprocess text.

Retriever for semantic search.

Streamlit UI for interactive querying.

Minimal, easy-to-extend structure.

📂 Project Structure
rag_pipeline/
│
├── app_streamlit.py   # Streamlit app UI
├── ingest.py          # Document ingestion logic
├── parsers.py         # Parsing and preprocessing functions
├── retriever.py       # Vector store + retriever setup
├── utils.py           # Helper functions
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
└── init.py        # Marks package as importable

🛠 Prerequisites

Python 3.9+

pip package manager

(Optional) Virtual environment tool like venv

⚡ Quickstart (Windows PowerShell)

Clone or Download

git clone <your-repo-url>
cd rag_pipeline


Create Virtual Environment

python -m venv .venv


Activate Virtual Environment

.venv\Scripts\Activate


Install Requirements

pip install -r requirements.txt


Run Streamlit App

streamlit run app_streamlit.py

💻 Quickstart (macOS / Linux)
git clone <your-repo-url>
cd rag_pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_streamlit.py

📜 How to Use

Ingest Documents

Place your documents in the appropriate location or modify ingest.py to load from your source.

Run:

python ingest.py


This will parse and store embeddings in your vector store.

Run the App

Start Streamlit:

streamlit run app_streamlit.py


Open the local URL shown in the terminal (usually http://localhost:8501).

Ask Questions

Type your query in the Streamlit UI.

The retriever will fetch relevant chunks.

The LLM will generate a context-aware answer.

🔍 File-by-File Guide

app_streamlit.py → Defines the interactive chat interface for RAG queries.

ingest.py → Handles loading, splitting, and embedding documents.

parsers.py → Contains document parsing logic.

retriever.py → Configures the retriever for semantic search.

utils.py → Utility helpers for logging, formatting, etc.

requirements.txt → List of all dependencies.
