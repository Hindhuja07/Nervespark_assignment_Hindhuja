import os
import tempfile
import shutil
from git import Repo
from pathlib import Path
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from utils import simple_chunker, is_text_file
import uuid
import json

CHROMA_DIR = "chroma_db"

def ingest_github_or_path(source: str, max_files=200, embedding_model="all-MiniLM-L6-v2"):
    """
    Clone or use local path, walk files, chunk, embed and upsert to Chroma.
    Returns dict with ok True/False and metadata.
    """
    tmpdir = None
    try:
        if source.startswith("http://") or source.startswith("https://") or source.endswith(".git"):
            tmpdir = tempfile.mkdtemp()
            Repo.clone_from(source, tmpdir, depth=1)
            repo_path = tmpdir
        else:
            repo_path = source
            if not os.path.exists(repo_path):
                return {"ok": False, "error": "Path not found"}

        # collect text files
        files = []
        for root, dirs, filenames in os.walk(repo_path):
            for fn in filenames:
                path = os.path.join(root, fn)
                if is_text_file(path) and os.path.getsize(path) < 2_000_000:
                    files.append(path)
                if len(files) >= max_files:
                    break
            if len(files) >= max_files:
                break

        chunks = []
        for p in files:
            rel = os.path.relpath(p, repo_path)
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            file_chunks = simple_chunker(text, rel)
            for c in file_chunks:
                c["metadata"]["path"] = rel
                chunks.append(c)

        # setup chroma
        chroma_dir = os.path.abspath(CHROMA_DIR + "_" + str(uuid.uuid4())[:8])
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=chroma_dir))
        collection = client.create_collection(name="code_docs")
        embedder = SentenceTransformer(embedding_model)

        ids, docs, metadatas, embeddings = [], [], [], []
        for i, ch in enumerate(chunks):
            ids.append(f"{i}")
            docs.append(ch["page_content"])
            metadatas.append(ch["metadata"])
            embeddings.append(embedder.encode(ch["page_content"]).tolist())

        collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
        client.persist()
        return {"ok": True, "n_chunks": len(chunks), "chroma_dir": chroma_dir}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
