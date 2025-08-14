# app_streamlit_hf.py
import os
import tempfile
import shutil
import zipfile
import requests
import streamlit as st
from pathlib import Path
from io import BytesIO
from git import Repo
from sentence_transformers import SentenceTransformer, util
import numpy as np
from rank_bm25 import BM25Okapi
import re
from tqdm import tqdm
import pygments.lexers
import pygments.formatters
from pygments import highlight

# ------------- Config -------------
st.set_page_config(page_title="Code Docs RAG - HF", layout="wide")
st.title("Code Documentation RAG")

# ------------- Helpers -------------
CODE_EXTS = (".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".md", ".rst", ".txt")

def download_repo_zip(github_url: str, branch="main"):
    if github_url.endswith("/"):
        github_url = github_url[:-1]
    if not github_url.startswith("https://github.com/"):
        raise ValueError("Provide a valid GitHub URL")
    parts = github_url.replace("https://github.com/", "").split("/")
    if len(parts) < 2:
        raise ValueError("Invalid GitHub URL")
    user, repo = parts[0], parts[1]
    for b in (branch, "main", "master"):
        url = f"https://codeload.github.com/{user}/{repo}/zip/refs/heads/{b}"
        r = requests.get(url, allow_redirects=True, timeout=30)
        if r.status_code == 200 and r.content[:4] == b'PK\x03\x04':
            return r.content
    raise RuntimeError("Failed to download repo zip")

def clone_repo(git_url: str, dest: str):
    Repo.clone_from(git_url, dest, depth=1)

def is_text_file(path: Path):
    try:
        with open(path, "rb") as f:
            raw = f.read(4096)
        return b'\x00' not in raw
    except Exception:
        return False

def read_supported_files(root_dir: Path, max_files=2000):
    files = []
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in CODE_EXTS and is_text_file(p):
            files.append(p)
            if len(files) >= max_files:
                break
    return files

def chunk_file_text(text: str, path: str, max_lines=200, overlap=20):
    lines = text.splitlines()
    chunks = []
    head_idx = [i for i,l in enumerate(lines) if re.match(r'^\s*#{1,6}\s+', l) or re.match(r'^\s*//', l)]
    sp = [0] + head_idx + [len(lines)]
    for i in range(len(sp)-1):
        s = sp[i]
        e = sp[i+1]
        block = "\n".join(lines[s:e]).strip()
        if block:
            if len(block.splitlines()) > max_lines:
                for start in range(0, len(block.splitlines()), max_lines - overlap):
                    sub = "\n".join(block.splitlines()[start:start+max_lines])
                    chunks.append({"text": sub, "path": path, "start_line": s+start+1, "end_line": s+start+len(sub.splitlines())})
            else:
                chunks.append({"text": block, "path": path, "start_line": s+1, "end_line": e})
    if not chunks and lines:
        for start in range(0, len(lines), max_lines - overlap):
            sub = "\n".join(lines[start:start+max_lines])
            chunks.append({"text": sub, "path": path, "start_line": start+1, "end_line": start+len(sub.splitlines())})
    return chunks

def reciprocal_rank_fusion(list_of_ranked_ids, k=10):
    score = {}
    for rank_list in list_of_ranked_ids:
        for i, docid in enumerate(rank_list):
            score[docid] = score.get(docid, 0) + 1.0/(50 + i)
    sorted_ids = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in sorted_ids][:k]

def pretty_code_block(code: str, lang_hint: str = ""):
    try:
        if lang_hint:
            lexer = pygments.lexers.get_lexer_by_name(lang_hint)
        else:
            lexer = pygments.lexers.guess_lexer(code)
    except Exception:
        lexer = pygments.lexers.TextLexer()
    formatter = pygments.formatters.HtmlFormatter()
    return pygments.highlight(code, lexer, formatter)

# ------------- Ingest -------------
@st.cache_data(show_spinner=False)
def ingest_from_github(url: str, tmp_root: str, max_files=1000):
    extracted = Path(tmp_root) / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    try:
        zip_bytes = download_repo_zip(url)
        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            zf.extractall(path=extracted)
    except Exception:
        clone_repo(url, str(extracted/"repo_clone"))
    files = read_supported_files(extracted, max_files=max_files)
    return files, extracted

@st.cache_data(show_spinner=False)
def ingest_from_local(path: str, max_files=1000):
    p = Path(path)
    files = read_supported_files(p, max_files=max_files)
    return files, p

def build_indices(files, embedding_model_name="all-MiniLM-L6-v2"):
    embedder = SentenceTransformer(embedding_model_name)
    all_chunks = []
    for f in tqdm(files, desc="Chunking files", leave=False):
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        chunks = chunk_file_text(text, str(f))
        all_chunks.extend(chunks)
    if not all_chunks:
        return [], None, None, None
    docs = [c["text"] for c in all_chunks]
    tokenized = [re.findall(r"\w+", d.lower()) for d in docs]
    bm25 = BM25Okapi(tokenized)
    emb_matrix = embedder.encode(docs, show_progress_bar=True, convert_to_numpy=True)
    return all_chunks, emb_matrix, bm25, embedder

def retrieve(query, all_chunks, emb_matrix, bm25, embedder, top_k=4):
    tokens = re.findall(r"\w+", query.lower())
    bm25_top = bm25.get_top_n(tokens, [c["text"] for c in all_chunks], n=top_k*5)
    bm25_ids = [str([c["text"] for c in all_chunks].index(d)) for d in bm25_top if d in [c["text"] for c in all_chunks]]
    q_emb = embedder.encode(query, convert_to_numpy=True)
    sims = (emb_matrix @ q_emb) / (np.linalg.norm(emb_matrix, axis=1) * (np.linalg.norm(q_emb)+1e-10))
    dense_ids = [str(i) for i in np.argsort(-sims)[:top_k*5]]
    fused = reciprocal_rank_fusion([dense_ids, bm25_ids], k=top_k)
    hits = []
    for fid in fused:
        idx = int(fid)
        hits.append({"id": fid, "text": all_chunks[idx]["text"], "path": all_chunks[idx]["path"],
                     "start_line": all_chunks[idx].get("start_line"), "end_line": all_chunks[idx].get("end_line")})
    return hits

# ------------- HF Generator -------------
def generate_answer_with_hf(query, hits, embedder, top_k=4):
    """
    Simple Hugging Face answer generator:
    - Combines retrieved snippets
    - Uses embeddings cosine similarity to rank relevance
    """
    context_texts = [h["text"] for h in hits]
    combined_context = "\n\n".join(context_texts)
    # Pick top sentences from context based on similarity to query
    sentences = re.split(r'\n+', combined_context)
    q_emb = embedder.encode(query, convert_to_numpy=True)
    sent_embs = embedder.encode(sentences, convert_to_numpy=True)
    sims = (sent_embs @ q_emb) / (np.linalg.norm(sent_embs, axis=1) * (np.linalg.norm(q_emb)+1e-10))
    top_idx = np.argsort(-sims)[:5]
    answer = "\n".join([sentences[i] for i in top_idx])
    return answer + "\n\n*Generated by Hugging Face Sentence Transformer*"

# ------------- Streamlit UI -------------
st.sidebar.title("Ingest / Settings")
source_type = st.sidebar.selectbox("Source", ["GitHub URL", "Local folder", "Upload zip"])
repo_input = st.sidebar.text_input("GitHub URL or local path", "")
uploaded_zip = st.sidebar.file_uploader("Upload repo .zip", type=["zip"])
max_files = st.sidebar.number_input("Max files to ingest", min_value=10, max_value=5000, value=500)
embedding_model_name = st.sidebar.text_input("Embedding model", value="all-MiniLM-L6-v2")
top_k = st.sidebar.slider("Top-k to retrieve", 1, 8, 4)

if st.sidebar.button("Ingest repository"):
    tmpdir = tempfile.mkdtemp()
    try:
        if source_type == "Upload zip" and uploaded_zip:
            zpath = Path(tmpdir) / "uploaded.zip"
            with open(zpath, "wb") as f:
                f.write(uploaded_zip.getbuffer())
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(tmpdir)
            files, extracted_root = read_supported_files(Path(tmpdir)), Path(tmpdir)
        elif source_type == "GitHub URL" and repo_input:
            st.info("Downloading GitHub repository...")
            files, extracted_root = ingest_from_github(repo_input, tmpdir, max_files=max_files)
        elif source_type == "Local folder" and repo_input:
            files, extracted_root = ingest_from_local(repo_input, max_files=max_files)
        else:
            st.error("Select a source and provide input")
            shutil.rmtree(tmpdir, ignore_errors=True)
            st.stop()

        if not files:
            st.error("No supported files found.")
            shutil.rmtree(tmpdir, ignore_errors=True)
            st.stop()

        st.success(f"Loaded {len(files)} files.")
        with st.spinner("Building embeddings and indices..."):
            all_chunks, emb_matrix, bm25, embedder = build_indices(files, embedding_model_name)
        st.session_state.update({"all_chunks": all_chunks, "emb_matrix": emb_matrix,
                                 "bm25": bm25, "embedder": embedder, "tmpdir": tmpdir})
        st.success("Ingest complete!")
    except Exception as e:
        st.error(f"Ingest failed: {e}")
        shutil.rmtree(tmpdir, ignore_errors=True)

# Query
st.header("Ask the codebase")
query = st.text_input("Question about code/docs", value="")
if st.button("Retrieve & Answer") and query.strip():
    if "all_chunks" not in st.session_state:
        st.warning("Ingest a repo first.")
    else:
        all_chunks = st.session_state["all_chunks"]
        emb_matrix = st.session_state["emb_matrix"]
        bm25 = st.session_state["bm25"]
        embedder = st.session_state["embedder"]
        with st.spinner("Retrieving..."):
            hits = retrieve(query, all_chunks, emb_matrix, bm25, embedder, top_k=top_k)
        st.subheader("Retrieved snippets")
        for i,h in enumerate(hits, start=1):
            st.markdown(f"**{i}. {Path(h['path']).name}** — {h['path']} ({h['start_line']}-{h['end_line']})")
            lang = Path(h['path']).suffix.replace(".", "")
            st.code(h['text'], language=lang if lang in ["py","js","ts","java","cpp","c","txt","md"] else "")
        with st.spinner("Generating answer..."):
            ans = generate_answer_with_hf(query, hits, embedder, top_k=top_k)
        st.subheader("Answer")
        st.markdown(ans)

# Cleanup
if st.session_state.get("tmpdir") and st.sidebar.button("Clear cache / temp"):
    shutil.rmtree(st.session_state["tmpdir"], ignore_errors=True)
    for k in ["all_chunks","emb_matrix","bm25","embedder","tmpdir"]:
        st.session_state.pop(k, None)
    st.success("Cleared temp files and session indices.")
