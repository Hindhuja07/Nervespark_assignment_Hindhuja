from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
from utils import simple_chunker
import re

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def reciprocal_rank_fusion(list_of_ranked_ids, k=10):
    # returns fused list of ids by RRF score
    score = {}
    for rank_list in list_of_ranked_ids:
        for i, docid in enumerate(rank_list):
            score[docid] = score.get(docid, 0) + 1.0/(50 + i)
    sorted_ids = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return [sid for sid, sc in sorted_ids][:k]

class HybridRetriever:
    def __init__(self, chroma_dir, embedding_model="all-MiniLM-L6-v2"):
        self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=chroma_dir))
        self.collection = self.client.get_collection("code_docs")
        self.embedder = SentenceTransformer(embedding_model)
        # load BM25 on all documents
        docs = self.collection.get(include=["documents","metadatas"])
        self.documents = docs["documents"]
        self.metadatas = docs["metadatas"]
        self.bm25 = BM25Okapi([tokenize(d) for d in self.documents])

    def retrieve(self, query, top_k=5):
        # BM25 top ids
        qtok = tokenize(query)
        bm25_scores = self.bm25.get_top_n(qtok, self.documents, n=top_k*5)
        # dense retrieval
        q_emb = self.embedder.encode(query)
        dens = self.collection.query(query_embeddings=[q_emb.tolist()], n_results=top_k*5, include=["documents","metadatas","ids"])
        dense_ids = [i for i in dens["ids"][0]]
        # map ids (strings index) — chroma ids are '0','1',...
        bm25_ids = []
        for d in bm25_scores:
            # find corresponding index
            try:
                idx = self.documents.index(d)
                bm25_ids.append(str(idx))
            except ValueError:
                continue
        # do RRF fusion
        ranklists = [dense_ids, bm25_ids]
        fused = reciprocal_rank_fusion(ranklists, k=top_k)
        hits = []
        for fid in fused:
            idx = int(fid)
            hits.append({
                "id": fid,
                "page_content": self.documents[idx],
                "metadata": self.metadatas[idx]
            })
        return hits
