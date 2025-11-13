"""
M2 Research Assistant
Author  : Md Mahbubur Rahman
License : Apache 2.0  
GitHub  : https://github.com/m-a-h-b-u-b/m2-research-assistant
URL     : https://m-a-h-b-u-b.github.io 
"""

"""
Hybrid retrieval combining dense vector similarity with BM25.
"""
import os
from typing import List
from rank_bm25 import BM25Okapi
from langchain.embeddings import OpenAIEmbeddings
import chromadb
from chromadb.config import Settings
from langchain.schema import Document

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./data/chroma")
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DB_DIR))
collection = chroma_client.get_collection("papers")

# Build BM25 corpus cache (in-memory, rebuilt lazily)
_bm25 = None
_corpus_texts = None


def _build_bm25():
    global _bm25, _corpus_texts
    docs = collection.get(include=['documents','metadatas'])
    texts = docs['documents']
    tokenized = [t.split() for t in texts]
    _corpus_texts = texts
    _bm25 = BM25Okapi(tokenized)


def hybrid_search(query: str, k: int = 5) -> List[Document]:
    # Dense search via chroma
    dense = collection.query(query_texts=[query], n_results=k, include=['documents','metadatas','distances'])
    dense_docs = []
    for docs_row, metas_row in zip(dense['documents'], dense['metadatas']):
        for d, m in zip(docs_row, metas_row):
            dense_docs.append(Document(page_content=d, metadata=m))

    # BM25
    global _bm25
    if _bm25 is None:
        _build_bm25()
    topn = _bm25.get_top_n(query.split(), _corpus_texts, n=k)
    bm25_docs = [Document(page_content=t, metadata={}) for t in topn]

    # Merge: naive merge preferring dense then bm25 uniques
    merged = dense_docs[:]
    existing_texts = set(d.page_content for d in merged)
    for d in bm25_docs:
        if d.page_content not in existing_texts:
            merged.append(d)
    return merged[:k]
