"""
M2 Research Assistant
Author  : Md Mahbubur Rahman
License : Apache 2.0  
GitHub  : https://github.com/m-a-h-b-u-b/m2-research-assistant
URL     : https://m-a-h-b-u-b.github.io 
"""

"""
PDF ingestion pipeline:
- Extract text with PyMuPDF
- Clean text minimally
- Chunk using LangChain's text splitters
- Create embeddings and upsert into Chroma
"""
import os
import uuid
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import chromadb
from chromadb.config import Settings
from langchain.schema import Document

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./data/chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai")

# Initialize chroma client (directory persist)
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DB_DIR))
collection = chroma_client.get_or_create_collection(name="papers", metadata={"hnsw:space": "cosine"})


def extract_text_from_pdf(path):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text and text.strip():
            pages.append({"page": i, "text": text})
    return pages


def clean_text(s: str) -> str:
    # Minimal cleaning — extend as needed
    return "\n".join([line.strip() for line in s.splitlines() if line.strip()])


def process_pdf_and_upsert(path, namespace=None):
    pages = extract_text_from_pdf(path)
    full_text = "\n\n".join([p['text'] for p in pages])

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_text(full_text)

    # Map each chunk back to approx page numbers — naive approach
    docs = []
    for i, chunk in enumerate(chunks):
        # metadata: naive page = i // (len(chunks)//max(1, len(pages))) + 1
        docs.append(Document(page_content=chunk, metadata={"source": os.path.basename(path), "chunk_id": str(uuid.uuid4())}))

    # Create embeddings
    emb = OpenAIEmbeddings()
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    ids = [m['chunk_id'] for m in metadatas]

    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    chroma_client.persist()

    return os.path.basename(path)
