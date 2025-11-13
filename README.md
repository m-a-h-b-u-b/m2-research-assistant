"""
# M2 Research Assistant


An end-to-end Retrieval-Augmented Generation (RAG) system for summarizing and
answering questions from uploaded research PDFs.


## Features
- PDF ingestion and text extraction (PyMuPDF)
- Chunking with overlap using LangChain text splitter
- Dense embeddings (OpenAI or local sentence-transformer)
- Vector store using Chroma (local)
- Optional BM25 hybrid retrieval (rank_bm25)
- LLM generation using OpenAI `gpt-4o` / `gpt-4-turbo` (configurable)
- Streamlit frontend for upload + Q&A


## Setup
1. Create a virtual environment and install requirements:


```bash
python -m venv .venv
source .venv/bin/activate # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


2. Set environment variables (example):


```bash
export OPENAI_API_KEY="sk-..."
export CHROMA_DB_DIR="./data/chroma"
```


3. Run the Streamlit app:


```bash
streamlit run app.py
```


## Notes
- This project uses Chroma as the vector store. For production, consider a managed
vector DB (Pinecone, Weaviate) and stronger PDF cleaning pipelines.
