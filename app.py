"""
M2 Research Assistant
Author  : Md Mahbubur Rahman
License : Apache 2.0  
GitHub  : https://github.com/m-a-h-b-u-b/m2-research-assistant
URL     : https://m-a-h-b-u-b.github.io 
"""

"""
Streamlit frontend for uploading PDFs, ingesting them, and asking questions.
"""
import os
import tempfile
import streamlit as st
from ingestion import process_pdf_and_upsert
from retrieval import hybrid_search
from generation import generate_answer

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("🧠 AI Research Assistant — RAG PDF Q&A")

st.sidebar.header("Settings")
model = st.sidebar.selectbox("LLM Model", [os.getenv("LLM_MODEL", "gpt-4o"), "gpt-4-turbo", "gpt-3.5-turbo"]) 
max_chunks = st.sidebar.slider("Max context chunks", 1, 10, 5)

uploaded = st.file_uploader("Upload research paper (PDF)", type=["pdf"], accept_multiple_files=False)
if uploaded is not None:
    st.info("Processing uploaded PDF — this may take a minute for the first run.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    # Ingest and upsert into Chroma; returns metadata about document
    doc_id = process_pdf_and_upsert(tmp_path, namespace=None)
    st.success(f"Ingested as doc_id: {doc_id}")

query = st.text_input("Ask a question about the uploaded paper:")
if st.button("Ask") and query:
    with st.spinner("Retrieving context..."):
        docs = hybrid_search(query, k=max_chunks)
    if not docs:
        st.warning("No relevant context found.")
    else:
        context = "\n\n".join([f"[Chunk {i+1}] {d.metadata.get('source','unknown')} (pg:{d.metadata.get('page', '?')})\n\n" + d.page_content for i, d in enumerate(docs)])
        st.subheader("Retrieved context (top chunks)")
        for i, d in enumerate(docs):
            st.markdown(f"**Chunk {i+1} — source:** {d.metadata.get('source','unknown')} — page {d.metadata.get('page','?')}")
            st.write(d.page_content[:1000] + ("..." if len(d.page_content) > 1000 else ""))
        st.subheader("Answer")
        answer = generate_answer(context, query, model=model)
        st.markdown(answer)
