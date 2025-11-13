"""
M2 Research Assistant
Author  : Md Mahbubur Rahman
License : Apache 2.0  
GitHub  : https://github.com/m-a-h-b-u-b/m2-research-assistant
URL     : https://m-a-h-b-u-b.github.io 
"""
"""Unit tests for the ingestion module."""
import os
import tempfile
import fitz
import pytest
from ingestion import extract_text_from_pdf, process_pdf_and_upsert




def create_sample_pdf(tmp_path):
pdf_path = tmp_path / "sample.pdf"
doc = fitz.open()
page = doc.new_page()
page.insert_text((72, 72), "This is a test PDF for ingestion.")
doc.save(pdf_path)
doc.close()
return str(pdf_path)




def test_extract_text_from_pdf(tmp_path):
pdf_path = create_sample_pdf(tmp_path)
pages = extract_text_from_pdf(pdf_path)
assert isinstance(pages, list)
assert len(pages) == 1
assert "test PDF" in pages[0]['text']




def test_process_pdf_and_upsert(tmp_path):
pdf_path = create_sample_pdf(tmp_path)
doc_id = process_pdf_and_upsert(pdf_path)
assert isinstance(doc_id, str)
assert doc_id.endswith('.pdf')