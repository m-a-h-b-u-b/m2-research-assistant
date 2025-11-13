"""
M2 Research Assistant
Author  : Md Mahbubur Rahman
License : Apache 2.0  
GitHub  : https://github.com/m-a-h-b-u-b/m2-research-assistant
URL     : https://m-a-h-b-u-b.github.io 
"""

"""Unit tests for retrieval module."""
import pytest
from retrieval import hybrid_search


def test_hybrid_search_returns_documents():
results = hybrid_search("test query", k=2)
assert isinstance(results, list)
# The list may be empty if no corpus loaded, but type should be correct
for doc in results:
assert hasattr(doc, 'page_content')