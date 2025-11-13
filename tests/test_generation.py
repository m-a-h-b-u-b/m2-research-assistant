"""
M2 Research Assistant
Author  : Md Mahbubur Rahman
License : Apache 2.0  
GitHub  : https://github.com/m-a-h-b-u-b/m2-research-assistant
URL     : https://m-a-h-b-u-b.github.io 
"""
import pytest
from generation import generate_answer


def test_generate_answer(monkeypatch):
def mock_run(inputs):
return f"Mock answer for: {inputs['question']}"
class MockChain:
def run(self, inputs):
return mock_run(inputs)
monkeypatch.setattr("generation.LLMChain", lambda prompt, llm: MockChain())
answer = generate_answer("context text", "What is AI?")
assert "Mock answer" in answer