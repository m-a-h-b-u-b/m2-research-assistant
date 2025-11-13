"""
M2 Research Assistant
Author  : Md Mahbubur Rahman
License : Apache 2.0  
GitHub  : https://github.com/m-a-h-b-u-b/m2-research-assistant
URL     : https://m-a-h-b-u-b.github.io 
"""

"""
Generation layer: receives retrieved context and produces a final answer with citations.
"""
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_PROMPT = """
You are an AI research assistant. Use only the information contained in the provided
context to answer the user's question. When you use content from the context,
cite it inline using the chunk metadata tag in parentheses, e.g. (Chunk #2, source: paper.pdf, pg: 3).
If the answer is not contained in the context, say "I don't know based on the provided documents.".

Context:
{context}

Question: {question}

Answer (be concise, factual, and provide citations):
"""


def generate_answer(context: str, question: str, model: str = None) -> str:
    if model is None:
        model = os.getenv("LLM_MODEL", "gpt-4o")
    llm = ChatOpenAI(model=model, temperature=0.0)
    prompt = PromptTemplate(input_variables=["context","question"], template=_PROMPT)
    chain = LLMChain(prompt=prompt, llm=llm)
    out = chain.run({"context": context, "question": question})
    return out

# ----------------------------
# File: utils.py
# ----------------------------
"""
Utility helpers: token counting, basic metadata helpers, etc.
"""
import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))

