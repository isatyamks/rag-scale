"""Small RAG helpers: retrieval and generation wrappers.

These functions are intentionally thin: they adapt a vector store and LLM
into a predictable retrieve/generate interface used by the CLI.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List
from langchain_core.documents import Document


def retrieve(state: Dict[str, Any], vector_store) -> Dict[str, List[Document]]:
    try:
        retrieved_docs = vector_store.similarity_search(state.get("question", ""), k=5)
        for i, doc in enumerate(retrieved_docs):
            if not hasattr(doc, "id") or doc.id is None:
                doc.id = i
        logging.info(f"Retrieved {len(retrieved_docs)} relevant chunks.")
        return {"context": retrieved_docs}
    except Exception as e:
        logging.error(f"Error in retrieval: {e}")
        return {"context": []}


def generate(state: Dict[str, Any], llm, prompt) -> Dict[str, Any]:
    try:
        context_text = "\n\n".join(doc.page_content for doc in state.get("context", []))
        messages = prompt.invoke(
            {"question": state.get("question", ""), "context": context_text}
        )
        response = llm.invoke(messages)
        return {"answer": response.content}
    except Exception as e:
        logging.error(f"Error in generation: {e}")
        return {"answer": "Failed to generate response."}


def retriever_data(query: str, retriever) -> None:
    retrieved_docs = retriever.invoke(query)
    print(f"\n\nNumber of retrieved documents: {len(retrieved_docs)}")
    print("\nRetrieved documents:")
    for i, doc in enumerate(retrieved_docs):
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content[:200]}...")
        if hasattr(doc, "metadata") and doc.metadata:
            print(f"Metadata: {doc.metadata}\n\n")


__all__ = ["retrieve", "generate", "retriever_data"]
