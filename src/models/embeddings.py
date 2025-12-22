from __future__ import annotations

from langchain_ollama import OllamaEmbeddings


def create_embeddings(model: str = "nomic-embed-text", num_gpu: int = 999) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=model,
        num_gpu=num_gpu
    )
