from __future__ import annotations

from pathlib import Path
from typing import Optional


class Settings:
    
    BASE_SESSION_DIR: Path = Path("sessions")
    DATA_DIR: Path = Path("data")
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    
    EMBEDDING_MODEL: str = "nomic-embed-text"
    LLM_MODEL: str = "mistral:7b-instruct-q4_K_M"
    
    NUM_GPU: int = 999
    LLM_TEMPERATURE: float = 0
    LLM_NUM_CTX: int = 2048
    LLM_NUM_PREDICT: int = 150
    
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    RETRIEVAL_K: int = 5
    SEARCH_TYPE: str = "similarity"
    
    FAISS_HNSW_M: int = 32
    BATCH_SIZE: int = 32
    
    WIKI_LANGUAGE: str = "en"
    WIKI_USER_AGENT: str = "RAG-Scale/1.0 (https://github.com/isatyamks/RAG)"
    
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(message)s"
    
    def __init__(self):
        self.BASE_SESSION_DIR.mkdir(exist_ok=True)
        self.DATA_DIR.mkdir(exist_ok=True)
        self.RAW_DATA_DIR.mkdir(exist_ok=True)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
