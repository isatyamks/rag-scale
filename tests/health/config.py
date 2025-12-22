from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class HealthCheckConfig:
    
    OLLAMA_ENDPOINT: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: int = 5
    
    NETWORK_TEST_URL: str = "https://www.google.com"
    NETWORK_TIMEOUT: int = 5
    
    WIKI_TEST_PAGE: str = "Python_(programming_language)"
    WIKI_TIMEOUT: int = 10
    
    TEST_EMBEDDING_TEXT: str = "This is a test sentence for embedding."
    TEST_LLM_PROMPT: str = "Say 'OK' if you can respond."
    TEST_LLM_MAX_TOKENS: int = 50
    
    TEST_DOC_CHUNK_SIZE: int = 100
    TEST_DOC_CHUNK_OVERLAP: int = 20
    TEST_DOC_REPEAT: int = 50
    
    TEMP_SESSION_NAME: str = "health_check_test"
    TEMP_FILE_NAME: str = "test_health.txt"
    
    CRITICAL_CHECKS: list[str] = None
    
    def __post_init__(self):
        if self.CRITICAL_CHECKS is None:
            self.CRITICAL_CHECKS = [
                "Ollama Service",
                "Embedding Model",
                "LLM Model",
                "Embedding Generation",
                "LLM Generation",
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ollama_endpoint": self.OLLAMA_ENDPOINT,
            "ollama_timeout": self.OLLAMA_TIMEOUT,
            "network_test_url": self.NETWORK_TEST_URL,
            "network_timeout": self.NETWORK_TIMEOUT,
            "wiki_test_page": self.WIKI_TEST_PAGE,
            "wiki_timeout": self.WIKI_TIMEOUT,
            "critical_checks": self.CRITICAL_CHECKS,
        }


_config = None


def get_health_config() -> HealthCheckConfig:
    global _config
    if _config is None:
        _config = HealthCheckConfig()
    return _config
