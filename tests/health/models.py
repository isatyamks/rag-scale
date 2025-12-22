

from typing import Tuple

from .base import BaseHealthCheck, HealthCheckResult


class ModelChecks(BaseHealthCheck):
    
    @property
    def category_name(self) -> str:
        return "Model Availability"
    
    def run_checks(self) -> list[HealthCheckResult]:
        self.results = []
        self.check_ollama_models()
        return self.results
    
    def check_ollama_models(self) -> Tuple[HealthCheckResult, HealthCheckResult]:
        try:
            import requests
            
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            
            if response.status_code != 200:
                r1 = self.add_result("Embedding Model", False, "Cannot check Ollama")
                r2 = self.add_result("LLM Model", False, "Cannot check Ollama")
                return r1, r2
            
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            embedding_exists = any(
                self.settings.EMBEDDING_MODEL in name for name in model_names
            )
            llm_exists = any(
                self.settings.LLM_MODEL in name for name in model_names
            )
            
            r1 = self.add_result(
                f"Embedding Model ({self.settings.EMBEDDING_MODEL})",
                embedding_exists,
                "Found" if embedding_exists else "Not found - Run: ollama pull " + self.settings.EMBEDDING_MODEL,
                metadata={"model": self.settings.EMBEDDING_MODEL}
            )
            
            r2 = self.add_result(
                f"LLM Model ({self.settings.LLM_MODEL})",
                llm_exists,
                "Found" if llm_exists else "Not found - Run: ollama pull " + self.settings.LLM_MODEL,
                metadata={"model": self.settings.LLM_MODEL}
            )
            
            return r1, r2
            
        except Exception as e:
            r1 = self.add_result("Embedding Model", False, f"Error: {str(e)[:30]}")
            r2 = self.add_result("LLM Model", False, f"Error: {str(e)[:30]}")
            return r1, r2
