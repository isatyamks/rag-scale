from __future__ import annotations

from datetime import datetime
from typing import Optional

from .base import BaseHealthCheck, HealthCheckResult


class InfrastructureChecks(BaseHealthCheck):
    
    @property
    def category_name(self) -> str:
        return "Infrastructure Checks"
    
    def run_checks(self) -> list[HealthCheckResult]:
        self.results = []
        self.check_ollama_service()
        self.check_network_connectivity()
        return self.results
    
    def check_ollama_service(self) -> HealthCheckResult:
        try:
            import requests
            
            start = datetime.now()
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            duration = (datetime.now() - start).total_seconds()
            
            if response.status_code == 200:
                return self.add_result(
                    "Ollama Service",
                    True,
                    "Running",
                    duration,
                    {"endpoint": "http://localhost:11434"}
                )
            else:
                return self.add_result(
                    "Ollama Service",
                    False,
                    f"Status: {response.status_code}",
                    duration
                )
        except Exception as e:
            error_msg = "Not running" if "ConnectionError" in str(type(e)) else f"Error: {str(e)[:30]}"
            return self.add_result("Ollama Service", False, error_msg)
    
    def check_network_connectivity(self) -> HealthCheckResult:
        try:
            import requests
            
            start = datetime.now()
            response = requests.get("https://www.google.com", timeout=5)
            duration = (datetime.now() - start).total_seconds()
            
            if response.status_code == 200:
                return self.add_result(
                    "Network Connectivity",
                    True,
                    "Online",
                    duration
                )
            else:
                return self.add_result(
                    "Network Connectivity",
                    False,
                    f"Status: {response.status_code}",
                    duration
                )
        except Exception as e:
            return self.add_result("Network Connectivity", False, "Offline")
