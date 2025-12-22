from __future__ import annotations

from datetime import datetime

from .base import BaseHealthCheck, HealthCheckResult


class ExternalServiceChecks(BaseHealthCheck):
    
    @property
    def category_name(self) -> str:
        return "External Services"
    
    def run_checks(self) -> list[HealthCheckResult]:
        self.results = []
        self.check_wikipedia_access()
        return self.results
    
    def check_wikipedia_access(self) -> HealthCheckResult:
        try:
            import requests
            
            start = datetime.now()
            response = requests.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/Python_(programming_language)",
                timeout=10,
                headers={"User-Agent": self.settings.WIKI_USER_AGENT}
            )
            duration = (datetime.now() - start).total_seconds()
            
            if response.status_code == 200:
                return self.add_result(
                    "Wikipedia API",
                    True,
                    "Accessible",
                    duration,
                    {"endpoint": "en.wikipedia.org"}
                )
            else:
                return self.add_result(
                    "Wikipedia API",
                    False,
                    f"Status: {response.status_code}",
                    duration
                )
        except Exception as e:
            return self.add_result("Wikipedia API", False, f"Error: {str(e)[:30]}")
