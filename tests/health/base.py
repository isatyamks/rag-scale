from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime


class HealthCheckResult:
    
    def __init__(
        self, 
        name: str, 
        passed: bool, 
        message: str = "",
        duration: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration = duration
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def __repr__(self) -> str:
        status = "[PASS]" if self.passed else "[FAIL]"
        duration_str = f", {self.duration:.2f}s" if self.duration else ""
        return f"{status} {self.name:<40} {self.message}{duration_str}"


class BaseHealthCheck(ABC):
    
    def __init__(self, settings):
        self.settings = settings
        self.results = []
    
    @abstractmethod
    def run_checks(self) -> list[HealthCheckResult]:
        pass
    
    @property
    @abstractmethod
    def category_name(self) -> str:
        pass
    
    def add_result(
        self, 
        name: str, 
        passed: bool, 
        message: str = "",
        duration: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> HealthCheckResult:
        result = HealthCheckResult(name, passed, message, duration, metadata)
        self.results.append(result)
        return result
    
    def print_results(self):
        for result in self.results:
            print(result)
