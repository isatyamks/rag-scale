

from datetime import datetime
from typing import List

from .base import BaseHealthCheck
from .infrastructure import InfrastructureChecks
from .models import ModelChecks
from .components import ComponentChecks
from .external import ExternalServiceChecks


class HealthCheckRunner:
    
    def __init__(self, settings):
        self.settings = settings
        self.check_modules: List[BaseHealthCheck] = []
        self.all_results = []
        
    def register_check_module(self, module: BaseHealthCheck):
        self.check_modules.append(module)
    
    def run_all_checks(self, verbose: bool = True) -> bool:
        if verbose:
            self._print_header("RAG SYSTEM HEALTH CHECK")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        infra_checks = InfrastructureChecks(self.settings)
        model_checks = ModelChecks(self.settings)
        component_checks = ComponentChecks(self.settings)
        external_checks = ExternalServiceChecks(self.settings)
        
        check_sequence = [
            (1, 4, infra_checks),
            (2, 4, model_checks),
            (3, 4, component_checks),
            (4, 4, external_checks),
        ]
        
        ollama_running = False
        embedding_ok = False
        llm_ok = False
        network_ok = False
        
        for step, total, check_module in check_sequence:
            if verbose:
                print(f"\n[{step}/{total}] {check_module.category_name}")
                print("-" * 60)
            
            if isinstance(check_module, InfrastructureChecks):
                results = check_module.run_checks()
                if verbose:
                    check_module.print_results()
                
                ollama_running = any(r.passed and "Ollama" in r.name for r in results)
                network_ok = any(r.passed and "Network" in r.name for r in results)
                
            elif isinstance(check_module, ModelChecks):
                if not ollama_running:
                    if verbose:
                        print("WARNING: Skipping model checks - Ollama not running")
                    continue
                
                results = check_module.run_checks()
                if verbose:
                    check_module.print_results()
                
                embedding_ok = any(r.passed and "Embedding" in r.name for r in results)
                llm_ok = any(r.passed and "LLM" in r.name for r in results)
                
            elif isinstance(check_module, ComponentChecks):
                if not ollama_running:
                    if verbose:
                        print("WARNING: Skipping component checks - Ollama not running")
                    continue
                
                if not embedding_ok:
                    if verbose:
                        print("WARNING: Skipping embedding tests - Model not available")
                
                if not llm_ok:
                    if verbose:
                        print("WARNING: Skipping LLM tests - Model not available")
                
                results = check_module.run_checks()
                if verbose:
                    check_module.print_results()
                
            elif isinstance(check_module, ExternalServiceChecks):
                if not network_ok:
                    if verbose:
                        print("WARNING: Skipping external service checks - No network")
                    continue
                
                results = check_module.run_checks()
                if verbose:
                    check_module.print_results()
            
            self.all_results.extend(check_module.results)
        
        if verbose:
            self._print_summary()
        
        critical_checks = [
            ollama_running,
            embedding_ok,
            llm_ok,
        ]
        
        return all(critical_checks)
    
    def _print_header(self, title: str):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    
    def _print_summary(self):
        self._print_header("SUMMARY")
        
        passed = sum(1 for r in self.all_results if r.passed)
        total = len(self.all_results)
        
        print(f"\nTests Passed: {passed}/{total}")
        
        if passed == total:
            print("\nAll checks passed. System is ready to run.")
            print("Execute: python main.py")
        else:
            print("\nSome checks failed. Please fix the issues above.")
            print("\nCommon fixes:")
            print("  - Ollama not running: Run 'ollama serve' in a terminal")
            print("  - Model not found: Run 'ollama pull <model-name>'")
            print("  - Network issues: Check your internet connection")
        
        print("\n" + "="*60 + "\n")
    
    def get_failed_checks(self) -> List[str]:
        return [r.name for r in self.all_results if not r.passed]
    
    def get_passed_checks(self) -> List[str]:
        return [r.name for r in self.all_results if r.passed]
