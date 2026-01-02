from datetime import datetime

from .base import BaseHealthCheck, HealthCheckResult


class ComponentChecks(BaseHealthCheck):
    
    @property
    def category_name(self) -> str:
        return "Component Functionality"
    
    def run_checks(self) -> list[HealthCheckResult]:
        self.results = []
        self.check_embedding_functionality()
        self.check_llm_functionality()
        self.check_vector_store()
        self.check_document_processing()
        self.check_session_management()
        return self.results
    
    def check_embedding_functionality(self) -> HealthCheckResult:
        try:
            from src.models import create_embeddings
            
            embeddings = create_embeddings(
                self.settings.EMBEDDING_MODEL, 
                self.settings.NUM_GPU
            )
            test_text = "This is a test sentence for embedding."
            
            start = datetime.now()
            embedding_vector = embeddings.embed_query(test_text)
            duration = (datetime.now() - start).total_seconds()
            
            if len(embedding_vector) > 0:
                return self.add_result(
                    "Embedding Generation",
                    True,
                    f"Dim: {len(embedding_vector)}",
                    duration,
                    {"dimension": len(embedding_vector)}
                )
            else:
                return self.add_result("Embedding Generation", False, "Empty vector")
                
        except Exception as e:
            return self.add_result(
                "Embedding Generation", 
                False, 
                f"Error: {str(e)[:30]}"
            )
    
    def check_llm_functionality(self) -> HealthCheckResult:
        try:
            from src.models import create_llm
            
            llm = create_llm(
                self.settings.LLM_MODEL,
                self.settings.NUM_GPU,
                self.settings.LLM_TEMPERATURE,
                self.settings.LLM_NUM_CTX,
                50
            )
            
            start = datetime.now()
            response = llm.invoke("Say 'OK' if you can respond.")
            duration = (datetime.now() - start).total_seconds()
            
            if response and hasattr(response, 'content') and len(response.content) > 0:
                return self.add_result(
                    "LLM Generation",
                    True,
                    f"Response: '{response.content[:20]}...'",
                    duration,
                    {"response_length": len(response.content)}
                )
            else:
                return self.add_result("LLM Generation", False, "No response")
                
        except Exception as e:
            return self.add_result("LLM Generation", False, f"Error: {str(e)[:30]}")
    
    def check_vector_store(self) -> HealthCheckResult:
        try:
            from src.core import VectorManager
            from src.models import create_embeddings
            
            embeddings = create_embeddings(
                self.settings.EMBEDDING_MODEL, 
                self.settings.NUM_GPU
            )
            embedding_dim = len(embeddings.embed_query("test"))
            
            vector_manager = VectorManager(
                embeddings, 
                self.settings.BASE_SESSION_DIR, 
                embedding_dim
            )
            
            return self.add_result(
                "Vector Store (FAISS)",
                True,
                f"Initialized, dim={embedding_dim}",
                metadata={"dimension": embedding_dim}
            )
            
        except Exception as e:
            return self.add_result(
                "Vector Store (FAISS)", 
                False, 
                f"Error: {str(e)[:30]}"
            )
    
    def check_document_processing(self) -> HealthCheckResult:
        try:
            from src.data import DocumentProcessor
            
            doc_processor = DocumentProcessor()
            
            test_file = self.settings.RAW_DATA_DIR / "test_health.txt"
            test_file.write_text("This is a test document for health check. " * 50)
            
            chunks = doc_processor.load_and_chunk(
                [str(test_file)],
                chunk_size=100,
                chunk_overlap=20
            )
            
            test_file.unlink()
            
            if len(chunks) > 0:
                return self.add_result(
                    "Document Processing",
                    True,
                    f"{len(chunks)} chunks created",
                    metadata={"chunks": len(chunks)}
                )
            else:
                return self.add_result("Document Processing", False, "No chunks created")
                
        except Exception as e:
            return self.add_result(
                "Document Processing", 
                False, 
                f"Error: {str(e)[:30]}"
            )
    
    def check_session_management(self) -> HealthCheckResult:
        try:
            from src.core import SessionManager
            
            session_manager = SessionManager(self.settings.BASE_SESSION_DIR)
            session_manager.create_session_directories("health_check_test")
            
            if session_manager.CURRENT_SESSION_DIR.exists():
                import shutil
                shutil.rmtree(session_manager.CURRENT_SESSION_DIR)
                return self.add_result("Session Management", True, "Working")
            else:
                return self.add_result(
                    "Session Management", 
                    False, 
                    "Directory not created"
                )
                
        except Exception as e:
            return self.add_result(
                "Session Management", 
                False, 
                f"Error: {str(e)[:30]}"
            )
