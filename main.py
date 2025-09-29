from src.core.config import setup_environment
from src.models.llm_models import initialize_models
from src.vectorstore.vector_store import create_vector_store
from src.ingestion.document_loader import load_and_process_documents, save_documents
from src.core.workflow import create_workflow

def main():
    setup_environment()
    
    embeddings, llm = initialize_models()
    
    vector_store = create_vector_store(embeddings)
    
    docs, chunks = load_and_process_documents()
    
    save_documents(docs, chunks)
    
    vector_store.add_documents(documents=chunks)
    
    graph = create_workflow(vector_store, llm)
    
    response = graph.invoke({"question": "What is a Sapiens?"})
    print(response["answer"])

if __name__ == "__main__":
    main()

