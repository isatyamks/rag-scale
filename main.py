from __future__ import annotations

from pathlib import Path
from datetime import datetime

from src.config import get_settings
from src.utils import setup_logging
from src.models import create_embeddings, create_llm
from src.core import SessionManager, VectorManager
from src.data import DocumentProcessor, WikiLoader
from src.rag import generate


settings = get_settings()
setup_logging(settings.LOG_LEVEL, settings.LOG_FORMAT)

embeddings = create_embeddings(settings.EMBEDDING_MODEL, settings.NUM_GPU)
llm = create_llm(
    settings.LLM_MODEL,
    settings.NUM_GPU,
    settings.LLM_TEMPERATURE,
    settings.LLM_NUM_CTX,
    settings.LLM_NUM_PREDICT
)

embedding_dim = len(embeddings.embed_query("hello world"))

session_manager = SessionManager(settings.BASE_SESSION_DIR)
vector_manager = VectorManager(embeddings, settings.BASE_SESSION_DIR, embedding_dim)
doc_processor = DocumentProcessor()

vector_store = None


if __name__ == "__main__":
    wiki_loader = WikiLoader(settings.WIKI_LANGUAGE)
    
    print("--- Advanced RAG System ---")
    corpus_input = input("Enter a Topic (Wikipedia title) or existing Corpus Name: ").strip()

    t_start_total = datetime.now()
    
    print("\n[Timing] Starting Data Discovery/Fetch...")
    t0 = datetime.now()

    raw_path = settings.RAW_DATA_DIR / f"{corpus_input}.txt"
    
    if raw_path.exists():
        print(f"Found existing local data: {raw_path}")
        corpus_name = corpus_input
    else:
        print(f"Local file not found for '{corpus_input}'. Checking Wikipedia...")
        fetched_path = wiki_loader.fetch_page(corpus_input, settings.RAW_DATA_DIR)
        
        if fetched_path:
            print(f"Successfully fetched '{corpus_input}' from Wikipedia.")
            corpus_name = corpus_input
        else:
            print(f"Could not find data for '{corpus_input}' locally or on Wikipedia.")
            print("Please ensure the Wikipedia title is correct or the file exists in data/raw/")
            exit(1)
    
    dt_fetch = (datetime.now() - t0).total_seconds()
    print(f"[Timing] Data Discovery/Fetch took {dt_fetch:.2f}s")

    is_existing_session = session_manager.create_session_directories(corpus_name)
    
    print("\nInitializing Vector Store...")
    t1 = datetime.now()
    vector_store = vector_manager.create_session_only_vector_store(session_manager)
    dt_init_vs = (datetime.now() - t1).total_seconds()
    print(f"[Timing] Vector Store Initialization took {dt_init_vs:.2f}s")
    
    retriever = vector_store.as_retriever(
        search_type=settings.SEARCH_TYPE,
        search_kwargs={"k": settings.RETRIEVAL_K}
    )

    print(f"Session ID: {session_manager.session_id}")
    print(f"Session Directory: {session_manager.CURRENT_SESSION_DIR}")

    if not is_existing_session:
        print("\n[Timing] Starting Chunking & Embedding...")
        t2 = datetime.now()
        
        txt_files = [str(settings.RAW_DATA_DIR / f"{corpus_name}.txt")]
        chunks = doc_processor.load_and_chunk(
            txt_files,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        t2_chunk = datetime.now()
        print(f"Chunking took {(t2_chunk - t2).total_seconds():.2f}s")
        
        vector_manager.add_to_vector_store(vector_store, chunks, session_manager)
        
        t2_embed = datetime.now()
        print(f"Embedding & Storage took {(t2_embed - t2_chunk).total_seconds():.2f}s")
        
        doc_processor.save_raw_and_chunks(txt_files, chunks, session_manager)
        
        dt_process = (datetime.now() - t2).total_seconds()
        print(f"[Timing] Total Processing (Chunk + Embed) took {dt_process:.2f}s")
    
    session_manager.initialize_csv_logs()

    print("\nSystem Ready! Ask questions based on the content.")
    print("  - Type 'exit' to quit")

    while True:
        questions = input("\nQuestion: ").strip()

        if questions.lower() in ["exit", "quit"]:
            print("\nExiting...")
            session_manager.export_session_summary()
            break
        
        if not questions:
            continue

        start_time = datetime.now()
        
        try:
            retrieved_docs = retriever.invoke(questions)
        except Exception as e:
            import logging
            logging.error(f"Retrieval failed: {e}")
            retrieved_docs = []

        retrieved_docs_count = len(retrieved_docs)

        state = {"question": questions, "context": retrieved_docs}
        
        gen_result = generate(state, llm, None) 
        answer = gen_result.get("answer", "")

        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()

        print(f"\n>> Answer: {answer}\n")
        print(f"[Meta: {retrieved_docs_count} docs | {response_time:.2f}s]")

        session_manager.log_interaction_to_csv(
            questions, answer, retrieved_docs_count, response_time
        )

