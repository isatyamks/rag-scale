"""Entry point for the RAG CLI.

This module wires together the session, vector, and document utilities and
exposes a small interactive loop for querying indexed documents.
"""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain import hub

# local modules (refactored)
from src.session_manager import SessionManager
from src.vector_manager import VectorManager
from src.doc_processor import DocumentProcessor
from src import rag_graph


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Models
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3")


# -----------------------
# Base paths (session paths will be created later)
# -----------------------
BASE_SESSION_DIR = Path("sessions")
BASE_SESSION_DIR.mkdir(exist_ok=True)

# Global variables that will be set after user input
CURRENT_SESSION_DIR = None
CURRENT_VECTOR_DIR = None
CURRENT_DATA_DIR = None
CURRENT_LOGS_DIR = None
CSV_LOG_FILE = None
GLOBAL_CSV_LOG_FILE = BASE_SESSION_DIR / "interactions.csv"
session_id = None

embedding_dim = len(embeddings.embed_query("hello world"))

# Instantiate modular helpers
session_manager = SessionManager(BASE_SESSION_DIR)
vector_manager = VectorManager(embeddings, BASE_SESSION_DIR, embedding_dim)
doc_processor = DocumentProcessor()

# Prompt used for generation
prompt = hub.pull("rlm/rag-prompt")

vector_store = None


# -----------------------
# Main loop
# -----------------------
if __name__ == "__main__":
    temp1 = input("Enter the Newer Corpus Document: ")

    # Create session directories based on corpus name
    is_existing_session = session_manager.create_session_directories(temp1)

    # Always use session-only mode
    print("Initializing session-only vector store...")
    vector_store = vector_manager.create_session_only_vector_store(session_manager)
    search_mode = "Session-only"

    # Create retriever after vector store is initialized
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )

    # Note: use simple retrieve -> generate flow (no StateGraph) so runtime deps are explicit

    session_type = "Existing" if is_existing_session else "New"
    print(f"=== {session_type} RAG Session ({search_mode} Mode) ===")
    print(f"Session ID: {session_manager.session_id}")
    print(f"Session Directory: {session_manager.CURRENT_SESSION_DIR}")
    print(f"CSV Logs: {session_manager.CSV_LOG_FILE}")
    print(f"Search Mode: {search_mode}")

    print("You can search only within this session's documents")
    print()

    if not is_existing_session:
        txt_files = [f"data/raw/{temp1}.txt"]
        chunks = doc_processor.load_and_chunk(txt_files)
        vector_manager.add_to_vector_store(vector_store, chunks, session_manager)
        doc_processor.save_raw_and_chunks(txt_files, chunks, session_manager)

    # Initialize CSV logging (for both new and existing sessions)
    session_manager.initialize_csv_logs()

    print("Available commands:")
    print("  - Type your question to search")
    print("  - Type 'exit' to quit")

    while True:
        temp = input("Enter your question (or 'exit' to quit):\n")

        if temp.lower() == "exit":
            print("\nExiting RAG session...")
            session_manager.export_session_summary()
            print(f"\nSession completed. Files saved to:")
            print(f"Session CSV: {session_manager.CSV_LOG_FILE}")
            print(f"Global CSV: {GLOBAL_CSV_LOG_FILE}")
            print(
                f"Session Summary: {session_manager.CURRENT_LOGS_DIR / 'session_summary.txt'}"
            )
            print(f"Session Directory: {session_manager.CURRENT_SESSION_DIR}")
            break
        # session-only mode: no switching supported
        start_time = datetime.now()
        query = {"question": temp}

        retriever_start = datetime.now()
        try:
            retrieved_docs = retriever.invoke(temp)
        except Exception:
            # fallback: use vector_store directly
            retrieved_docs = vector_store.similarity_search(temp, k=6)

        retrieved_docs_count = len(retrieved_docs)

        # Build state and generate answer
        state = {"question": temp, "context": retrieved_docs}
        gen = rag_graph.generate(state, llm, prompt)
        answer = gen.get("answer", "")

        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()

        print(f"\nAnswer: {answer}\n")
        print(
            f"Retrieved {retrieved_docs_count} documents | {response_time:.2f}s | {search_mode} mode"
        )

        # Log to CSV
        session_manager.log_interaction_to_csv(
            temp, answer, retrieved_docs_count, response_time
        )
        print()
