

from pathlib import Path
import logging
import pickle
import faiss

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from ..storage.faiss.indexing import load_all_existing_indices
from ..storage.faiss.persistence import load_session_index, save_vector_store


__all__ = ["VectorManager"]


class VectorManager:

    def __init__(self, embeddings, base_session_dir: Path, embedding_dim: int):
        self.embeddings = embeddings
        self.BASE_SESSION_DIR = Path(base_session_dir)
        self.embedding_dim = embedding_dim

    def load_all_existing_indices(self):
        return load_all_existing_indices(self.BASE_SESSION_DIR, self.embedding_dim)

    def list_all_sessions(self):
        session_dirs = [
            d
            for d in self.BASE_SESSION_DIR.iterdir()
            if d.is_dir() and d.name.startswith("session_")
        ]
        if session_dirs:
            logging.info(
                "Available sessions: %s",
                [d.name for d in sorted(session_dirs)],
            )
            for session_dir in sorted(session_dirs):
                data_dir = session_dir / "data"
                faiss_dir = session_dir / "faiss_index"
                raw_files = (
                    list((data_dir / "raw").glob("*.txt"))
                    if (data_dir / "raw").exists()
                    else []
                )
                chunks_file = (
                    (data_dir / "processed" / "chunks.pkl")
                    if (data_dir / "processed").exists()
                    else None
                )
                faiss_exists = (
                    (faiss_dir / "faiss.index").exists()
                    if faiss_dir.exists()
                    else False
                )

                logging.info(f"  {session_dir.name}:")
                logging.info("    - Raw files: %d files", len(raw_files))
                chunks_status = "Yes" if chunks_file and chunks_file.exists() else "No"
                logging.info("    - Chunks: %s", chunks_status)
                logging.info("    - FAISS index: %s", "Yes" if faiss_exists else "No")
        return sorted(session_dirs)

    def load_chunks_from_session(self, session_name: str):
        session_dir = self.BASE_SESSION_DIR / session_name
        chunks_file = session_dir / "data" / "processed" / "chunks.pkl"

        if chunks_file.exists():
            with open(chunks_file, "rb") as f:
                chunks = pickle.load(f)
            logging.info(f"Loaded {len(chunks)} chunks from {session_name}")
            return chunks
        else:
            logging.warning(f"No chunks found in session {session_name}")
            return []

    def create_session_only_vector_store(self, session_manager):
        res = load_session_index(session_manager.CURRENT_SESSION_DIR)
        if res is not None:
            session_index, session_docstore, session_mapping = res
            vector_store = FAISS(
                embedding_function=self.embeddings,
                index=session_index,
                docstore=session_docstore,
                index_to_docstore_id=session_mapping,
            )
            return vector_store

        logging.info("Creating new empty HNSW vector store for current session")
        empty_index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        empty_docstore = InMemoryDocstore()
        empty_mapping = {}

        vector_store = FAISS(
            embedding_function=self.embeddings,
            index=empty_index,
            docstore=empty_docstore,
            index_to_docstore_id=empty_mapping,
        )

        return vector_store

    def create_global_vector_store(self, session_manager):
        combined_index, combined_docstore, combined_mapping = (
            self.load_all_existing_indices()
        )

        vector_store = FAISS(
            embedding_function=self.embeddings,
            index=combined_index,
            docstore=combined_docstore,
            index_to_docstore_id=combined_mapping,
        )

        return vector_store

    def add_to_vector_store(self, vector_store, chunks, session_manager):
        if not chunks:
            return

        session_manager.CURRENT_VECTOR_DIR.mkdir(exist_ok=True)
        session_manager.CURRENT_DATA_DIR.mkdir(exist_ok=True)

        logging.info(
            "Adding chunks to vector store in session: %s",
            session_manager.CURRENT_VECTOR_DIR.name,
        )
        
        batch_size = 32
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} (Chunks {i+1}-{min(i+batch_size, total_chunks)})")
            vector_store.add_documents(batch)
            logging.info(f"Batch {i//batch_size + 1} added successfully.")

        save_vector_store(session_manager, vector_store)
