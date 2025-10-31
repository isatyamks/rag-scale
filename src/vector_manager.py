"""Vector management helpers.

This module contains :class:`VectorManager` which is responsible for loading
and merging FAISS indices, creating per-session or combined vector stores, and
persisting index artifacts.
"""

from __future__ import annotations

from pathlib import Path
import logging
import pickle
from typing import Dict, List, Optional
import faiss

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


__all__ = ["VectorManager"]


class VectorManager:
    """Manage FAISS indices across sessions and create vector stores.

    The implementation intentionally keeps FAISS-specific code here so
    :mod:`main` and other modules don't need to import faiss directly.
    """

    def __init__(self, embeddings, base_session_dir: Path, embedding_dim: int):
        self.embeddings = embeddings
        self.BASE_SESSION_DIR = Path(base_session_dir)
        self.embedding_dim = embedding_dim

    def load_all_existing_indices(self):
        all_docs = []
        combined_docstore = InMemoryDocstore()
        combined_index = faiss.IndexFlatL2(self.embedding_dim)
        combined_mapping: Dict[int, str] = {}
        current_id_offset = 0

        session_dirs = [
            d
            for d in self.BASE_SESSION_DIR.iterdir()
            if d.is_dir() and d.name.startswith("session_")
        ]

        if not session_dirs:
            logging.info("No existing FAISS indices found.")
            return combined_index, combined_docstore, combined_mapping

        logging.info(f"Found {len(session_dirs)} existing FAISS session directories.")

        for session_dir in sorted(session_dirs):
            faiss_dir = session_dir / "faiss_index"
            index_file = faiss_dir / "faiss.index"
            docstore_file = faiss_dir / "docstore.pkl"
            mapping_file = faiss_dir / "index_mapping.pkl"

            if index_file.exists() and docstore_file.exists():
                logging.info(f"Loading FAISS index from {session_dir.name}...")

                session_index = faiss.read_index(str(index_file))
                with open(docstore_file, "rb") as f:
                    session_docstore = pickle.load(f)

                if mapping_file.exists():
                    with open(mapping_file, "rb") as f:
                        session_mapping = pickle.load(f)
                else:
                    session_mapping = {i: str(i) for i in range(session_index.ntotal)}

                if session_index.ntotal > 0:
                    vectors = session_index.reconstruct_n(0, session_index.ntotal)
                    combined_index.add(vectors)

                    for old_idx, doc_id in session_mapping.items():
                        try:
                            doc = session_docstore.search(doc_id)
                            new_doc_id = str(current_id_offset + old_idx)
                            combined_docstore.add({new_doc_id: doc})
                            combined_mapping[current_id_offset + old_idx] = new_doc_id
                        except KeyError:
                            logging.warning(f"Document {doc_id} not found in docstore")

                    current_id_offset += session_index.ntotal

        logging.info(f"Combined {len(combined_mapping)} documents from all sessions.")
        return combined_index, combined_docstore, combined_mapping

    def list_all_sessions(self):
        session_dirs = [
            d
            for d in self.BASE_SESSION_DIR.iterdir()
            if d.is_dir() and d.name.startswith("session_")
        ]
        if session_dirs:
            logging.info(
                f"Available sessions: {[d.name for d in sorted(session_dirs)]}"
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
                logging.info(f"    - Raw files: {len(raw_files)} files")
                logging.info(
                    f"    - Chunks: {'Yes' if chunks_file and chunks_file.exists() else 'No'}"
                )
                logging.info(f"    - FAISS index: {'Yes' if faiss_exists else 'No'}")
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
        session_manager.CURRENT_VECTOR_DIR.mkdir(exist_ok=True)
        session_manager.CURRENT_DATA_DIR.mkdir(exist_ok=True)

        current_index_file = session_manager.CURRENT_VECTOR_DIR / "faiss.index"
        current_docstore_file = session_manager.CURRENT_VECTOR_DIR / "docstore.pkl"
        current_mapping_file = session_manager.CURRENT_VECTOR_DIR / "index_mapping.pkl"

        if current_index_file.exists() and current_docstore_file.exists():
            logging.info(
                f"Loading existing FAISS index from current session: {session_manager.CURRENT_SESSION_DIR.name}"
            )
            session_index = faiss.read_index(str(current_index_file))
            with open(current_docstore_file, "rb") as f:
                session_docstore = pickle.load(f)

            if current_mapping_file.exists():
                with open(current_mapping_file, "rb") as f:
                    session_mapping = pickle.load(f)
            else:
                session_mapping = {i: str(i) for i in range(session_index.ntotal)}

            vector_store = FAISS(
                embedding_function=self.embeddings,
                index=session_index,
                docstore=session_docstore,
                index_to_docstore_id=session_mapping,
            )
        else:
            logging.info("Creating new empty vector store for current session")
            empty_index = faiss.IndexFlatL2(self.embedding_dim)
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
        # Try to load all existing indices first
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
            f"Adding chunks to vector store in session: {session_manager.CURRENT_VECTOR_DIR.name}"
        )
        vector_store.add_documents(chunks)

        current_index_file = session_manager.CURRENT_VECTOR_DIR / "faiss.index"
        current_docstore_file = session_manager.CURRENT_VECTOR_DIR / "docstore.pkl"
        current_mapping_file = session_manager.CURRENT_VECTOR_DIR / "index_mapping.pkl"

        faiss.write_index(vector_store.index, str(current_index_file))
        with open(current_docstore_file, "wb") as f:
            pickle.dump(vector_store.docstore, f)
        with open(current_mapping_file, "wb") as f:
            pickle.dump(vector_store.index_to_docstore_id, f)

        logging.info(
            f"FAISS index and docstore saved to {session_manager.CURRENT_VECTOR_DIR}"
        )
