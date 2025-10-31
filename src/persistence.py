"""Persistence helpers for FAISS indices and docstores.

This module centralizes loading and saving of per-session FAISS artifacts so
that other modules can remain small and focused.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore


def load_session_index(
    session_dir: Path,
) -> Optional[Tuple[faiss.Index, InMemoryDocstore, Dict[int, str]]]:
    """Load FAISS index, docstore and mapping for a given session directory.

    Returns None if the expected files are not present.
    """
    faiss_dir = session_dir / "faiss_index"
    index_file = faiss_dir / "faiss.index"
    docstore_file = faiss_dir / "docstore.pkl"
    mapping_file = faiss_dir / "index_mapping.pkl"

    if not index_file.exists() or not docstore_file.exists():
        return None

    session_index = faiss.read_index(str(index_file))
    with open(docstore_file, "rb") as f:
        session_docstore = pickle.load(f)

    if mapping_file.exists():
        with open(mapping_file, "rb") as f:
            session_mapping = pickle.load(f)
    else:
        session_mapping = {i: str(i) for i in range(session_index.ntotal)}

    logging.info("Loaded session FAISS artifacts from %s", session_dir)
    return session_index, session_docstore, session_mapping


def save_vector_store(session_manager, vector_store) -> None:
    """Persist a vector store's index, docstore and mapping to the session dir.

    The function ensures the necessary directories exist and writes three
    files into the session's `faiss_index` directory.
    """
    session_manager.CURRENT_VECTOR_DIR.mkdir(exist_ok=True)
    session_manager.CURRENT_DATA_DIR.mkdir(exist_ok=True)

    current_index_file = session_manager.CURRENT_VECTOR_DIR / "faiss.index"
    current_docstore_file = session_manager.CURRENT_VECTOR_DIR / "docstore.pkl"
    current_mapping_file = session_manager.CURRENT_VECTOR_DIR / "index_mapping.pkl"

    faiss.write_index(vector_store.index, str(current_index_file))
    with open(current_docstore_file, "wb") as f:
        pickle.dump(vector_store.docstore, f)
    with open(current_mapping_file, "wb") as f:
        pickle.dump(vector_store.index_to_docstore_id, f)

    logging.info(
        "FAISS index and docstore saved to %s", session_manager.CURRENT_VECTOR_DIR
    )
