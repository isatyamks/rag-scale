from __future__ import annotations
from pathlib import Path
import logging
import pickle
from typing import Dict, Tuple
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore


def load_all_existing_indices(
    base_session_dir: Path, embedding_dim: int
) -> Tuple[faiss.Index, InMemoryDocstore, Dict[int, str]]:
    combined_docstore = InMemoryDocstore()
    combined_index = faiss.IndexFlatL2(embedding_dim)
    combined_mapping: Dict[int, str] = {}
    current_id_offset = 0

    session_dirs = [
        d
        for d in base_session_dir.iterdir()
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
