"""Document processing helpers.

Small utilities for loading local text files, splitting into chunks using
the configured text splitter, and saving both the raw and processed chunks
into the session directory.
"""

from __future__ import annotations

from pathlib import Path
import logging
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


__all__ = ["DocumentProcessor"]


class DocumentProcessor:
    """Document loading and chunking helpers."""

    def __init__(self) -> None:
        pass

    def load_and_chunk(
        self, txt_files: List[str], chunk_size: int = 2000, chunk_overlap: int = 200
    ) -> List[Document]:
        all_docs: List[Document] = []
        logging.info("Loading documents from local .txt files...")

        for file_path in txt_files:
            if Path(file_path).exists():
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
                logging.info(f"Loaded {len(docs)} documents from {file_path}")
                all_docs.extend(docs)
            else:
                logging.warning(f"File not found: {file_path}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(all_docs)
        logging.info(f"Created {len(chunks)} chunks.")
        return chunks

    def save_raw_and_chunks(
        self, txt_files: List[str], chunks: List[Document], session_manager
    ) -> None:
        """Save raw files and chunks to session data directory"""
        if not chunks:
            return

        # Create data directories within the session
        session_manager.CURRENT_DATA_DIR.mkdir(exist_ok=True)
        processed_dir = session_manager.CURRENT_DATA_DIR / "processed"
        raw_dir = session_manager.CURRENT_DATA_DIR / "raw"
        processed_dir.mkdir(exist_ok=True)
        raw_dir.mkdir(exist_ok=True)

        # Save chunks as pickle file
        chunks_file = processed_dir / "chunks.pkl"
        import pickle

        with open(chunks_file, "wb") as f:
            pickle.dump(chunks, f)

        # Save chunks as readable text file for reference
        chunks_text_file = processed_dir / "chunks.txt"
        with open(chunks_text_file, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"=== CHUNK {i+1} ===\n")
                f.write(f"Source: {chunk.metadata.get('source', 'Unknown')}\n")
                f.write(f"Content:\n{chunk.page_content}\n\n")

        # Copy raw files to session directory
        from pathlib import Path as _P

        for txt_file in txt_files:
            if _P(txt_file).exists():
                source_file = _P(txt_file)
                dest_file = raw_dir / source_file.name
                dest_file.write_text(
                    source_file.read_text(encoding="utf-8"), encoding="utf-8"
                )
                logging.info(f"Copied {source_file.name} to session raw directory")

        logging.info(
            f"Raw files and chunks saved to {session_manager.CURRENT_DATA_DIR}"
        )
