# RAG Scale

This repository contains a small Retrieval-Augmented Generation (RAG) demo.

Structure
- `main.py` - CLI entrypoint (interactive). Prefer running via `run.py`.
- `run.py` - small launcher that ensures the repository root is on `sys.path`.
- `src/` - helper modules:
  - `session_manager.py` - session directory and CSV logging utilities
  - `vector_manager.py` - FAISS index and vector store helpers
  - `doc_processor.py` - local file loading and chunking helpers
  - `rag_graph.py` - thin retrieve/generate wrappers

Run
From the project root:

```cmd
python run.py
```

Notes
- The project relies on Ollama-based embeddings/LLM (configured in `main.py`).
- The `src` package contains the functional modules; `main.py` wires them together.
