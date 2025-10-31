# RAG Scale

RAG Scale is a compact, practical Retrieval-Augmented Generation (RAG)
prototype that demonstrates how to build an end-to-end retrieval + LLM
workflow using local text corpora and a FAISS-backed vector store. It is
designed for exploration, small experiments, and as a starting point for
prototyping RAG-based assistants.

This repository focuses on clarity and modularity: document processing,
vector management, session handling and the simple retrieve->generate bridge
are implemented as small, testable modules under `src/`.

Key features
- Session-based indexing: each corpus is stored under `sessions/` and
  maintained independently.
- FAISS-backed vector stores for fast similarity search.
- Simple CLI for adding a corpus, creating chunks, building a vector store,
  and running interactive queries against a local LLM.
- CSV-based interaction logging for basic analytics and debugging.

Quickstart (run locally)
1. Create a Python virtual environment and activate it (Windows example):

```cmd
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install runtime dependencies (see `requirements.txt`):

```cmd
pip install -r requirements.txt
```

3. Add a text file under `data/raw/` (for example `data/raw/my_corpus.txt`) or
   use `wiki.py` to scrape a single Wikipedia page into `data/raw/`.

4. Run the CLI to create a session and interact with the corpus:

```cmd
python main.py
```

High-level usage notes
- The CLI prompts for a corpus name and creates a session directory at
  `sessions/session_<corpus_name>/`.
- The application always runs in session-only mode (no global search across
  all sessions). This keeps sessions isolated and reproducible.
- New corpora are chunked and indexed; interactions (questions/answers)
  are logged to CSV for each session and into a global CSV under
  `sessions/interactions.csv`.

Project layout

- `main.py` — interactive CLI that wires components together.
- `wiki.py` — small utility to save a single Wikipedia page into `data/raw/`.
- `src/doc_processor.py` — document loading and chunking utilities.
- `src/vector_manager.py` — vector store and FAISS helpers (delegates
  persistence and FAISS combination to smaller helpers).
- `src/faiss_utils.py` — helpers to load and combine FAISS indices.
- `src/persistence.py` — helpers to load/save per-session FAISS artifacts.
- `src/session_manager.py` — session directory creation, CSV logging, and
  simple introspection helpers.
- `src/rag_graph.py` — thin retrieve/generate bridge that adapts the
  vector store and LLM into a predictable interface used by the CLI.

Design notes and assumptions
- The project uses Ollama embeddings and an Ollama chat model in
  `main.py`. This requires a local Ollama runtime or an endpoint compatible
  with the `langchain-ollama` integration. If you do not have Ollama, you
  can substitute another embeddings / LLM provider by updating the
  initialization in `main.py`.
- Sessions are intentionally isolated; there is no automatic global index
  merge during the default CLI flow. If you need a combined/global index,
  use the `src.faiss_utils` helper to load and merge indices manually.

Development
- Formatting and linting: this repository uses `black` and `ruff`.
  Install them (optionally) from `requirements.txt` (they are listed under
  developer tools) and run:

```cmd
python -m ruff check --fix src
python -m black .
```

- Add unit tests under `tests/` and run them with pytest

Contributing
- Contributions are welcome. Please open issues for bugs or feature
  requests and submit pull requests with focused changes. Follow the
  existing code style and run formatters/linters before submitting.

License
- This project is provided as-is for learning and prototyping. Add a
  LICENSE file as appropriate for your use-case.

If you'd like, I can also:
- Pin exact package versions in `requirements.txt` for reproducible installs.
- Add a `requirements-dev.txt` containing dev tools only.
- Create a small `Makefile` or `scripts/` launcher to simplify common
  developer tasks (format, lint, test, run).

Enjoy exploring RAG workflows — tell me if you want the README tuned for a
particular audience (researchers, engineers, or end users).
