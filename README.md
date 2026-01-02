# RAG Scale

**RAG Scale** is a modular, privacy-focused framework designed for building local Retrieval-Augmented Generation (RAG) applications. It leverages **Ollama** for local Large Language Model (LLM) inference and embeddings, combined with **FAISS** for efficient high-dimensional vector search.

The system is architected to support isolated research "sessions," allowing users to index and query distinct text corpora or Wikipedia topics without context contamination.

## System Architecture

The codebase follows a domain-driven design pattern to ensure scalability and maintainability:

*   **`src/core`**: Core infrastructure managing Session lifecycles (`SessionManager`) and Vector Store operations (`VectorManager`).
*   **`src/data`**: Robust data ingestion pipeline including:
    *   **Loaders**: Utilities for fetching external content (e.g., `WikiLoader`).
    *   **Processors**: Text normalization and chunking logic (`DocumentProcessor`) based on best practices.
*   **`src/models`**: Abstraction layer for LLM and Embedding model initialization, currently optimized for Ollama.
*   **`src/rag`**: The inference engine containing the prompt construction and generation logic.
*   **`src/config`**: Centralized configuration management for model parameters, directory paths, and runtime settings.
*   **`tests/health`**: Comprehensive system health check suite to validate infrastructure components.

## Prerequisites

To execute this framework, ensure the following are installed and configured:

1.  **Python 3.10+**
2.  **[Ollama](https://ollama.com/)**: Required for local model inference.
    *   Pull the default embedding model: `ollama pull nomic-embed-text`
    *   Pull the default chat model: `ollama pull mistral`

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd rag-scale
    ```

2.  Install the required Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## System Validation

Before running the main application, you can execute the included health check suite to verify that your environment (Ollama, Network, Python dependencies) is correctly configured.

```bash
python test_system.py
```

This ensures all subsystems are operational before you begin.

## Usage

The framework is driven by a Command Line Interface (CLI) that orchestrates the RAG pipeline.

1.  **Initialize the Application**:
    ```bash
    python main.py
    ```

2.  **Select a Corpus**:
    When prompted, enter a topic name.
    *   **Local Processing**: The system first checks `data/raw/` for a matching `.txt` file.
    *   **External Fetch**: If no local file is found, it attempts to retrieve and sanitize the corresponding article from Wikipedia.

3.  **Interaction Phase**:
    Once the index is built or loaded, the interactive session begins. The system will retrieve relevant context for each query and generate a citation-backed response.

4.  **Session Management**:
    All artifacts (raw data, serialized chunks, and FAISS indexes) serve as a persistent state in the `sessions/` directory, allowing for instant resumption of previous topics.

## Features

*   **Privacy-First Design**: Operates entirely locally with no data transmitted to external APIs.
*   **Isolated Sessions**: Automatically manages separate environment states for different datasets.
*   **Health Monitoring**: Integrated tools to validate system integrity and model availability.
*   **Performance Metrics**: Detailed logging of query latency (`response_time_s`) and retrieval statistics (`retrieved_docs_count`) in CSV format for analysis.
*   **Extensible Design**: Modular architecture allows for straightforward integration of alternative vector stores or LLM providers.
