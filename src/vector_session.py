def create_session_only_vector_store():
    """Create a vector store for current session only (isolated mode)"""
    CURRENT_VECTOR_DIR.mkdir(exist_ok=True)
    CURRENT_DATA_DIR.mkdir(exist_ok=True)

    # Check if current session has existing FAISS index
    current_index_file = CURRENT_VECTOR_DIR / "faiss.index"
    current_docstore_file = CURRENT_VECTOR_DIR / "docstore.pkl"
    current_mapping_file = CURRENT_VECTOR_DIR / "index_mapping.pkl"

    if current_index_file.exists() and current_docstore_file.exists():
        logging.info(
            f"Loading existing FAISS index from current session: {CURRENT_SESSION_DIR.name}"
        )

        # Load existing session data
        session_index = faiss.read_index(str(current_index_file))
        with open(current_docstore_file, "rb") as f:
            session_docstore = pickle.load(f)

        if current_mapping_file.exists():
            with open(current_mapping_file, "rb") as f:
                session_mapping = pickle.load(f)
        else:
            session_mapping = {i: str(i) for i in range(session_index.ntotal)}

        vector_store = FAISS(
            embedding_function=embeddings,
            index=session_index,
            docstore=session_docstore,
            index_to_docstore_id=session_mapping,
        )
    else:
        # Create new empty vector store
        logging.info("Creating new empty vector store for current session")
        empty_index = faiss.IndexFlatL2(embedding_dim)
        empty_docstore = InMemoryDocstore()
        empty_mapping = {}

        vector_store = FAISS(
            embedding_function=embeddings,
            index=empty_index,
            docstore=empty_docstore,
            index_to_docstore_id=empty_mapping,
        )

    return vector_store
