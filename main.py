import os
import faiss
import logging
import pickle
from pathlib import Path
from typing_extensions import List, TypedDict

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain import hub
from langgraph.graph import START, StateGraph

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------
# Environment & models
# -----------------------
os.environ["LANGSMITH_API_KEY"] = "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"
embeddings = OllamaEmbeddings(model="llama3")
llm = ChatOllama(model="llama3")

# -----------------------
# FAISS + Docstore persistence
# -----------------------
VECTOR_DIR = Path("faiss_index")
VECTOR_DIR.mkdir(exist_ok=True)
index_file = VECTOR_DIR / "faiss.index"
docstore_file = VECTOR_DIR / "docstore.pkl"

embedding_dim = len(embeddings.embed_query("hello world"))

# Load or create
if index_file.exists() and docstore_file.exists():
    logging.info("Loading existing FAISS index and docstore...")
    index = faiss.read_index(str(index_file))
    with open(docstore_file, "rb") as f:
        docstore = pickle.load(f)
else:
    logging.info("Creating new FAISS index and docstore...")
    index = faiss.IndexFlatL2(embedding_dim)
    docstore = {}  # will store {id: Document}

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=docstore,
    index_to_docstore_id={},
)

# -----------------------
# Load & split documents
# -----------------------
def load_and_chunk(txt_files: list[str], chunk_size=1000, chunk_overlap=200):
    all_docs: list[Document] = []
    logging.info("Loading documents from local .txt files...")
    for file_path in txt_files:
        if Path(file_path).exists():
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            logging.info(f"Loaded {len(docs)} documents from {file_path}")
            all_docs.extend(docs)
        else:
            logging.warning(f"File not found: {file_path}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(all_docs)
    logging.info(f"Created {len(chunks)} chunks.")
    return chunks

# -----------------------
# Add new chunks to vector store
# -----------------------
def add_to_vector_store(chunks: list[Document]):
    if chunks:
        logging.info("Adding chunks to vector store...")
        vector_store.add_documents(chunks)
        # Save FAISS index
        faiss.write_index(vector_store.index, str(index_file))
        # Save docstore
        with open(docstore_file, "wb") as f:
            pickle.dump(vector_store.docstore, f)
        logging.info("FAISS index and docstore saved.")

# -----------------------
# RAG State & functions
# -----------------------
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

prompt = hub.pull("rlm/rag-prompt")

def retrieve(state: State, top_k=5):
    try:
        retrieved_docs = vector_store.similarity_search(state["question"], k=top_k)
        # keep doc IDs as-is (handles UUIDs)
        for i, doc in enumerate(retrieved_docs):
            if not hasattr(doc, "id") or doc.id is None:
                doc.id = i
        logging.info(f"Retrieved {len(retrieved_docs)} relevant chunks.")
        return {"context": retrieved_docs}
    except Exception as e:
        logging.error(f"Error in retrieval: {e}")
        return {"context": []}

def generate(state: State):
    try:
        context_text = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": context_text})
        response = llm.invoke(messages)
        return {"answer": response.content}
    except Exception as e:
        logging.error(f"Error in generation: {e}")
        return {"answer": "Failed to generate response."}

# -----------------------
# Build RAG graph
# -----------------------
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# -----------------------
# Main loop
# -----------------------
if __name__ == "__main__":
    # Example: add new files incrementally
    txt_files = ["data/raw/test.txt"]  # add new files here
    chunks = load_and_chunk(txt_files)
    add_to_vector_store(chunks)

    print("RAG chat ready! (type 'exit' to quit)")
    while True:
        temp = input("\nEnter your question:\n")
        if temp.lower() == "exit":
            print("Exiting...")
            break
        query = {"question": temp}
        response = graph.invoke(query)
        print("\nAnswer:", response["answer"], "\n")
