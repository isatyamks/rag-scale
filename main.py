import os
import faiss
import logging
from pathlib import Path

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# -----------------------
# Logging configuration
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -----------------------
# Environment setup
# -----------------------
os.environ["LANGSMITH_API_KEY"] = "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"

# -----------------------
# Models & Embeddings
# -----------------------
embeddings = OllamaEmbeddings(model="llama3")
llm = ChatOllama(model="llama3")


# -----------------------
# FAISS VectorStore setup
# -----------------------
VECTOR_DIR = Path("faiss_index")
VECTOR_DIR.mkdir(exist_ok=True)

embedding_dim = len(embeddings.embed_query("hello world"))
index_file = VECTOR_DIR / "faiss.index"

if index_file.exists():
    logging.info("Loading existing FAISS index from disk...")
    index = faiss.read_index(str(index_file))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
else:
    logging.info("Creating new FAISS index...")
    index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

# -----------------------
# Load & split documents from .txt files
# -----------------------
def load_and_chunk(
    txt_files: list[str], 
    chunk_size=1000, 
    chunk_overlap=200
):
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

    # Split documents into chunks
    logging.info("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(all_docs)
    logging.info(f"Created {len(chunks)} chunks.")

    # Save raw & chunked text for reference
    with open("txt_raw.txt", "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(doc.page_content.strip() + "\n\n")
    with open("txt_chunks.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"--- Chunk {i} ---\n")
            f.write(chunk.page_content.strip() + "\n\n")

    return chunks

# -----------------------
# Add documents to vector store
# -----------------------
def add_to_vector_store(chunks: list[Document]):
    if chunks:
        logging.info("Adding chunks to vector store...")
        vector_store.add_documents(chunks)
        faiss.write_index(vector_store.index, str(index_file))
        logging.info("FAISS index saved to disk.")

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
        for i, doc in enumerate(retrieved_docs):
            if hasattr(doc, "id") and isinstance(doc.id, str):
                doc.id = doc.id 
            else:
                doc.id = i
        logging.info(f"Retrieved {len(retrieved_docs)} relevant chunks.")
        return {"context": retrieved_docs}
    except Exception as e:
        logging.error(f"Error in retrieval: {e}")
        return {"context": []}




def generate(state: State):
    try:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
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


if __name__ == "__main__":
    print("phase1")
    if vector_store.index.ntotal == 0:
        txt_files = ["data\\raw\\test.txt"]  
        chunks = load_and_chunk(txt_files=txt_files)
        add_to_vector_store(chunks)
    print(prompt)  
    print("phase1") 
    if hasattr(prompt, "template"):
        print(prompt.template) 
    print("phase1")
    while True:
        temp = input("Enter your question (type 'exit' to quit):\n")
        if temp.lower() == "exit":
            print("Exiting...")
            break
        
        query = {"question": temp}
        response = graph.invoke(query)
        print("\nAnswer:", response["answer"], "\n")


