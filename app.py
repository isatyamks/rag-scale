import os
import faiss
import logging
from pathlib import Path

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import bs4



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
os.environ["LANGSMITH_API_KEY"] = "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"

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
# Load & split documents
# -----------------------
def load_and_chunk(
    urls: list[str] = None, 
    txt_files: list[str] = None, 
    chunk_size=1000, 
    chunk_overlap=200
):
    all_docs: list[Document] = []

    # Load from URLs
    if urls:
        logging.info("Loading documents from web URLs...")
        loader = WebBaseLoader(
            web_paths=urls,
            bs_kwargs={"parse_only": bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )}
        )
        docs = loader.load()
        logging.info(f"Loaded {len(docs)} documents from web.")
        all_docs.extend(docs)

    # Load from local .txt files
    if txt_files:
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
    with open("blog_raw.txt", "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(doc.page_content.strip() + "\n\n")
    with open("blog_chunks.txt", "w", encoding="utf-8") as f:
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

print(prompt)   
if hasattr(prompt, "template"):
    print(prompt.template) 


def retrieve(state: State, top_k=5):
    try:
        retrieved_docs = vector_store.similarity_search(state["question"], k=top_k)
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

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # Load documents if FAISS index is empty
    if vector_store.index.ntotal == 0:
        chunks = load_and_chunk(
            urls=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
            txt_files=["data\\raw\\sapiens.txt"]
        )
        add_to_vector_store(chunks)
    
    print(prompt.type)   
    if hasattr(prompt, "template"):
        print(prompt.template) 

    # Query example
    query = {"question": "Sapiens?"}
    response = graph.invoke(query)
    print(response["answer"])
