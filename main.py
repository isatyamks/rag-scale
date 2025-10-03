import os
import faiss
import logging
import pickle
from pathlib import Path
from typing_extensions import List, TypedDict
from datetime import datetime

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain import hub
from langgraph.graph import START, StateGraph

#system chain retrieval chain pipeline

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# -----------------------
# Logging configuration
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------
# Environment & models
# -----------------------
os.environ["LANGSMITH_API_KEY"] = "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"
embeddings = OllamaEmbeddings(model="llama3")
llm = ChatOllama(model="llama3")

# -----------------------
# Paths for session-based storage (both FAISS and data in same session folder)
# -----------------------
BASE_SESSION_DIR = Path("sessions")
BASE_SESSION_DIR.mkdir(exist_ok=True)

# Create timestamped session directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_SESSION_DIR = BASE_SESSION_DIR / f"session_{timestamp}"
CURRENT_SESSION_DIR.mkdir(exist_ok=True)

# Subdirectories within the session
CURRENT_VECTOR_DIR = CURRENT_SESSION_DIR / "faiss_index"
CURRENT_DATA_DIR = CURRENT_SESSION_DIR / "data"

embedding_dim = len(embeddings.embed_query("hello world"))

# -----------------------
# Functions to handle multiple FAISS indices
# -----------------------
def load_all_existing_indices():
    """Load and merge all existing FAISS indices from previous sessions"""
    all_docs = []
    combined_docstore = InMemoryDocstore()
    combined_index = faiss.IndexFlatL2(embedding_dim)
    combined_mapping = {}
    current_id_offset = 0
    
    # Find all session directories
    session_dirs = [d for d in BASE_SESSION_DIR.iterdir() if d.is_dir() and d.name.startswith("session_")]
    
    if not session_dirs:
        logging.info("No existing FAISS indices found.")
        return combined_index, combined_docstore, combined_mapping
    
    logging.info(f"Found {len(session_dirs)} existing FAISS session directories.")
    
    for session_dir in sorted(session_dirs):
        # FAISS files are in the faiss_index subdirectory of each session
        faiss_dir = session_dir / "faiss_index"
        index_file = faiss_dir / "faiss.index"
        docstore_file = faiss_dir / "docstore.pkl"
        mapping_file = faiss_dir / "index_mapping.pkl"
        
        if index_file.exists() and docstore_file.exists():
            logging.info(f"Loading FAISS index from {session_dir.name}...")
            
            # Load individual components
            session_index = faiss.read_index(str(index_file))
            with open(docstore_file, "rb") as f:
                session_docstore = pickle.load(f)
            
            # Load mapping if exists
            if mapping_file.exists():
                with open(mapping_file, "rb") as f:
                    session_mapping = pickle.load(f)
            else:
                # Reconstruct mapping
                session_mapping = {}
                for i in range(session_index.ntotal):
                    session_mapping[i] = str(i)
            
            # Get vectors from the session index
            if session_index.ntotal > 0:
                vectors = session_index.reconstruct_n(0, session_index.ntotal)
                combined_index.add(vectors)
                
                # Merge docstore and update mapping
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

def list_all_sessions():
    """List all available sessions"""
    session_dirs = [d for d in BASE_SESSION_DIR.iterdir() if d.is_dir() and d.name.startswith("session_")]
    if session_dirs:
        logging.info(f"Available sessions: {[d.name for d in sorted(session_dirs)]}")
        for session_dir in sorted(session_dirs):
            data_dir = session_dir / "data"
            faiss_dir = session_dir / "faiss_index"
            raw_files = list((data_dir / "raw").glob("*.txt")) if (data_dir / "raw").exists() else []
            chunks_file = (data_dir / "processed" / "chunks.pkl") if (data_dir / "processed").exists() else None
            faiss_exists = (faiss_dir / "faiss.index").exists() if faiss_dir.exists() else False
            
            logging.info(f"  {session_dir.name}:")
            logging.info(f"    - Raw files: {len(raw_files)} files")
            logging.info(f"    - Chunks: {'Yes' if chunks_file and chunks_file.exists() else 'No'}")
            logging.info(f"    - FAISS index: {'Yes' if faiss_exists else 'No'}")
    return sorted(session_dirs)

def load_chunks_from_session(session_name: str):
    """Load chunks from a specific session"""
    session_dir = BASE_SESSION_DIR / session_name
    chunks_file = session_dir / "data" / "processed" / "chunks.pkl"
    
    if chunks_file.exists():
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
        logging.info(f"Loaded {len(chunks)} chunks from {session_name}")
        return chunks
    else:
        logging.warning(f"No chunks found in session {session_name}")
        return []

def create_new_session_vector_store():
    """Create a new vector store for the current session"""
    CURRENT_VECTOR_DIR.mkdir(exist_ok=True)
    CURRENT_DATA_DIR.mkdir(exist_ok=True)
    
    # List existing sessions
    existing_sessions = list_all_sessions()
    
    # Try to load all existing indices first
    combined_index, combined_docstore, combined_mapping = load_all_existing_indices()
    
    # Create vector store with combined data
    vector_store = FAISS(
        embedding_function=embeddings,
        index=combined_index,
        docstore=combined_docstore,
        index_to_docstore_id=combined_mapping,
    )
    
    return vector_store

# Create vector store
vector_store = create_new_session_vector_store()

# -----------------------
# Load & chunk documents
# -----------------------
def load_and_chunk(txt_files: list[str], chunk_size=1000, chunk_overlap=200):
    all_docs: list[Document] = []
    logging.info("Loading documents from local .txt files...")
    
    for file_path in txt_files:
        if Path(file_path).exists():
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            # Normalize content to lowercase
            # for doc in docs:
            #     doc.page_content = doc.page_content.lower()
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
        # Create directories for current session
        CURRENT_VECTOR_DIR.mkdir(exist_ok=True)
        CURRENT_DATA_DIR.mkdir(exist_ok=True)
        
        logging.info(f"Adding chunks to vector store in session: {CURRENT_VECTOR_DIR.name}")
        vector_store.add_documents(chunks)
        
        # Define file paths for current session
        current_index_file = CURRENT_VECTOR_DIR / "faiss.index"
        current_docstore_file = CURRENT_VECTOR_DIR / "docstore.pkl"
        current_mapping_file = CURRENT_VECTOR_DIR / "index_mapping.pkl"
        
        # Save FAISS index
        faiss.write_index(vector_store.index, str(current_index_file))
        # Save the InMemoryDocstore object
        with open(current_docstore_file, "wb") as f:
            pickle.dump(vector_store.docstore, f)
        # Save the index_to_docstore_id mapping
        with open(current_mapping_file, "wb") as f:
            pickle.dump(vector_store.index_to_docstore_id, f)
        
        logging.info(f"FAISS index and docstore saved to {CURRENT_VECTOR_DIR}")

def save_raw_and_chunks(txt_files: list[str], chunks: list[Document]):
    """Save raw files and chunks to session data directory"""
    if chunks:
        # Create data directories within the session
        CURRENT_DATA_DIR.mkdir(exist_ok=True)
        processed_dir = CURRENT_DATA_DIR / "processed"
        raw_dir = CURRENT_DATA_DIR / "raw"
        processed_dir.mkdir(exist_ok=True)
        raw_dir.mkdir(exist_ok=True)
        
        # Save chunks as pickle file
        chunks_file = processed_dir / "chunks.pkl"
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
        for txt_file in txt_files:
            if Path(txt_file).exists():
                source_file = Path(txt_file)
                dest_file = raw_dir / source_file.name
                dest_file.write_text(source_file.read_text(encoding="utf-8"), encoding="utf-8")
                logging.info(f"Copied {source_file.name} to session raw directory")
        
        logging.info(f"Raw files and chunks saved to {CURRENT_DATA_DIR}")
        logging.info(f"Session structure: {CURRENT_SESSION_DIR}")
        logging.info(f"  ├── faiss_index/ (FAISS vectors, docstore, mapping)")
        logging.info(f"  └── data/")
        logging.info(f"      ├── raw/ (original txt files)")
        logging.info(f"      └── processed/ (chunks.pkl, chunks.txt)")

# -----------------------
# RAG State & functions
# -----------------------
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

prompt = hub.pull("rlm/rag-prompt")

"""
#creating retrival chain by system prompt
system_prompt = (
    "instruct 1"
    "instruct 2"
    "instruct 3"
    "instruct 4"
    "instruct 5"
    "instruct 6"
    "instruct 7"
    "instruct 8"

)

system_prompt = ChatPromptTemplate.from_messages(

    [

        ("system", system_prompt),
        ("human", "{input}"),
    ]

)


qa_chain  = create_retrieval_chain(llm,system_prompt)

"""




def retrieve(state: State, top_k=5):
    try:
        retrieved_docs = vector_store.similarity_search(state["question"], k=top_k)
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
        print(messages)
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

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})


def retriever_data (str):
    retrieved_docs = retriever.invoke(str)
    print(f"\n\nNumber of retrieved documents: {len(retrieved_docs)}")
    print("\nRetrieved documents:")
    for i, doc in enumerate(retrieved_docs):
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content[:200]}...")  
        if hasattr(doc, 'metadata') and doc.metadata:
            print(f"Metadata: {doc.metadata}\n\n")


# -----------------------
# Main loop
# -----------------------
if __name__ == "__main__":
    # Add new files incrementally
    txt_files = ["data/raw/test.txt"]  # add more files here as needed
    
    print(f"=== Starting New RAG Session ===")
    print(f"Session ID: {timestamp}")
    print(f"Session Directory: {CURRENT_SESSION_DIR}")
    print(f"Structure:")
    print(f"  ├── faiss_index/ (vectors, docstore, mappings)")
    print(f"  └── data/")
    print(f"      ├── raw/ (original txt files)")
    print(f"      └── processed/ (chunks)")
    print()
    
    chunks = load_and_chunk(txt_files)
    add_to_vector_store(chunks)
    save_raw_and_chunks(txt_files, chunks)

    print("RAG chat ready! (type 'exit' to quit)")
    while True:
        temp = input("\nEnter your question:\n")
        if temp.lower() == "exit":
            print("Exiting...")
            break
        query = {"question": temp}
        print(query)
        retriever_data(temp)
        response = graph.invoke(query)
        print("\nAnswer:", response["answer"], "\n")
