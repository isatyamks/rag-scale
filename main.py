import os
import faiss
import logging
import pickle
import csv
import uuid
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


from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

embeddings = OllamaEmbeddings(model="llama3")
llm = ChatOllama(model="llama3")

BASE_SESSION_DIR = Path("sessions")
BASE_SESSION_DIR.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_SESSION_DIR = BASE_SESSION_DIR / f"session_{timestamp}"
CURRENT_SESSION_DIR.mkdir(exist_ok=True)

CURRENT_VECTOR_DIR = CURRENT_SESSION_DIR / "faiss_index"
CURRENT_DATA_DIR = CURRENT_SESSION_DIR / "data"
CURRENT_LOGS_DIR = CURRENT_SESSION_DIR / "logs"
CURRENT_LOGS_DIR.mkdir(exist_ok=True)

CSV_LOG_FILE = CURRENT_LOGS_DIR / "interactions.csv"
GLOBAL_CSV_LOG_FILE = BASE_SESSION_DIR / "interactions.csv"

embedding_dim = len(embeddings.embed_query("hello world"))


def initialize_csv_logs():
    """Initialize CSV files with headers if they don't exist"""
    headers = [
        'interaction_id', 'session_id', 'timestamp', 'question', 'response', 
        'retrieved_docs_count', 'response_time_seconds', 'session_directory'
    ]
    
    # Initialize session-specific CSV
    if not CSV_LOG_FILE.exists():
        with open(CSV_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    # Initialize global CSV
    if not GLOBAL_CSV_LOG_FILE.exists():
        with open(GLOBAL_CSV_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def log_interaction_to_csv(question: str, response: str, retrieved_docs_count: int, response_time: float):
    """Log interaction to both session-specific and global CSV files"""
    interaction_id = str(uuid.uuid4())
    session_id = f"session_{timestamp}"
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    row_data = [
        interaction_id,
        session_id,
        current_timestamp,
        question.replace('\n', ' ').replace('\r', ''),  # Clean newlines for CSV
        response.replace('\n', ' ').replace('\r', ''),  # Clean newlines for CSV
        retrieved_docs_count,
        round(response_time, 3),
        str(CURRENT_SESSION_DIR)
    ]
    
    # Log to session-specific CSV
    with open(CSV_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)
    
    # Log to global CSV
    with open(GLOBAL_CSV_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)
    
    logging.info(f"Logged interaction {interaction_id} to CSV files")

def get_interaction_stats():
    """Get statistics from the global interactions CSV"""
    if not GLOBAL_CSV_LOG_FILE.exists():
        return "No interactions logged yet."
    
    try:
        with open(GLOBAL_CSV_LOG_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            interactions = list(reader)
        
        total_interactions = len(interactions)
        unique_sessions = len(set(row['session_id'] for row in interactions))
        avg_response_time = sum(float(row['response_time_seconds']) for row in interactions) / total_interactions if total_interactions > 0 else 0
        
        return f"📈 Stats: {total_interactions} interactions, {unique_sessions} sessions, {avg_response_time:.2f}s avg response time"
    except Exception as e:
        return f"Error reading interaction stats: {e}"

def view_recent_interactions(limit: int = 5):
    """View recent interactions from the global CSV"""
    if not GLOBAL_CSV_LOG_FILE.exists():
        print("No interactions logged yet.")
        return
    
    try:
        with open(GLOBAL_CSV_LOG_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            interactions = list(reader)
        
        recent = interactions[-limit:] if len(interactions) >= limit else interactions
        
        print(f"\n📋 Recent {len(recent)} interactions:")
        for i, interaction in enumerate(recent, 1):
            print(f"\n{i}. [{interaction['timestamp']}] Session: {interaction['session_id']}")
            print(f"   Q: {interaction['question'][:80]}...")
            print(f"   A: {interaction['response'][:80]}...")
            print(f"   📄 {interaction['retrieved_docs_count']} docs, ⏱️ {interaction['response_time_seconds']}s")
            
    except Exception as e:
        print(f"Error reading recent interactions: {e}")

def export_session_summary():
    """Export a summary of the current session"""
    summary_file = CURRENT_LOGS_DIR / "session_summary.txt"
    
    try:
        interactions = []
        if CSV_LOG_FILE.exists():
            with open(CSV_LOG_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                interactions = list(reader)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"RAG Session Summary\n")
            f.write(f"==================\n\n")
            f.write(f"Session ID: session_{timestamp}\n")
            f.write(f"Session Directory: {CURRENT_SESSION_DIR}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Interactions: {len(interactions)}\n\n")
            
            if interactions:
                avg_time = sum(float(row['response_time_seconds']) for row in interactions) / len(interactions)
                f.write(f"Average Response Time: {avg_time:.2f} seconds\n")
                f.write(f"Total Documents Retrieved: {sum(int(row['retrieved_docs_count']) for row in interactions)}\n\n")
                
                f.write("Interactions:\n")
                f.write("=============\n\n")
                for i, interaction in enumerate(interactions, 1):
                    f.write(f"{i}. [{interaction['timestamp']}]\n")
                    f.write(f"   Question: {interaction['question']}\n")
                    f.write(f"   Answer: {interaction['response']}\n")
                    f.write(f"   Retrieved: {interaction['retrieved_docs_count']} docs\n")
                    f.write(f"   Time: {interaction['response_time_seconds']}s\n\n")
        
        logging.info(f"Session summary exported to {summary_file}")
        
    except Exception as e:
        logging.error(f"Error exporting session summary: {e}")

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
        logging.info(f"  ├── data/")
        logging.info(f"  │   ├── raw/ (original txt files)")
        logging.info(f"  │   └── processed/ (chunks.pkl, chunks.txt)")
        logging.info(f"  └── logs/")
        logging.info(f"      └── interactions.csv (session Q&A log)")

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

retriever = vector_store.as_retriever(search_type="similarity", CURRENT_VECTOR_DIR = "context_vector" ,search_kwargs={"k": 6})


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
    temp1  = input("Enter the Newer Corpus Document: ")
    txt_files = [f"data/raw/{temp1}.txt"]  # add more files here as needed
    
    print(f"=== Starting New RAG Session ===")
    print(f"Session ID: {timestamp}")
    print(f"Session Directory: {CURRENT_SESSION_DIR}")
    print(f"CSV Logs: {CSV_LOG_FILE}")
    print()
    
    # Initialize CSV logging
    initialize_csv_logs()
    
    chunks = load_and_chunk(txt_files)
    add_to_vector_store(chunks)
    save_raw_and_chunks(txt_files, chunks)
    

    while True:
        temp = input("\nEnter your question:\n")
        if temp.lower() == "exit":
            print("\nExiting RAG session...")
            export_session_summary()
            print(f"\nSession completed. Files saved to:")
            print(f"Session CSV: {CSV_LOG_FILE}")
            print(f"Global CSV: {GLOBAL_CSV_LOG_FILE}")
            print(f"Session Summary: {CURRENT_LOGS_DIR / 'session_summary.txt'}")
            print(f"Session Directory: {CURRENT_SESSION_DIR}")
            break
        start_time = datetime.now()
        
        query = {"question": temp}
        retriever_start = datetime.now()
        retrieved_docs = retriever.invoke(temp)
        retrieved_docs_count = len(retrieved_docs)
        response = graph.invoke(query)
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        answer = response["answer"]
        print(f"\033[92m\nAnswer: {answer}\n\033[0m")
        print(f"Response time: {response_time:.2f} seconds")
        
        # Log to CSV
        log_interaction_to_csv(temp, answer, retrieved_docs_count, response_time)
        print()