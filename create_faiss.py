import os
import faiss
import pickle
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Directory to store FAISS index
SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

# Example: Create FAISS index from all .txt files in data/raw
RAW_DATA_DIR = Path("data/raw")

def create_faiss_index():
    txt_files = list(RAW_DATA_DIR.glob("*.txt"))
    if not txt_files:
        print("No .txt files found in data/raw.")
        return

    docs = []
    for file_path in txt_files:
        loader = TextLoader(str(file_path))
        docs.extend(loader.load())

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    # Create embeddings
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Save FAISS index
    faiss_index_path = SESSIONS_DIR / "faiss_index.pkl"
    with open(faiss_index_path, "wb") as f:
        pickle.dump(vectorstore, f)
    print(f"FAISS index created and saved to {faiss_index_path}")

if __name__ == "__main__":
    create_faiss_index()
