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
# Paths for FAISS index & docstore
# -----------------------
VECTOR_DIR = Path("faiss_index")
VECTOR_DIR.mkdir(exist_ok=True)
index_file = VECTOR_DIR / "faiss.index"
docstore_file = VECTOR_DIR / "docstore.pkl"
mapping_file = VECTOR_DIR / "index_mapping.pkl"

embedding_dim = len(embeddings.embed_query("hello world"))

# -----------------------
# Load or create FAISS + InMemoryDocstore
# -----------------------
if index_file.exists() and docstore_file.exists():
    logging.info("Loading existing FAISS index and docstore...")
    index = faiss.read_index(str(index_file))
    with open(docstore_file, "rb") as f:
        docstore = pickle.load(f)  # already an InMemoryDocstore
    
    # Load or reconstruct index_to_docstore_id mapping
    if mapping_file.exists():
        with open(mapping_file, "rb") as f:
            index_to_docstore_id = pickle.load(f)
    else:
        # Reconstruct index_to_docstore_id mapping
        index_to_docstore_id = {}
        for i in range(index.ntotal):
            index_to_docstore_id[i] = str(i)
else:
    logging.info("Creating new FAISS index and docstore...")
    index = faiss.IndexFlatL2(embedding_dim)
    docstore = InMemoryDocstore()
    index_to_docstore_id = {}

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
)

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
        logging.info("Adding chunks to vector store...")
        vector_store.add_documents(chunks)
        # Save FAISS index
        faiss.write_index(vector_store.index, str(index_file))
        # Save the InMemoryDocstore object
        with open(docstore_file, "wb") as f:
            pickle.dump(vector_store.docstore, f)
        # Save the index_to_docstore_id mapping
        mapping_file = VECTOR_DIR / "index_mapping.pkl"
        with open(mapping_file, "wb") as f:
            pickle.dump(vector_store.index_to_docstore_id, f)
        logging.info("FAISS index and docstore saved.")

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
    chunks = load_and_chunk(txt_files)
    add_to_vector_store(chunks)

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
