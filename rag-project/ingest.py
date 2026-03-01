import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "data/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def ingest_documents(docs: List[Document]):
    """
    Ingests loaded documents using a sequence of: Chunking -> Embedding -> Storing in Chroma.
    """
    if not docs:
        print("No documents were provided.")
        return None
        
    print(f"Total documents loaded: {len(docs)}")

    # 1. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,  
        chunk_overlap=300, 
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    all_chunks = text_splitter.split_documents(docs)
    
    if not all_chunks:
        print("No chunks were created.")
        return None
        
    print(f"Total chunks created: {len(all_chunks)}")
    
    # 2. Create Vector DB
    print(f"Initializing embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print("Building Chroma vector store...")
    # This stores the vectors locally in CHROMA_PATH
    db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"Successfully ingrained {len(all_chunks)} chunks to {CHROMA_PATH}")
    return db

if __name__ == "__main__":
    pass
