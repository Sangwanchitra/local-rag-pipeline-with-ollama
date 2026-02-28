from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

CHROMA_PATH = "data/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_retriever():
    """
    Initializes and returns a retriever from the local Chroma vector store.
    """
    if not os.path.exists(CHROMA_PATH):
        print(f"Index not found at {CHROMA_PATH}. Please run ingestion first.")
        return None
        
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # Retrieve top 5 most similar chunks
    # We could plug in a ReRanker here (e.g., Cohere/BGE Reranker) for further optimization
    # but for simplicity and strict local constraints, we stick to standard vector search initially.
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    return retriever

if __name__ == "__main__":
    pass
