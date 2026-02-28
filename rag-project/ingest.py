import os
import fitz
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "data/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class SmartPDFLoader:
    """
    Custom PDF loader that uses PyMuPDF to extract text and heuristic section formatting.
    Extracts metadata: source filename, page number, and section title.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.filename = os.path.basename(file_path)

    def load(self) -> List[Document]:
        doc = fitz.open(self.file_path)
        documents = []
        current_section = "Unknown Section"
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if not text:
                                continue
                            
                            font_size = span.get("size", 0)
                            font_flags = span.get("flags", 0)
                            # 16 is flag for bold in PyMuPDF
                            is_bold = bool(font_flags & 16) or "bold" in span.get("font", "").lower()
                            
                            # Heuristic: headers are usually bold or large, short lines
                            if (is_bold or font_size > 12) and len(text) < 100:
                                current_section = text
                            else:
                                # We treat it as paragraph content
                                metadata = {
                                    "source": self.filename,
                                    "page": page_num + 1,
                                    "section": current_section
                                }
                                documents.append(Document(page_content=text, metadata=metadata))
        
        # Combine documents that have the same page and section to form larger paragraphs before chunking
        combined_docs = []
        if not documents:
            return []
            
        current_doc = documents[0]
        for next_doc in documents[1:]:
            if current_doc.metadata["page"] == next_doc.metadata["page"] and current_doc.metadata["section"] == next_doc.metadata["section"]:
                current_doc.page_content += " " + next_doc.page_content
            else:
                combined_docs.append(current_doc)
                current_doc = next_doc
        combined_docs.append(current_doc)
        
        return combined_docs

def ingest_documents(pdf_paths: List[str]):
    """
    Ingests PDFs using a sequence of: Loading -> Chunking -> Embedding -> Storing in Chroma.
    """
    all_chunks = []
    
    # 1. Extract text and metadata
    for path in pdf_paths:
        if os.path.exists(path):
            print(f"Processing: {path}")
            loader = SmartPDFLoader(path)
            docs = loader.load()
            
            # 2. Chunking
            # RecursiveCharacterTextSplitter for semantic sentence boundaries 
            # ~500-800 tokens corresponds to roughly 2000-3200 characters. Overlap ~10-15%.
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2500,  
                chunk_overlap=300, 
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
        else:
            print(f"File not found: {path}")
            
    if not all_chunks:
        print("No documents were processed.")
        return None
        
    print(f"Total chunks created: {len(all_chunks)}")
    
    # 3. Create Vector DB
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
