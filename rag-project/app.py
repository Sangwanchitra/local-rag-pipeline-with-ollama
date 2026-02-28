import streamlit as st
import os
import shutil
from ingest import ingest_documents
from retriever import get_retriever
from generator import get_generator_chain

DATA_DIR = "data/documents"
CHROMA_PATH = "data/chroma_db"

st.set_page_config(page_title="RAG App", layout="wide")

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "index_built" not in st.session_state:
        st.session_state.index_built = os.path.exists(CHROMA_PATH)
    if "retriever" not in st.session_state:
        if st.session_state.index_built:
            st.session_state.retriever = get_retriever()

def local_css():
    st.markdown("""
    <style>
        .context-box { background-color: #f1f3f4; padding: 10px; border-radius: 5px; margin-bottom: 5px; }
        .source-tag { font-size: 0.8rem; font-weight: bold; color: #1f77b4; }
    </style>
    """, unsafe_allow_html=True)

def sidebar():
    with st.sidebar:
        st.header("Document Upload")
        st.write("Upload 2 PDF files (30-50 pages each)")
        
        uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
        
        if st.button("Build Index"):
            if not uploaded_files:
                st.error("Please upload PDF files first.")
                return
                
            with st.spinner("Processing documents (Chunking & Embedding)..."):
                # Clean up existing data directory
                if os.path.exists(DATA_DIR):
                    shutil.rmtree(DATA_DIR)
                os.makedirs(DATA_DIR, exist_ok=True)
                
                # Clean up Chroma index
                if os.path.exists(CHROMA_PATH):
                    shutil.rmtree(CHROMA_PATH)
                    
                filepaths = []
                for f in uploaded_files:
                    path = os.path.join(DATA_DIR, f.name)
                    with open(path, "wb") as file_out:
                        file_out.write(f.getbuffer())
                    filepaths.append(path)
                
                # Ingest pipeline
                db = ingest_documents(filepaths)
                if db:
                    st.session_state.index_built = True
                    st.session_state.retriever = get_retriever()
                    st.success("Index built successfully!")
                else:
                    st.error("Failed to process documents.")

def main():
    local_css()
    init_session_state()
    sidebar()
    
    st.title("Local RAG System with Open-Weight LLM")
    
    if not st.session_state.index_built:
        st.info("Please upload documents and build the index on the left sidebar to start asking questions.")
        return
        
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "docs" in message:
                with st.expander("Retrieved Context Preview"):
                    for d in message["docs"]:
                        st.markdown(f"""
                        <div class="context-box">
                            <span class="source-tag">📄 Source: {d.metadata.get('source')} (Page {d.metadata.get('page')})</span><br>
                            <b>Section:</b> {d.metadata.get('section', 'Unknown')}<br>
                            {d.page_content}
                        </div>
                        """, unsafe_allow_html=True)

    if query := st.chat_input("Ask a question about your documents..."):
        # Add user question
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            with st.spinner("Retrieving local context & generating response..."):
                retriever = st.session_state.retriever
                # Get context
                docs = retriever.invoke(query)
                
                # Generate answer using chain
                rag_chain = get_generator_chain(retriever)
                answer = rag_chain.invoke(query)
                
                st.markdown(answer)
                
                with st.expander("Retrieved Context Preview"):
                    for d in docs:
                        st.markdown(f"""
                        <div class="context-box">
                            <span class="source-tag">📄 Source: {d.metadata.get('source')} (Page {d.metadata.get('page')})</span><br>
                            <b>Section:</b> {d.metadata.get('section', 'Unknown')}<br>
                            {d.page_content}
                        </div>
                        """, unsafe_allow_html=True)
                        
        st.session_state.messages.append({"role": "assistant", "content": answer, "docs": docs})

if __name__ == "__main__":
    main()
