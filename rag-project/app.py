import streamlit as st
import os
import shutil
from ingest import ingest_documents
from retriever import get_retriever
from generator import get_generator_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

CHROMA_PATH = "data/chroma_db"

st.set_page_config(page_title="BNS-LegalBot ⚖️", layout="wide")

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "index_built" not in st.session_state:
        st.session_state.index_built = os.path.exists(CHROMA_PATH)
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chain" not in st.session_state:
        st.session_state.chain = None

def local_css():
    st.markdown("""
    <style>
        .context-box { background-color: #f1f3f4; padding: 10px; border-radius: 5px; margin-bottom: 5px; }
        .source-tag { font-size: 0.8rem; font-weight: bold; color: #1f77b4; }
        /* Hide main menu and footer for a cleaner look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def load_documents_from_data_folder(data_dir: str = "./data/") -> list[Document]:
    if not os.path.exists(data_dir):
        st.error("Data folder not found.")
        st.stop()
        
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        st.warning("No PDF files found in data folder.")
        st.stop()

    all_documents = []
    
    for filename in pdf_files:
        filepath = os.path.join(data_dir, filename)
        try:
            loader = PyPDFLoader(filepath)
            documents = loader.load()
            all_documents.extend(documents)
        except Exception as e:
            st.error(f"Failed to load {filename}: {str(e)}")
            st.stop()
            
    return all_documents

def sidebar():
    with st.sidebar:
        st.header("Session Management")
        
        if st.button("New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

def main():
    local_css()
    init_session_state()
    sidebar()
    
    st.title("BNS-LegalBot ⚖️")
    
    # Auto-initialize backend if index does not exist
    if not st.session_state.index_built:
        with st.spinner("Initializing knowledge base for the first time... This may take a few moments."):
            documents = load_documents_from_data_folder(data_dir="./data/")
            db = ingest_documents(documents)
            if db:
                st.session_state.index_built = True
                st.session_state.retriever = get_retriever()
                st.session_state.vectorstore = db
                st.session_state.chain = get_generator_chain(st.session_state.retriever)
                st.success("Knowledge base initialized successfully!")
                st.rerun()
            else:
                st.error("Failed to initialize knowledge base. Please check the data folder.")
                return
    else:
        # Load retriever and chain if index exists but not loaded in session
        if st.session_state.chain is None:
            with st.spinner("Loading legal knowledge base..."):
                st.session_state.retriever = get_retriever()
                st.session_state.chain = get_generator_chain(st.session_state.retriever)

    # Initial Welcome Message
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("Hello! 👋 I'm **BNS LegalBot** — your AI assistant for Bharatiya Nyaya Sanhita queries.\n\nHow can I assist you today?")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "docs" in message and message["docs"]:
                with st.expander("References"):
                    for d in message["docs"]:
                        st.markdown(f"""
                        <div class="context-box">
                            <span class="source-tag">📄 Source: {d.metadata.get('source')} (Page {d.metadata.get('page')})</span><br>
                            <b>Section:</b> {d.metadata.get('section', 'Unknown')}<br>
                            {d.page_content}
                        </div>
                        """, unsafe_allow_html=True)

    if query := st.chat_input("Ask a question about the Bharatiya Nyaya Sanhita..."):
        # Add user question
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing legal documents..."):
                # Get context
                docs = st.session_state.retriever.invoke(query)
                
                # Generate answer using chain
                rag_chain = st.session_state.chain
                answer = rag_chain.invoke(query)
                
                st.markdown(answer)
                
                if docs:
                    with st.expander("References"):
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
