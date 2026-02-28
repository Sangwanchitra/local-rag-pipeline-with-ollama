from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Using same config as user's main app
OLLAMA_BASE_URL = "http://localhost:11434"
# Using an open-weight model as specified
OLLAMA_MODEL = "llama3.2:3b"

def get_generator_chain(retriever):
    """
    Builds the generation chain using the retriever and local LLM.
    """
    
    # Initialize Ollama LLM
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0, # Deterministic answers
    )
    
    # Strict Prompt Template
    template = """
    SYSTEM:
    You are a helpful assistant that answers strictly from context.
    If the answer is not found in the context, say "I don't know."
    Answer clearly and cite source pages when providing facts.
    
    USER:
    Question: {question}
    
    CONTEXT:
    {context}
    
    Answer clearly and cite source pages.
    """
    
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        # Merge chunks and extract metadata for context window
        formatted = []
        for d in docs:
            page = d.metadata.get("page", "Unknown")
            source = d.metadata.get("source", "Unknown")
            section = d.metadata.get("section", "Unknown")
            
            # Format chunk with its metadata
            formatted.append(f"[Source: {source}, Page: {page}, Section: {section}]\n{d.page_content}\n")
            
        return "\n".join(formatted)
        
    # Build RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

if __name__ == "__main__":
    pass
