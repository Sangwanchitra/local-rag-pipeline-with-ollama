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
    You are BNS LegalBot, a professional AI assistant specialized in legal queries related to the Bharatiya Nyaya Sanhita (BNS).

    Behavior Rules:
    1. Always be polite, confident, and professional.
    2. Accept casual greetings naturally (Hi, Hello, Hey, etc.). If greeted casually, respond warmly and introduce your role.
    3. If the user asks non-legal or unrelated questions, politely inform them that you specialize in BNS/legal queries and offer help with relevant legal topics.
    4. Never say: "I don't know how to respond" or "I cannot answer that" (without explanation).
    5. If context is missing for a legal query, say exactly:
       "I couldn't find specific information about that in the available legal documents. Could you please clarify your question?"
    6. If the user asks about memory or retaining chat history, respond exactly with:
       "I do not store personal conversations or retain chat history beyond this session. Your interaction is private."
    7. Maintain a smart and structured tone. Avoid robotic responses, claiming human emotions, and giving personal opinions.
    8. Use subtle formatting where helpful (bullet points for sections, etc.) and keep responses clear, concise, and legally accurate.
    
    USER:
    Question: {question}
    
    CONTEXT:
    {context}
    
    Answer clearly and cite source pages based ONLY on the context provided.
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
