import os
from retriever import get_retriever
from generator import get_generator_chain

# Sample Evaluation Questions. 
# INSTRUCTIONS: Replace these with actual questions from your uploaded PDFs.
# expected_page is the page number where the answer should be found.
EVAL_QA = [
    {"q": "What is the main topic of the first document?", "expected_page": 1},
    {"q": "How does the system handle cross-section queries?", "expected_page": 5},
    {"q": "What embedding model is used?", "expected_page": 2},
    # Add up to 10-15 questions here based on your dataset
]

def run_evaluation():
    print("Starting RAG Evaluation...")
    retriever = get_retriever()
    if not retriever:
        print("Error: Retriever not initialized. Did you build the index?")
        return
        
    rag_chain = get_generator_chain(retriever)
    
    total_q = len(EVAL_QA)
    retrieval_hits = 0
    citation_hits = 0
    
    for i, item in enumerate(EVAL_QA):
        question = item["q"]
        expected_page = item["expected_page"]
        
        print(f"\n--- Question {i+1}/{total_q} ---")
        print(f"Q: {question}")
        
        # 1. Test Retrieval (Precision@k)
        docs = retriever.invoke(question)
        retrieved_pages = [d.metadata.get("page") for d in docs]
        
        # Did we retrieve the block with the expected page?
        if expected_page in retrieved_pages:
            retrieval_hits += 1
            print(f"✅ Retrieval HIT (Found expected page {expected_page} in top-k)")
        else:
            print(f"❌ Retrieval MISS (Expected page {expected_page}, got {retrieved_pages})")
            
        # 2. Test Generation & Citation
        answer = rag_chain.invoke(question)
        print(f"A: {answer}")
        
        # Simple heuristic to check if the generated answer cites the correct page
        if str(expected_page) in answer:
            citation_hits += 1
            print(f"✅ Citation HIT (Model cited page {expected_page})")
        else:
            print(f"❌ Citation MISS (Model did not clearly cite page {expected_page})")
            
    # Report Metrics
    print("\n==============================")
    print("EVALUATION RESULTS")
    print("==============================")
    print(f"Total Questions: {total_q}")
    print(f"Retrieval Precision@k Hit Rate: {retrieval_hits / total_q * 100:.1f}%")
    print(f"Correct Citation Hit Rate: {citation_hits / total_q * 100:.1f}%")

if __name__ == "__main__":
    run_evaluation()
