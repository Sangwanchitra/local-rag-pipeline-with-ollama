# BNS-LegalBot ⚖️

This is a production-ready, highly constrained Retrieval-Augmented Generation (RAG) system built entirely on local/open-source components (except the LLM, which is open-weight running via Ollama).
![image alt](https://github.com/Sangwanchitra/local-rag-pipeline-with-ollama/blob/668c98d0ae4be0caa7f11453fe4c31dcf8e664cb/IMG-20260301-WA0011.jpg)

## Architecture

1. **Frontend**: Streamlit
2. **Document Loading**: PyMuPDF (`fitz`) with heuristic header detection.
3. **Chunking**: LangChain `RecursiveCharacterTextSplitter`.
4. **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (Local).
5. **Vector DB**: ChromaDB (Local).
6. **Generation**: `ChatOllama` utilizing local open-weight models (e.g., `llama3:8b`).

## Chunking Strategy

The system uses a "smart" semantic-aware chunking strategy:

- First, PyMuPDF parses the PDF, extracting font sizes and weights to guess section headers.
- Paragraphs originating from the same page and section are combined.
- `RecursiveCharacterTextSplitter` chunk overlaps around semantic sentence boundaries (`\n\n`, `\n`, ` `).
- **Chunk Size**: ~2500 characters (~500-800 tokens).
- **Overlap**: ~300 characters (10-15%).
- **Metadata stored**: Filename, Page Number, Section Title.

## Why `all-MiniLM-L6-v2`?

It is a fast, highly optimized open-source embedding model that runs purely locally. It provides an excellent balance between speed and vector quality for general English text, without requiring external API calls or GPUs.

## Setup Instructions

### 1. Pre-requisites

- Python 3.10+
- Install Ollama for your OS: [https://ollama.com/](https://ollama.com/)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull the Local Model

Ensure Ollama is running in the background, then pull the required model:

```bash
ollama run llama3.2:3b
# Press Ctrl+D once it starts to exit the prompt, the model is now downloaded.
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

## How to use the App

1. Open the UI (usually `http://localhost:8501`).
2. Upload exactly 2 PDF files in document folder. 
3. Wait for the embeddings to be generated and stored locally in `data/chroma_db`.
4. Ask a question. The system will retrieve the top 5 relevant chunks and generate a grounded answer citing the source file and page numbers.

## Evaluation

A standalone script is provided to measure the effectiveness of the retrieval and generation pipeline.

**To run the evaluation:**

1. Open `evaluation.py`.
2. Modify the `EVAL_QA` list to include 10-15 actual questions based on your specific uploaded PDFs, noting the expected page number for the answer.
3. Run the script:

```bash
python evaluation.py
```

**Metrics measured:**

- **Retrieval Precision@k:** Checks if the expected page was within the top-5 retrieved chunks.
- **Citation Hit Rate:** Checks if the LLM correctly cited the expected page number in its generated response.
