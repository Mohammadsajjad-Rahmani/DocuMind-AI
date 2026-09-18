# DocuMind AI 📄🧠

An intelligent Document Summarization and Question-Answering (RAG) system built with FastAPI, Streamlit, and Sentence Transformers.

## 🌟 Features
- **PDF & Text Extraction**: Robust text parsing from PDF documents using `pdfplumber`.
- **Extractive Summarization**: Generates concise summaries by computing sentence-to-document similarity vectors.
- **Semantic Question-Answering (RAG)**: Answers user questions directly from document context using dense vector search (`all-MiniLM-L6-v2`) and Cosine Similarity.
- **Modern REST API**: Modular API architecture powered by FastAPI and Pydantic schemas.
- **Interactive UI**: Clean user dashboard built with Streamlit.

## 📈 Precision & Performance Evolution
To deliver high Q&A accuracy and reliable text extraction, the system underwent key optimization iterations:
1. **Document Parsing**: Upgraded from basic `pypdf` to `pdfplumber` to accurately capture complex layouts, fonts, and inline tables.
2. **From Keyword Matching to Semantic Embeddings**: Replaced initial TF-IDF vectorization with `sentence-transformers` (`all-MiniLM-L6-v2`). This transition enabled deep semantic understanding (handling synonyms and contextual meaning) rather than strict word overlap.
3. **Sentence-Level Chunking**: Refined text chunking to sentence boundaries with cosine similarity thresholds (>0.25), returning highly precise, targeted answer sentences instead of bloated paragraphs.

## 🛠️ Tech Stack
- **Backend**: Python 3.11+, FastAPI, Uvicorn, Pydantic
- **Frontend**: Streamlit
- **ML / NLP Engine**: `sentence-transformers` (`all-MiniLM-L6-v2`), `scikit-learn`, `pdfplumber`
- **Package Manager**: `uv`

## 🚀 Getting Started

### Prerequisites
Make sure you have `uv` installed.

### 1. Clone & Setup Environment
git clone https://github.com/Mohammadsajjad-Rahmani/DocuMind-AI.git
cd documind-ai
uv sync

### 2. Run Backend Server
uv run uvicorn main:app --reload
The API will be available at http://localhost:8000 (Swagger docs at http://localhost:8000/docs).

### 3. Run Streamlit UI
In a separate terminal, execute:
uv run streamlit run app_ui.py

## 🔌 API Endpoints
- POST /extract-text: Uploads a PDF file and extracts raw text.
- POST /summarize: Accepts raw text and returns top summary sentences.
- POST /query: Takes document text and a user question, returning the most semantically relevant answer sentence.