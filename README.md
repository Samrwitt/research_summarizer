# ResearchGuide

ResearchGuide is research intelligence platform for analyzing academic papers, technical reports, and document collections. It combines a Django application backend, a FastAPI AI service, and a Vue frontend to support multimodal document ingestion, advanced retrieval-augmented generation, and evidence-grounded summaries.

## Architecture

- `backend/`: Django and Django REST Framework API for document orchestration, persistence, and client-facing endpoints.
- `ai_service/`: FastAPI service for AI workloads, including parsing, OCR, summarization, retrieval, and answer synthesis.
- `frontend/`: Vue 3 application built with Vite for document upload, analysis review, and grounded question answering.
- `ai_service/nlp/`: Reusable NLP modules for preprocessing, summarization, insights, evaluation, and export support.

## Core Features

- Multimodal document ingestion for PDFs and plain text.
- Table extraction from PDFs through `pdfplumber`.
- OCR support for scanned pages and embedded visual content through `pdf2image` and `pytesseract`.
- Hybrid summarization that combines extractive filtering with abstractive generation.
- Advanced RAG pipeline with BERT sentence embeddings, BM25-style lexical grounding, modality-aware ranking, and Maximal Marginal Relevance diversification.
- Evidence-grounded question answering with citations to pages, tables, and OCR blocks.
- Keyword and insight extraction through KeyBERT when available.
- Django REST API for upload, document listing, and question answering.
- Vue frontend for a polished research analysis workspace.

## Technology Stack

- Backend: Django, Django REST Framework, SQLite for local development.
- AI service: FastAPI, Hugging Face Transformers, sentence-transformers, pdfplumber, pytesseract.
- Frontend: Vue 3, Vite, Axios, lucide-vue-next.

## Local Setup

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

For OCR support, install system dependencies:

```bash
sudo apt-get install tesseract-ocr poppler-utils
```

Start the AI service:

```bash
cd ai_service
uvicorn app.main:app --reload --port 8001
```

Start the Django backend:

```bash
cd backend
python manage.py migrate
python manage.py runserver 8000
```

Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

The frontend expects the Django API at `http://localhost:8000/api`. Override this with `VITE_API_URL` when needed.

## Docker Setup

Build and start the full application stack:

```bash
docker compose up --build
```

The same command is available through the included Makefile:

```bash
make up
```

The services are exposed as follows:

- Frontend: `http://localhost:5173`
- Django API: `http://localhost:8000/api`
- FastAPI AI service: `http://localhost:8001`

The Docker setup includes:

- A FastAPI AI container with OCR system dependencies, including Tesseract and Poppler.
- A Django API container that runs database migrations before starting Gunicorn.
- A Vue production build served by Nginx, with `/api` proxied to the Django backend.
- A persistent model cache volume for Hugging Face assets.
- A persistent SQLite data volume for local Django data.

## Notes

Large local models may require significant memory and initial download time. The AI service includes fallback paths so document parsing, indexing, and basic summarization can still operate when transformer models are unavailable.
