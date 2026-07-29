# CorpusMind

CorpusMind is a portfolio-grade research intelligence platform for analyzing academic papers, technical reports, and document collections. It combines a Django application backend, a FastAPI AI service, and a Vue frontend to support multimodal document ingestion, advanced retrieval-augmented generation, and evidence-grounded summaries.

## Purpose

The project is designed to demonstrate production-oriented software engineering and applied AI architecture. It moves beyond basic summarization by reading document text, tables, and OCR-derived image content, then indexing the material for question answering with citations.

## Architecture

- `backend/`: Django and Django REST Framework API for document orchestration, persistence, and client-facing endpoints.
- `ai_service/`: FastAPI service for AI workloads, including parsing, OCR, summarization, retrieval, and answer synthesis.
- `frontend/`: Vue 3 application built with Vite for document upload, analysis review, and grounded question answering.
- `src/`: Existing NLP modules retained for ingestion, preprocessing, summarization, insights, evaluation, and export support.

## Core Features

- Multimodal document ingestion for PDFs and plain text.
- Table extraction from PDFs through `pdfplumber`.
- OCR support for scanned pages and embedded visual content through `pdf2image` and `pytesseract`.
- Hybrid summarization that combines extractive filtering with abstractive generation.
- Advanced RAG pipeline with TF-IDF retrieval, BM25-style scoring, optional sentence-transformer semantic retrieval, and Maximal Marginal Relevance diversification.
- Evidence-grounded question answering with citations to pages, tables, and OCR blocks.
- Keyword and insight extraction through KeyBERT when available.
- Django REST API for upload, document listing, and question answering.
- Vue frontend for a polished research analysis workspace.

## Technology Stack

- Backend: Django, Django REST Framework, SQLite for local development.
- AI service: FastAPI, scikit-learn, Hugging Face Transformers, sentence-transformers, pdfplumber, pytesseract.
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

## API Overview

Upload and analyze a document:

```bash
POST /api/documents/upload/
```

Ask a question about an indexed document:

```bash
POST /api/documents/{ai_document_id}/ask/
```

List recent documents:

```bash
GET /api/documents/
```

## Notes

Large local models may require significant memory and initial download time. The AI service includes fallback paths so document parsing, indexing, and basic summarization can still operate when transformer models are unavailable.
