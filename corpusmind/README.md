# CorpusMind

CorpusMind is a multimodal research intelligence platform for analyzing research papers, technical reports, and dense document collections. The application extracts text, tables, and OCR-readable visual content from uploaded files, builds a BERT-based retrieval index, generates structured summaries, and answers user questions with ranked evidence and citations.

The project is designed as a portfolio-grade full-stack AI application. It demonstrates frontend product design, backend service orchestration, multimodal document processing, retrieval-augmented generation, containerized deployment, and practical fallback behavior for local AI environments.

## Table of Contents

- [Overview](#overview)
- [Key Capabilities](#key-capabilities)
- [System Architecture](#system-architecture)
- [AI Pipeline](#ai-pipeline)
- [Backend Features](#backend-features)
- [Frontend Experience](#frontend-experience)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Docker Setup](#docker-setup)
- [Local Development Setup](#local-development-setup)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Operational Notes](#operational-notes)
- [Portfolio Highlights](#portfolio-highlights)
- [Future Improvements](#future-improvements)

## Overview

Many research summarization projects stop at extracting plain text and producing a generic summary. CorpusMind is built around a more realistic research workflow:

1. A user uploads a PDF, text file, markdown file, or CSV.
2. The AI service extracts document text, tables, and OCR content from scanned or image-heavy pages.
3. The document is converted into modality-aware chunks.
4. A BERT embedding index is created for dense retrieval.
5. BM25-style lexical grounding and modality-aware scoring improve ranking for exact technical terms, tables, images, figures, and metrics.
6. The system generates an executive summary, keywords, findings, and a grounded question-answering interface.
7. The frontend presents the results as an interactive research analysis workspace rather than a basic CRUD dashboard.

CorpusMind is intentionally separated into three services:

- Django handles application orchestration, persistence, and public API endpoints.
- FastAPI handles AI workloads and document intelligence.
- Vue provides the user-facing analysis console.

This separation keeps the system easier to reason about, scale, and present in a technical interview.

## Key Capabilities

- Upload and analyze research documents through a Django REST API.
- Parse PDFs and text-based documents.
- Extract tables from PDFs using `pdfplumber`.
- Run OCR on PDF pages using `pdf2image`, Poppler, and Tesseract.
- Build a dense BERT retrieval index using `sentence-transformers`.
- Rank retrieved chunks with BERT similarity, BM25-style lexical grounding, modality-aware boosts, and Maximal Marginal Relevance.
- Generate summaries using a hybrid extractive and abstractive workflow.
- Extract semantic keywords when KeyBERT is available.
- Ask questions about indexed documents and receive grounded answers with citations.
- Inspect retrieved evidence in the frontend.
- View service readiness and AI capability metadata from the frontend.
- Run the full stack with Docker Compose.

## System Architecture

CorpusMind follows a service-oriented architecture:

```text
Vue Frontend
    |
    | HTTP /api
    v
Django Backend
    |
    | Internal HTTP
    v
FastAPI AI Service
    |
    | PDF parsing, OCR, embeddings, summarization, retrieval
    v
Document Intelligence Pipeline
```

### Frontend

The Vue frontend is a single-page application built with Vite. It provides:

- File upload workflow.
- Analysis status indicators.
- Executive summary view.
- Key findings view.
- Question-answering interface.
- Evidence and citation browser.
- Retrieval telemetry such as embedding model, retrieval strategy, indexed chunk count, table count, and OCR block count.

### Backend

The Django backend is the application-facing API layer. It stores document analysis records and delegates AI-heavy work to the FastAPI service. It also exposes a system intelligence endpoint so the UI can display service status and backend capabilities.

### AI Service

The FastAPI service performs document understanding and AI tasks. It keeps an in-memory document retrieval pipeline for analyzed files and exposes endpoints for analysis and question answering.

## AI Pipeline

The AI pipeline is implemented primarily in:

- `ai_service/app/ingestion.py`
- `ai_service/app/rag.py`
- `ai_service/app/main.py`
- `ai_service/nlp/`

### 1. Multimodal Ingestion

The ingestion layer accepts uploaded files and routes them by file type.

For PDFs, it attempts:

- Text extraction with `pdfplumber`.
- Table extraction with `pdfplumber.extract_tables`.
- OCR extraction using `pdf2image` and `pytesseract`.
- PDF text fallback using `pypdf`.

For text-like files, it decodes the payload and treats the content as a plain document.

The output is normalized into a structured internal document representation:

- Pages
- Tables
- OCR blocks
- Metadata
- Full extracted text

### 2. Chunking

Documents are split into overlapping chunks. Chunks retain source metadata such as:

- Source modality
- Page number
- Extraction method
- Word offset

This allows the frontend to display citations and retrieved context in a useful way.

### 3. BERT Retrieval

CorpusMind uses `sentence-transformers/all-MiniLM-L6-v2` by default for dense retrieval. Each document chunk is embedded into a normalized vector space. User questions are embedded into the same vector space and ranked by cosine similarity.

If the embedding model is unavailable in a constrained local environment, the service falls back to a deterministic dense embedding method. This fallback is not a replacement for a true transformer model, but it keeps the pipeline operational for demos and development.

### 4. Lexical Grounding

BERT retrieval is strong for semantic similarity, but exact research terms, abbreviations, measurements, and table labels can still matter. CorpusMind adds a BM25-style score to improve retrieval for exact terminology.

The final retrieval score combines:

- BERT dense similarity
- BM25-style lexical relevance
- Modality-aware boosts

### 5. Modality-Aware Ranking

Queries that mention tables, rows, columns, values, metrics, results, figures, diagrams, charts, or images receive ranking boosts for the relevant source modality.

This helps questions such as:

- "Which results are supported by tables?"
- "What does the figure show?"
- "Which metric improved the most?"

### 6. Maximal Marginal Relevance

CorpusMind applies Maximal Marginal Relevance after initial ranking. This reduces duplicate context and improves evidence diversity by balancing relevance against similarity to already selected chunks.

### 7. Summarization and Answering

The summarization workflow uses retained NLP modules under `ai_service/nlp/`. It supports hybrid summarization and graceful fallback behavior. The question-answering flow synthesizes an answer from retrieved evidence and returns citations for the selected chunks.

## Backend Features

The Django backend is more than a simple CRUD layer. It coordinates the application workflow and handles communication between the frontend and AI service.

Implemented backend features include:

- Document upload endpoint.
- AI service delegation.
- Analysis result persistence.
- Document list endpoint.
- Grounded question-answering endpoint.
- AI service health and capability endpoint.
- Request validation for empty questions.
- Failure handling when the AI service is unavailable.
- Environment-based configuration for deployment.
- SQLite persistence for local development and Docker volumes.

The backend app is located in:

```text
backend/documents/
```

Important files:

- `models.py`: stores analyzed research documents.
- `views.py`: orchestrates upload, analysis, status, and question-answering requests.
- `serializers.py`: defines document API serialization.
- `urls.py`: exposes API routes.

## Frontend Experience

The frontend is designed to show product and interface skills, not only backend functionality.

The Vue interface includes:

- A research command center layout.
- Document upload panel.
- Backend and AI service readiness indicators.
- Overview, Ask, and Evidence workspace views.
- Metric tiles for BERT chunks, parsed tables, OCR blocks, and keywords.
- Executive summary panel.
- Key findings panel.
- Prompt suggestions for common research questions.
- Question-answering input with loading state.
- Citation list with score and source modality.
- Retrieved evidence viewer.
- Responsive layout for smaller screens.
- Error banners for failed analysis or unavailable AI service.

The main frontend files are:

- `frontend/src/App.vue`
- `frontend/src/styles.css`
- `frontend/src/services/api.js`

## Technology Stack

### Backend

- Python
- Django
- Django REST Framework
- SQLite for local development
- Gunicorn for containerized serving
- Requests for internal service communication

### AI Service

- Python
- FastAPI
- Uvicorn
- Hugging Face Transformers
- Sentence Transformers
- KeyBERT
- NLTK
- NumPy
- pdfplumber
- pypdf
- pdf2image
- pytesseract
- Tesseract OCR
- Poppler

### Frontend

- Vue 3
- Vite
- Axios
- lucide-vue-next
- CSS Grid and responsive custom CSS
- Nginx for production container serving

### DevOps

- Docker
- Docker Compose
- Service-specific Dockerfiles
- Persistent Docker volumes for model cache and SQLite data
- Makefile command shortcuts

## Project Structure

```text
corpusmind/
  ai_service/
    app/
      ingestion.py
      main.py
      rag.py
      schemas.py
    nlp/
      abstractive.py
      analysis.py
      evaluate.py
      export.py
      extractive.py
      hybrid.py
      ingest.py
      postprocess.py
      preprocess.py
    Dockerfile
    README.md

  backend/
    corpusmind_backend/
      settings.py
      urls.py
      asgi.py
      wsgi.py
    documents/
      models.py
      serializers.py
      urls.py
      views.py
      migrations/
    Dockerfile
    manage.py

  frontend/
    src/
      App.vue
      main.js
      styles.css
      services/
        api.js
    Dockerfile
    nginx.conf
    package.json
    vite.config.js

  docker-compose.yml
  Makefile
  requirements-ai.txt
  requirements-backend.txt
  requirements.txt
```

## Docker Setup

Docker is the recommended way to run the full application stack.

From the `corpusmind` directory:

```bash
docker compose up --build
```

Or use the Makefile:

```bash
make up
```

### Docker Services

| Service | Description | Port |
| --- | --- | --- |
| `frontend` | Vue production build served by Nginx | `5173` |
| `backend` | Django REST API served by Gunicorn | `8000` |
| `ai-service` | FastAPI AI service for OCR, embeddings, summarization, and RAG | `8001` |

### Docker URLs

- Frontend: `http://localhost:5173`
- Django API: `http://localhost:8000/api`
- AI service health: `http://localhost:8001/health`

### Docker Volumes

The Compose setup defines:

- `ai-model-cache`: stores Hugging Face model files across container rebuilds.
- `backend-data`: stores the SQLite database for local Docker runs.

### Docker Notes

The AI image installs Tesseract and Poppler so OCR can run inside the container. The first AI run may take longer because transformer models may need to download and cache.

## Local Development Setup

Local development is useful when working on one service at a time.

### 1. Python Dependencies

From the `corpusmind` directory:

```bash
pip install -r requirements.txt
```

For smaller service-specific installs:

```bash
pip install -r requirements-backend.txt
pip install -r requirements-ai.txt
```

### 2. OCR System Dependencies

For Ubuntu or Debian-based systems:

```bash
sudo apt-get install tesseract-ocr poppler-utils
```

### 3. Start the AI Service

```bash
cd ai_service
uvicorn app.main:app --reload --port 8001
```

### 4. Start the Django Backend

In a separate terminal:

```bash
cd backend
python manage.py migrate
python manage.py runserver 8000
```

### 5. Start the Frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend uses `http://localhost:8000/api` by default. For a different backend URL, set:

```bash
VITE_API_URL=http://localhost:8000/api
```

## API Reference

All public application endpoints are exposed by the Django backend under `/api`.

### Intelligence Status

```http
GET /api/intelligence/status/
```

Returns backend status, AI service health, and supported capabilities.

Example response:

```json
{
  "backend": "ok",
  "ai_service": {
    "status": "ok",
    "service": "corpusmind-ai"
  },
  "capabilities": [
    "document_upload_orchestration",
    "bert_retrieval",
    "ocr_ingestion",
    "table_extraction",
    "grounded_question_answering",
    "citation_ranking"
  ]
}
```

### Upload and Analyze Document

```http
POST /api/documents/upload/
Content-Type: multipart/form-data
```

Form field:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `file` | File | Yes | PDF, text, markdown, or CSV document |

The backend forwards the file to the AI service, stores the analysis result, and returns document metadata plus analysis output.

### List Documents

```http
GET /api/documents/
```

Returns recently analyzed documents stored by Django.

### Ask a Question

```http
POST /api/documents/{ai_document_id}/ask/
Content-Type: application/json
```

Request body:

```json
{
  "question": "Which results are supported by tables?",
  "top_k": 8
}
```

Response fields include:

- `answer`: generated answer grounded in retrieved evidence.
- `citations`: ranked citation metadata.
- `contexts`: retrieved evidence chunks.

## AI Service Endpoints

The FastAPI service is intended to be called by Django, but it can also be used directly during development.

### Health

```http
GET /health
```

### Analyze

```http
POST /analyze
Content-Type: multipart/form-data
```

### Ask

```http
POST /documents/{document_id}/ask
Content-Type: application/json
```

## Configuration

### Django Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `DJANGO_SECRET_KEY` | `development-only-corpusmind-key` | Secret key for local development |
| `DJANGO_DEBUG` | `1` | Enables Django debug mode |
| `DJANGO_ALLOWED_HOSTS` | `*` | Comma-separated allowed hosts |
| `DJANGO_DB_PATH` | `backend/db.sqlite3` | SQLite database path |
| `AI_SERVICE_URL` | `http://localhost:8001` | URL used by Django to reach FastAPI |

### Frontend Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `VITE_API_URL` | `http://localhost:8000/api` | Django API base URL |

### AI Service Environment Variables

| Variable | Description |
| --- | --- |
| `TRANSFORMERS_CACHE` | Cache directory for transformer model files |
| `HF_HOME` | Hugging Face home/cache directory |

## Operational Notes

- Transformer-based models can be large. The first run may require network access and additional time for downloads.
- OCR requires Tesseract and Poppler. These are installed in the AI Docker image.
- The AI service stores active RAG pipelines in memory. Restarting the AI service clears in-memory indexes.
- SQLite is used for local development. A production deployment should use PostgreSQL or another managed database.
- The deterministic dense embedding fallback keeps local development functional when the BERT model is unavailable, but best retrieval quality requires sentence-transformer embeddings.
- The Vue frontend build depends on npm package installation. If package registry access is slow or unavailable, frontend build verification may be delayed.

## Makefile Commands

From the `corpusmind` directory:

| Command | Description |
| --- | --- |
| `make up` | Build and start the Docker Compose stack |
| `make build` | Build Docker images |
| `make down` | Stop Docker services |
| `make logs` | Follow Docker service logs |
| `make backend-check` | Run Django system checks locally |
| `make ai-health` | Call the FastAPI health endpoint |

## Portfolio Highlights

CorpusMind is structured to demonstrate practical engineering depth:

- Full-stack architecture with separate frontend, backend, and AI services.
- Multimodal document processing with text, tables, and OCR.
- BERT-based retrieval rather than simple keyword search.
- Retrieval scoring that combines semantic similarity, lexical grounding, and source modality.
- Evidence-grounded question answering with citations.
- Non-trivial backend orchestration beyond CRUD operations.
- Responsive Vue interface with multiple workspace views and retrieval telemetry.
- Dockerized deployment with service-specific containers and persistent volumes.
- Graceful failure handling for unavailable AI services and missing local models.

## Future Improvements

Planned or natural extensions include:

- Persisting vector indexes in a dedicated vector database such as Qdrant, Weaviate, Milvus, or pgvector.
- Replacing in-memory RAG state with durable document indexes.
- Adding background jobs with Celery or Django Q for long-running document analysis.
- Adding user accounts, workspaces, and document collections.
- Supporting comparative analysis across multiple papers.
- Adding PDF page previews and citation-to-page navigation.
- Adding streaming answer generation.
- Adding evaluation dashboards for retrieval quality and summary quality.
- Adding production PostgreSQL, object storage, and observability.

## License

No license has been specified yet. Add a license before publishing or distributing the project.
