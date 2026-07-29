# CorpusMind AI Service

The AI service is a FastAPI application responsible for multimodal document understanding, summarization, and retrieval-augmented generation.

## Responsibilities

- Extract text, tables, and OCR text from uploaded research documents.
- Build a BERT embedding retrieval index with BM25-style lexical grounding, modality-aware ranking, and MMR diversification.
- Generate summaries and evidence-grounded answers with citations.

## Run

```bash
uvicorn app.main:app --reload --port 8001
```
