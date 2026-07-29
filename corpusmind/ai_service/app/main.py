from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.ingestion import ingest_upload
from app.rag import AdvancedRAGPipeline
from app.schemas import AnalyzeResponse, AskRequest, AskResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from src.analysis import extract_insights
    from src.hybrid import summarize_hybrid
    from src.postprocess import generate_bullet_points
    from src.preprocess import preprocess
except Exception:
    extract_insights = None
    summarize_hybrid = None
    generate_bullet_points = None
    preprocess = None


app = FastAPI(title="CorpusMind AI Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PIPELINES: dict[str, AdvancedRAGPipeline] = {}
SUMMARIES: dict[str, str] = {}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "corpusmind-ai"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    document_id = hashlib.sha256(payload).hexdigest()[:16]
    try:
        document = ingest_upload(file.filename or "document", payload, file.content_type)
        prepared = _preprocess_for_summary(document)
        summary = _summarize(prepared)
        bullets = generate_bullet_points(summary, num_bullets=6) if generate_bullet_points else _bullets(summary)
        insights = _extract_insights(prepared["clean_text"])
        pipeline = AdvancedRAGPipeline()
        rag_stats = pipeline.build(document)
        PIPELINES[document_id] = pipeline
        SUMMARIES[document_id] = summary
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AnalyzeResponse(
        document_id=document_id,
        title=document.title,
        summary=summary,
        bullets=bullets,
        insights=insights,
        rag=rag_stats,
        metadata=document.metadata,
    )


@app.post("/documents/{document_id}/ask", response_model=AskResponse)
def ask(document_id: str, request: AskRequest) -> AskResponse:
    pipeline = PIPELINES.get(document_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Document is not indexed in the AI service")
    result = pipeline.answer(request.question, summary=SUMMARIES.get(document_id), top_k=request.top_k)
    return AskResponse(**result)


def _preprocess_for_summary(document):
    data = {
        "source": document.metadata.get("content_type", "upload"),
        "paper_id": None,
        "title": document.title,
        "abstract": None,
        "text": document.text,
        "meta": document.metadata,
    }
    if preprocess:
        return preprocess(data)
    return {"clean_text": document.text, "focus_text": document.text, "chunks": [document.text], "stats": {}}


def _summarize(prepared: dict) -> str:
    if summarize_hybrid:
        try:
            return summarize_hybrid(prepared["focus_text"], reduction_ratio=0.42)
        except Exception:
            pass
    return " ".join(prepared.get("chunks", []))[:1800]


def _extract_insights(text: str) -> dict:
    if extract_insights:
        try:
            return extract_insights(text, top_n=12)
        except Exception:
            pass
    words = [word.strip(".,;:()[]").lower() for word in text.split()]
    candidates = [word for word in words if len(word) > 5 and word.isalpha()]
    ranked = sorted(set(candidates), key=candidates.count, reverse=True)
    return {"keywords": ranked[:12]}


def _bullets(summary: str) -> list[str]:
    sentences = [sentence.strip() for sentence in summary.split(".") if sentence.strip()]
    return sentences[:6]
