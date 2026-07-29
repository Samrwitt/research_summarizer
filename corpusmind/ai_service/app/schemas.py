from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=8, ge=1, le=20)


class AnalyzeResponse(BaseModel):
    document_id: str
    title: str
    summary: str
    bullets: list[str]
    insights: dict
    rag: dict
    metadata: dict


class AskResponse(BaseModel):
    answer: str
    citations: list[dict]
    contexts: list[dict]
