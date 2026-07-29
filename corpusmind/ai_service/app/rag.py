from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Chunk:
    id: str
    text: str
    source_type: str
    page: int | None
    metadata: dict[str, Any]


class AdvancedRAGPipeline:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def build(self, document: Any) -> dict[str, Any]:
        self.chunks = _make_chunks(document)
        texts = [chunk.text for chunk in self.chunks]
        self._load_embedding_model()
        self.embeddings = self._encode(texts) if texts else None
        return {
            "chunk_count": len(self.chunks),
            "retrieval": "bert-dense-plus-bm25",
            "embedding_model": self.embedding_model_name if self.embedding_model else "deterministic-embedding-fallback",
            "modalities": {
                "text_pages": len(document.pages),
                "tables": len(document.tables),
                "image_ocr_blocks": len(document.images),
            },
        }

    def retrieve(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        if not self.chunks or self.embeddings is None:
            return []
        query_vector = self._encode([query])[0]
        bert_scores = self.embeddings @ query_vector
        bm25_scores = _bm25(query, self.chunks)
        modality_scores = _modality_boost(query, self.chunks)
        fused = _normalize(bert_scores) * 0.72 + _normalize(bm25_scores) * 0.20 + modality_scores * 0.08
        candidates = sorted(range(len(self.chunks)), key=lambda idx: fused[idx], reverse=True)[: max(top_k * 4, top_k)]
        selected = _mmr(candidates, fused, self.embeddings, top_k=top_k)
        return [
            {
                "id": self.chunks[idx].id,
                "text": self.chunks[idx].text,
                "score": float(fused[idx]),
                "source_type": self.chunks[idx].source_type,
                "page": self.chunks[idx].page,
                "metadata": self.chunks[idx].metadata,
            }
            for idx in selected
        ]

    def answer(self, query: str, summary: str | None = None, top_k: int = 8) -> dict[str, Any]:
        contexts = self.retrieve(query, top_k=top_k)
        if not contexts:
            return {"answer": "", "citations": [], "contexts": []}
        evidence = "\n\n".join(f"[{item['id']}] {item['text']}" for item in contexts)
        answer = _synthesize_answer(query, evidence, summary)
        return {
            "answer": answer,
            "citations": [
                {"id": item["id"], "page": item["page"], "source_type": item["source_type"], "score": item["score"]}
                for item in contexts
            ],
            "contexts": contexts,
        }

    def _load_embedding_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        except Exception:
            self.embedding_model = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=float)
        if self.embedding_model:
            try:
                return np.asarray(self.embedding_model.encode(texts, normalize_embeddings=True), dtype=float)
            except Exception:
                pass
        vectors = np.asarray([_deterministic_embedding(text) for text in texts], dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-9)


def _deterministic_embedding(text: str, dimensions: int = 384) -> np.ndarray:
    vector = np.zeros(dimensions, dtype=float)
    terms = _terms(text)
    if not terms:
        return vector
    for term in terms:
        digest = hashlib.sha256(term.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1 if digest[4] % 2 == 0 else -1
        vector[index] += sign * (1 + math.log1p(len(term)))
    return vector


def _modality_boost(query: str, chunks: list[Chunk]) -> np.ndarray:
    query_terms = set(_terms(query))
    boosts = []
    for chunk in chunks:
        boost = 0.0
        if {"table", "row", "column", "value", "metric", "result"} & query_terms and chunk.source_type == "table":
            boost += 1.0
        if {"image", "figure", "diagram", "chart", "scan", "visual"} & query_terms and chunk.source_type == "image_ocr":
            boost += 1.0
        if chunk.source_type == "page":
            boost += 0.25
        boosts.append(boost)
    return _normalize(np.asarray(boosts, dtype=float))


def _make_chunks(document: Any, chunk_words: int = 220, overlap: int = 45) -> list[Chunk]:
    assets = [*document.pages, *document.tables, *document.images]
    chunks: list[Chunk] = []
    for asset in assets:
        words = asset.content.split()
        if not words:
            continue
        step = max(1, chunk_words - overlap)
        for start in range(0, len(words), step):
            window = " ".join(words[start : start + chunk_words])
            if not window.strip():
                continue
            digest = hashlib.sha1(f"{asset.asset_type}:{asset.page}:{start}:{window[:80]}".encode()).hexdigest()[:10]
            chunks.append(
                Chunk(
                    id=f"{asset.asset_type}-{asset.page or 0}-{digest}",
                    text=window,
                    source_type=asset.asset_type,
                    page=asset.page,
                    metadata={**asset.metadata, "word_start": start},
                )
            )
    return chunks


def _bm25(query: str, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    query_terms = _terms(query)
    docs = [_terms(chunk.text) for chunk in chunks]
    avgdl = sum(len(doc) for doc in docs) / max(len(docs), 1)
    scores = []
    for doc in docs:
        score = 0.0
        frequencies = {term: doc.count(term) for term in set(doc)}
        for term in query_terms:
            n_q = sum(1 for candidate in docs if term in candidate)
            idf = math.log(1 + (len(docs) - n_q + 0.5) / (n_q + 0.5))
            tf = frequencies.get(term, 0)
            denom = tf + k1 * (1 - b + b * len(doc) / max(avgdl, 1))
            score += idf * ((tf * (k1 + 1)) / denom) if denom else 0
        scores.append(score)
    return np.asarray(scores)


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{1,}", text.lower())


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum < 1e-9:
        return np.zeros_like(values, dtype=float)
    return (values - minimum) / (maximum - minimum)


def _mmr(candidates: list[int], scores: np.ndarray, embeddings: np.ndarray, top_k: int, lambda_mult: float = 0.72) -> list[int]:
    selected: list[int] = []
    remaining = candidates[:]
    similarity = embeddings @ embeddings.T
    while remaining and len(selected) < top_k:
        if not selected:
            choice = remaining.pop(0)
            selected.append(choice)
            continue
        choice = max(
            remaining,
            key=lambda idx: lambda_mult * scores[idx] - (1 - lambda_mult) * max(similarity[idx][sel] for sel in selected),
        )
        remaining.remove(choice)
        selected.append(choice)
    return selected


def _synthesize_answer(query: str, evidence: str, summary: str | None) -> str:
    try:
        from transformers import pipeline

        generator = pipeline("text2text-generation", model="google/flan-t5-base")
        prompt = (
            "Answer the research question using only the cited evidence. "
            "Mention uncertainty when evidence is incomplete.\n"
            f"Question: {query}\nEvidence:\n{evidence[:6000]}\nSummary:\n{summary or ''}"
        )
        return generator(prompt, max_new_tokens=260, do_sample=False)[0]["generated_text"]
    except Exception:
        intro = f"Evidence-grounded response for: {query}"
        lines = [intro]
        if summary:
            lines.append(f"Document summary: {summary[:900]}")
        lines.append("Relevant evidence:")
        for block in evidence.split("\n\n")[:4]:
            lines.append(f"- {block[:500]}")
        return "\n".join(lines)
