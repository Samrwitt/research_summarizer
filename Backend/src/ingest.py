"""
ingest.py
---------
Input sources:
- arXiv link or id: fetch title/abstract; try HTML full text; fallback to PDF.
- PDF file: extract text with PyMuPDF.
- .txt file: read raw text.

Returns a normalized dict with:
{
  "source": "arxiv" | "pdf" | "text",
  "paper_id": str | None,
  "title": str | None,
  "abstract": str | None,
  "text": str,           # best-effort full text (may be partial)
  "meta": dict
}
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import requests
from bs4 import BeautifulSoup

import fitz  # pip install pymupdf failing because of internet


ARXIV_ABS_RE = re.compile(r"(?:arxiv\.org\/abs\/)?(?P<id>\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
ARXIV_NEWSTYLE_RE = ARXIV_ABS_RE
ARXIV_OLDSTYLE_RE = re.compile(r"(?:arxiv\.org\/abs\/)?(?P<id>[a-z\-]+\/\d{7})(?:v\d+)?", re.IGNORECASE)


@dataclass
class IngestResult:
    source: str
    paper_id: Optional[str]
    title: Optional[str]
    abstract: Optional[str]
    text: str
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "text": self.text,
            "meta": self.meta,
        }


def _requests_get(url: str, timeout: int = 20) -> requests.Response:
    headers = {
        "User-Agent": "paper-summarizer/1.0 (+https://example.com; educational project)"
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r


def parse_arxiv_id(arxiv_input: str) -> str:
    """
    Accepts:
    - full URL: https://arxiv.org/abs/1706.03762
    - full URL: https://arxiv.org/pdf/1706.03762.pdf
    - bare id: 1706.03762
    - old style: cs/0112017
    Returns: normalized arXiv id string (without version suffix).
    """
    s = arxiv_input.strip()

    # Normalize common PDF URLs to abs-style patterns
    s = s.replace("arxiv.org/pdf/", "arxiv.org/abs/").replace(".pdf", "")

    m = ARXIV_NEWSTYLE_RE.search(s)
    if m:
        return m.group("id")

    m2 = ARXIV_OLDSTYLE_RE.search(s)
    if m2:
        return m2.group("id")

    raise ValueError(f"Could not parse arXiv id from: {arxiv_input}")


def fetch_arxiv_abs(arxiv_id: str) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """
    Fetch title + abstract from arXiv abs page.
    """
    url = f"https://arxiv.org/abs/{arxiv_id}"
    r = _requests_get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    # Title
    title = None
    title_tag = soup.find("h1", class_="title")
    if title_tag:
        title = title_tag.get_text(" ", strip=True)
        title = re.sub(r"^Title:\s*", "", title).strip()

    # Abstract
    abstract = None
    abs_tag = soup.find("blockquote", class_="abstract")
    if abs_tag:
        abstract = abs_tag.get_text(" ", strip=True)
        abstract = re.sub(r"^Abstract:\s*", "", abstract).strip()

    # Basic meta
    meta: Dict[str, Any] = {"abs_url": url}

    # Authors (optional)
    authors_div = soup.find("div", class_="authors")
    if authors_div:
        authors = authors_div.get_text(" ", strip=True)
        authors = re.sub(r"^Authors?:\s*", "", authors).strip()
        meta["authors"] = authors

    # Categories (optional)
    subj_span = soup.find("span", class_="primary-subject")
    if subj_span:
        meta["primary_subject"] = subj_span.get_text(" ", strip=True)

    return title, abstract, meta


def fetch_arxiv_html_fulltext(arxiv_id: str) -> Optional[str]:
    """
    Try to fetch HTML version (not available for all papers).
    We heuristically extract main text and drop nav/sidebars.
    """
    url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        r = _requests_get(url, timeout=25)
    except Exception:
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    # arXiv HTML pages vary; often main content is inside <article> or <main>
    main = soup.find("article") or soup.find("main") or soup.body
    if not main:
        return None

    # Remove script/style
    for tag in main.find_all(["script", "style", "noscript"]):
        tag.decompose()

    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    
    if len(text) < 2000:
        return None

    return text


def download_arxiv_pdf(arxiv_id: str, dest_path: str) -> str:
    """
    Download arXiv PDF to dest_path.
    """
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    r = _requests_get(pdf_url, timeout=30)
    with open(dest_path, "wb") as f:
        f.write(r.content)
    return dest_path


def extract_text_from_pdf(pdf_path: str, max_pages: Optional[int] = None) -> str:
    """
    Extract text from a PDF using PyMuPDF.
    max_pages can be set to speed up (e.g., 12 pages) for quick demos.
    """
    doc = fitz.open(pdf_path)
    pages = range(len(doc)) if max_pages is None else range(min(len(doc), max_pages))
    chunks = []
    for i in pages:
        page = doc.load_page(i)
        chunks.append(page.get_text("text"))
    doc.close()

    text = "\n".join(chunks)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def ingest_arxiv(arxiv_input: str, cache_dir: str = ".cache", prefer_html: bool = True) -> IngestResult:
    arxiv_id = parse_arxiv_id(arxiv_input)

    title, abstract, meta = fetch_arxiv_abs(arxiv_id)

    full_text = None
    if prefer_html:
        full_text = fetch_arxiv_html_fulltext(arxiv_id)
        if full_text:
            meta["fulltext_source"] = "html"
            meta["html_url"] = f"https://arxiv.org/html/{arxiv_id}"

    # Fallback to PDF extraction
    if not full_text:
        import os
        os.makedirs(cache_dir, exist_ok=True)
        pdf_path = os.path.join(cache_dir, f"{arxiv_id}.pdf")
        try:
            download_arxiv_pdf(arxiv_id, pdf_path)
            full_text = extract_text_from_pdf(pdf_path)
            meta["fulltext_source"] = "pdf"
            meta["pdf_url"] = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            meta["cached_pdf"] = pdf_path
        except Exception as e:
            # at least return abstract as text
            meta["fulltext_source"] = "abstract_only"
            meta["ingest_warning"] = f"Could not fetch full text; using abstract only. Error: {e}"
            full_text = abstract or ""


    text = full_text or ""
    return IngestResult(
        source="arxiv",
        paper_id=arxiv_id,
        title=title,
        abstract=abstract,
        text=text,
        meta=meta,
    )


def ingest_pdf(pdf_path: str, max_pages: Optional[int] = None) -> IngestResult:
    text = extract_text_from_pdf(pdf_path, max_pages=max_pages)
    return IngestResult(
        source="pdf",
        paper_id=None,
        title=None,
        abstract=None,
        text=text,
        meta={"pdf_path": pdf_path, "max_pages": max_pages},
    )


def ingest_text(text_path: str) -> IngestResult:
    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    text = text.strip()
    return IngestResult(
        source="text",
        paper_id=None,
        title=None,
        abstract=None,
        text=text,
        meta={"text_path": text_path},
    )


def ingest(
    *,
    arxiv: Optional[str] = None,
    pdf: Optional[str] = None,
    text: Optional[str] = None,
    cache_dir: str = ".cache",
    prefer_html: bool = True,
    max_pdf_pages: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Single entrypoint for the rest of the pipeline.
    Exactly one of arxiv/pdf/text should be provided.
    """
    provided = [x is not None for x in [arxiv, pdf, text]]
    if sum(provided) != 1:
        raise ValueError("Provide exactly one input: arxiv=..., pdf=..., or text=...")

    if arxiv:
        return ingest_arxiv(arxiv, cache_dir=cache_dir, prefer_html=prefer_html).to_dict()
    if pdf:
        return ingest_pdf(pdf, max_pages=max_pdf_pages).to_dict()
    # text
    return ingest_text(text).to_dict()


if __name__ == "__main__":
    # Quick manual test:
    # python -m src.ingest  (not used by your CLI, just sanity checks)
    example = ingest(arxiv="1706.03762")
    print(json.dumps({k: (v[:300] + "..." if isinstance(v, str) and len(v) > 300 else v) for k, v in example.items()}, indent=2))
