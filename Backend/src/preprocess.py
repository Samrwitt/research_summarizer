# src/preprocess.py
"""
preprocess.py
-------------
Takes the output dict from ingest.py and produces:
- clean_text: normalized text with boilerplate removed
- sections: best-effort section extraction (abstract/introduction/method/results/conclusion)
- chunks: model-friendly chunks for summarization (token-count based)
- stats: useful debugging counters

Design goals:
- Works on messy PDF-extracted text and cleaner arXiv HTML text
- No hard dependency on spaCy/NLTK (optional sentence splitting)
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Optional, Tuple


# ---------------------------
# Cleaning helpers
# ---------------------------

_RE_MULTISPACE = re.compile(r"[ \t]+")
_RE_MULTI_NEWLINES = re.compile(r"\n{3,}")

# Common arXiv/HTML boilerplate seen in some papers
_BOILERPLATE_PATTERNS = [
    r"Provided proper attribution is provided.*?scholarly works\.",
    r"arXiv:\s*\d{4}\.\d{4,5}v?\d*",
    r"https?://arxiv\.org/\S+",
]

# A “good enough” section heading detector for common research papers
# We’ll use this to split sections in plain text.
_SECTION_HEADING_RE = re.compile(
    r"^\s*(?:\d+\s*[\.\)]\s*)?"
    r"(abstract|introduction|background|related work|preliminaries|methods?|methodology|approach|model|"
    r"experiments?|experimental setup|results?|discussion|analysis|conclusion|conclusions|limitations|"
    r"future work|acknowledg(e)?ments?|references|bibliography|appendix)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# If the doc includes "References" we usually want to cut everything after it
_CUT_AFTER_RE = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE)

# Footnote markers etc. (light touch)
_FOOTNOTE_MARKERS = [
    r"\bfootnote\b",
    r"\bfootnotemark\b",
]

# Inline citations like [12], [3, 5], (Smith et al., 2020)
_BRACKET_CITATION_RE = re.compile(r"\[[0-9,\s\-]{1,20}\]")
_PAREN_CITATION_RE = re.compile(r"\(([A-Z][A-Za-z]+ et al\.,?\s*\d{4}|[A-Z][A-Za-z]+,\s*\d{4})\)")

# Sometimes PDF extraction creates hyphen + newline breaks: "trans-\nformer"
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")

# Remove page numbers alone on a line (often from PDFs)
_PAGE_NUM_RE = re.compile(r"^\s*\d+\s*$", re.MULTILINE)


def _strip_boilerplate(text: str) -> str:
    out = text
    for pat in _BOILERPLATE_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE | re.DOTALL)
    for pat in _FOOTNOTE_MARKERS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    return out


def _normalize_text(text: str) -> str:
    # Fix hyphen line breaks first
    text = re.sub(_HYPHEN_LINEBREAK_RE, r"\1\2", text)

    # Normalize CRLF
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove page numbers that are alone on a line
    text = re.sub(_PAGE_NUM_RE, "", text)

    # Normalize spaces
    text = re.sub(_RE_MULTISPACE, " ", text)

    # Normalize newlines
    text = re.sub(_RE_MULTI_NEWLINES, "\n\n", text)

    return text.strip()


def _remove_references_tail(text: str) -> Tuple[str, bool]:
    """
    Cut everything after the first "References"/"Bibliography" heading, if found.
    Returns (trimmed_text, did_cut).
    """
    m = _CUT_AFTER_RE.search(text)
    if not m:
        return text, False
    return text[: m.start()].strip(), True


def _light_denoise(text: str, remove_citations: bool = False) -> str:
    """
    Optional: remove citation markers to reduce noise.
    For summarization, leaving citations is usually fine; removing them can help chunking.
    """
    out = text
    if remove_citations:
        out = re.sub(_BRACKET_CITATION_RE, "", out)
        out = re.sub(_PAREN_CITATION_RE, "", out)

    # Clean leftover double spaces/newlines
    out = re.sub(_RE_MULTISPACE, " ", out)
    out = re.sub(_RE_MULTI_NEWLINES, "\n\n", out)
    return out.strip()


# ---------------------------
# Section parsing
# ---------------------------

def _find_sections(text: str) -> Dict[str, str]:
    """
    Best-effort section extraction from plain text. Works reasonably for arXiv HTML text.
    Returns a dict: {section_name_lower: content}
    """
    # Find headings with their positions
    hits = []
    for m in _SECTION_HEADING_RE.finditer(text):
        name = m.group(1).lower()
        # Normalize a few variants
        name = {
            "methods": "method",
            "methodology": "method",
            "experiments": "experiments",
            "results": "results",
            "conclusions": "conclusion",
            "acknowledgements": "acknowledgments",
            "bibliography": "references",
        }.get(name, name)
        hits.append((m.start(), m.end(), name))

    if not hits:
        return {}

    # Sort and slice
    hits.sort(key=lambda x: x[0])
    sections: Dict[str, str] = {}

    for i, (start, end, name) in enumerate(hits):
        next_start = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        content = text[end:next_start].strip()

        # Ignore tiny/empty sections
        if len(content) < 200:
            continue

        # Keep first occurrence of a section name (usually best)
        if name not in sections:
            sections[name] = content

    return sections


# ---------------------------
# Chunking
# ---------------------------

def _approx_token_count(s: str) -> int:
    """
    Approximate token count. This is not exact BPE tokenization, but works for chunk sizing.
    """
    # Splitting on whitespace tends to slightly undercount compared to BPE tokens.
    # Multiply by ~1.2 to be safer for transformer limits.
    words = len(s.split())
    return int(words * 1.2)


def chunk_text(
    text: str,
    *,
    max_tokens: int = 900,
    overlap_tokens: int = 120,
) -> List[str]:
    """
    Chunk long text into overlapping chunks using paragraph boundaries.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []

    current: List[str] = []
    current_tokens = 0

    def flush():
        nonlocal current, current_tokens
        if current:
            chunks.append("\n\n".join(current).strip())
        current = []
        current_tokens = 0

    for p in paragraphs:
        p_tokens = _approx_token_count(p)

        # If a single paragraph is huge, split it by sentences-ish
        if p_tokens > max_tokens:
            # naive sentence splitting on period/newline boundaries
            parts = re.split(r"(?<=[\.\!\?])\s+|\n+", p)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                part_tokens = _approx_token_count(part)
                if current_tokens + part_tokens > max_tokens:
                    flush()
                current.append(part)
                current_tokens += part_tokens
            continue

        if current_tokens + p_tokens > max_tokens:
            flush()

            # Add overlap from previous chunk end
            if chunks and overlap_tokens > 0:
                prev = chunks[-1]
                # Take last N words as overlap proxy
                words = prev.split()
                overlap_words = words[-max(1, int(overlap_tokens / 1.2)) :]
                overlap = " ".join(overlap_words).strip()
                if overlap:
                    current = [overlap]
                    current_tokens = _approx_token_count(overlap)

        current.append(p)
        current_tokens += p_tokens

    flush()
    return chunks


def build_focus_text(
    *,
    abstract: Optional[str],
    sections: Dict[str, str],
    clean_text: str,
    max_chars: int = 120_000,
) -> str:
    """
    Create a "focus text" that prioritizes sections useful for summarizing research papers.
    Order: Abstract -> Introduction -> Method -> Results -> Conclusion.
    If missing, fallback to clean_text.
    """
    parts: List[str] = []

    if abstract:
        parts.append("ABSTRACT\n" + abstract.strip())

    def add_if(name: str, label: str):
        if name in sections:
            parts.append(f"{label}\n{sections[name].strip()}")

    add_if("introduction", "INTRODUCTION")
    # Some papers use "approach" / "model" instead of "method"
    if "method" in sections:
        add_if("method", "METHOD")
    else:
        add_if("approach", "APPROACH")
        add_if("model", "MODEL")

    add_if("experiments", "EXPERIMENTS")
    add_if("results", "RESULTS")
    add_if("discussion", "DISCUSSION")
    add_if("analysis", "ANALYSIS")
    add_if("conclusion", "CONCLUSION")

    if not parts:
        focus = clean_text
    else:
        focus = "\n\n".join(parts).strip()

    # Hard cap to avoid accidental huge memory/compute spikes
    if len(focus) > max_chars:
        focus = focus[:max_chars]

    return focus


# ---------------------------
# Public API
# ---------------------------

def preprocess_ingest(
    ingest_dict: Dict[str, Any],
    *,
    remove_references: bool = True,
    remove_citations: bool = False,
    max_tokens_per_chunk: int = 900,
    overlap_tokens: int = 120,
) -> Dict[str, Any]:
    """
    Main entrypoint: pass the dict returned by ingest.ingest(...)

    Returns:
    {
      "title": ...,
      "abstract": ...,
      "clean_text": ...,
      "sections": {...},
      "focus_text": ...,
      "chunks": [...],
      "stats": {...},
      "meta": {...}  # carry-through
    }
    """
    title = ingest_dict.get("title")
    abstract = ingest_dict.get("abstract")
    raw_text = ingest_dict.get("text") or ""

    # Start with raw text
    text = raw_text

    # 1) Strip boilerplate + normalize
    text = _strip_boilerplate(text)
    text = _normalize_text(text)

    # 2) Optionally cut references tail
    did_cut = False
    if remove_references:
        text, did_cut = _remove_references_tail(text)

    # 3) Optional denoise (citations)
    text = _light_denoise(text, remove_citations=remove_citations)

    # 4) Extract sections (best effort)
    sections = _find_sections(text)

    # 5) Build a focused text for summarization (better than dumping entire paper)
    focus_text = build_focus_text(
        abstract=abstract,
        sections=sections,
        clean_text=text,
    )

    # 6) Chunking (use focus_text as default for chunking)
    chunks = chunk_text(
        focus_text,
        max_tokens=max_tokens_per_chunk,
        overlap_tokens=overlap_tokens,
    )

    stats = {
        "raw_chars": len(raw_text),
        "clean_chars": len(text),
        "focus_chars": len(focus_text),
        "num_sections": len(sections),
        "num_chunks": len(chunks),
        "cut_references": did_cut,
        "remove_citations": remove_citations,
        "pdf_or_html_source": (ingest_dict.get("meta") or {}).get("fulltext_source"),
    }

    return {
        "title": title,
        "abstract": abstract,
        "clean_text": text,
        "sections": sections,
        "focus_text": focus_text,
        "chunks": chunks,
        "stats": stats,
        "meta": ingest_dict.get("meta") or {},
        "source": ingest_dict.get("source"),
        "paper_id": ingest_dict.get("paper_id"),
    }


if __name__ == "__main__":
    # Quick local test (works if ingest.py works)
    from src.ingest import ingest
    from src.export import save_preprocess_docx

    d = ingest(arxiv="1706.03762")
    out = preprocess_ingest(d, remove_references=True, remove_citations=False)
    print("Stats:", out["stats"])
    path = save_preprocess_docx(out, "preprocess.docx")
    print("saved:", path)
    print("\nSections found:", list(out["sections"].keys())[:10])
    print("\nFirst chunk preview:\n", out["chunks"][0][:800], "...\n")
