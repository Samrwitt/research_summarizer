from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentAsset:
    asset_type: str
    page: int | None
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestedDocument:
    title: str
    text: str
    tables: list[DocumentAsset]
    images: list[DocumentAsset]
    pages: list[DocumentAsset]
    metadata: dict[str, Any]


def ingest_upload(filename: str, payload: bytes, content_type: str | None = None) -> IngestedDocument:
    extension = os.path.splitext(filename.lower())[1]
    if extension == ".pdf" or content_type == "application/pdf":
        return _ingest_pdf(filename, payload)
    if extension in {".txt", ".md", ".csv"} or (content_type or "").startswith("text/"):
        text = payload.decode("utf-8", errors="replace")
        return IngestedDocument(
            title=filename,
            text=text,
            tables=[],
            images=[],
            pages=[DocumentAsset("page", 1, text, {"source": "plain_text"})],
            metadata={"filename": filename, "content_type": content_type, "parser": "text"},
        )
    raise ValueError(f"Unsupported file type for {filename}")


def _ingest_pdf(filename: str, payload: bytes) -> IngestedDocument:
    tables: list[DocumentAsset] = []
    images: list[DocumentAsset] = []
    pages: list[DocumentAsset] = []
    page_texts: list[str] = []
    parser_stack: list[str] = []

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(payload)
        pdf_path = tmp.name

    try:
        try:
            import pdfplumber

            parser_stack.append("pdfplumber")
            with pdfplumber.open(pdf_path) as pdf:
                for page_index, page in enumerate(pdf.pages, start=1):
                    extracted_text = page.extract_text(layout=True) or ""
                    table_blocks = page.extract_tables() or []
                    table_text = "\n".join(_format_table(table) for table in table_blocks if table)
                    if table_text:
                        tables.append(
                            DocumentAsset(
                                "table",
                                page_index,
                                table_text,
                                {"table_count": len(table_blocks), "extraction": "pdfplumber"},
                            )
                        )
                    merged_text = "\n\n".join(part for part in [extracted_text, table_text] if part)
                    if merged_text.strip():
                        page_texts.append(merged_text)
                        pages.append(DocumentAsset("page", page_index, merged_text, {"extraction": "pdfplumber"}))
        except Exception as exc:
            parser_stack.append(f"pdfplumber_failed:{exc.__class__.__name__}")

        ocr_pages = _ocr_pdf_pages(pdf_path)
        if ocr_pages:
            parser_stack.append("ocr")
            for page_index, ocr_text in ocr_pages:
                images.append(DocumentAsset("image_ocr", page_index, ocr_text, {"engine": "pytesseract"}))
            if not page_texts:
                for page_index, ocr_text in ocr_pages:
                    page_texts.append(ocr_text)
                    pages.append(DocumentAsset("page", page_index, ocr_text, {"extraction": "ocr"}))
            else:
                page_texts.append("\n\n".join(text for _, text in ocr_pages))

        if not page_texts:
            page_texts = [_fallback_pdf_text(pdf_path, parser_stack)]
            pages.append(DocumentAsset("page", 1, page_texts[0], {"extraction": "fallback"}))
    finally:
        try:
            os.remove(pdf_path)
        except OSError:
            pass

    return IngestedDocument(
        title=filename,
        text="\n\n".join(page_texts),
        tables=tables,
        images=images,
        pages=pages,
        metadata={
            "filename": filename,
            "content_type": "application/pdf",
            "parser_stack": parser_stack,
            "table_count": len(tables),
            "image_ocr_blocks": len(images),
        },
    )


def _format_table(table: list[list[Any]]) -> str:
    rows = []
    for row in table:
        cells = [str(cell).strip() if cell is not None else "" for cell in row]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def _ocr_pdf_pages(pdf_path: str) -> list[tuple[int, str]]:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception:
        return []

    output: list[tuple[int, str]] = []
    try:
        images = convert_from_path(pdf_path, dpi=220, fmt="png")
        for page_index, image in enumerate(images, start=1):
            text = pytesseract.image_to_string(image)
            if text.strip():
                output.append((page_index, text.strip()))
    except Exception:
        return []
    return output


def _fallback_pdf_text(pdf_path: str, parser_stack: list[str]) -> str:
    try:
        import pypdf

        parser_stack.append("pypdf")
        reader = pypdf.PdfReader(pdf_path)
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        parser_stack.append(f"pypdf_failed:{exc.__class__.__name__}")
        raise RuntimeError("Unable to extract text from the uploaded PDF") from exc
