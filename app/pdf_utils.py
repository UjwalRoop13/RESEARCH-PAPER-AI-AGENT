"""
PDF ingestion: text extraction per page, plus chunking into overlapping
token-budgeted chunks that retain page numbers for citation (FR-05, FR-06,
RAG Design > Document ingestion / Chunking strategy in the SRS).

Uses pypdf as the primary extractor and falls back to pdfplumber for pages
that come back empty (pypdf sometimes struggles with certain encodings).
Scanned/OCR-only PDFs are explicitly out of scope for this MVP - such pages
are flagged in the returned metadata rather than silently dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings


@dataclass
class PageText:
    page_number: int  # 1-indexed
    text: str


@dataclass
class ExtractionResult:
    pages: list[PageText]
    page_count: int
    empty_pages: list[int] = field(default_factory=list)  # likely scanned/OCR-needed


@dataclass
class Chunk:
    chunk_index: int
    text: str
    page_number: int
    token_count: int


def _approx_token_count(text: str) -> int:
    # Cheap, dependency-free approximation: ~4 characters per token for
    # English academic text. Good enough for chunk-sizing decisions; not
    # used for LLM billing accuracy.
    return max(1, len(text) // 4)


def extract_text(pdf_path: str | Path) -> ExtractionResult:
    pdf_path = Path(pdf_path)
    pages: list[PageText] = []
    empty_pages: list[int] = []

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            pages.append(PageText(page_number=i, text=text))
    except Exception:
        pages = []

    # Fallback / patch empty pages with pdfplumber, which handles some
    # layouts pypdf misses.
    needs_fallback = not pages or any(not p.text for p in pages)
    if needs_fallback:
        try:
            import pdfplumber

            with pdfplumber.open(str(pdf_path)) as pdf:
                if not pages:
                    pages = [PageText(page_number=i, text="") for i in range(1, len(pdf.pages) + 1)]
                for i, plumber_page in enumerate(pdf.pages, start=1):
                    if i - 1 < len(pages) and not pages[i - 1].text:
                        extracted = (plumber_page.extract_text() or "").strip()
                        pages[i - 1] = PageText(page_number=i, text=extracted)
        except Exception:
            pass

    for p in pages:
        if not p.text:
            empty_pages.append(p.page_number)

    return ExtractionResult(pages=pages, page_count=len(pages), empty_pages=empty_pages)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_into_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return _SENTENCE_SPLIT_RE.split(text)


def chunk_pages(
    pages: list[PageText],
    target_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[Chunk]:
    """
    Sentence-aware sliding-window chunking. Chunks never cross a page
    boundary (so every chunk has exactly one page_number for citation),
    and consecutive chunks on the same page overlap by ~overlap_tokens
    to avoid losing context at chunk edges (SRS RAG Design > Chunking
    strategy).
    """
    target_tokens = target_tokens or settings.chunk_target_tokens
    overlap_tokens = overlap_tokens or settings.chunk_overlap_tokens

    chunks: list[Chunk] = []
    idx = 0
    for page in pages:
        sentences = _split_into_sentences(page.text)
        if not sentences:
            continue

        current: list[str] = []
        current_tokens = 0
        for sentence in sentences:
            sent_tokens = _approx_token_count(sentence)
            if current and current_tokens + sent_tokens > target_tokens:
                chunk_text = " ".join(current).strip()
                chunks.append(Chunk(idx, chunk_text, page.page_number, current_tokens))
                idx += 1
                # carry the tail of the previous chunk forward for overlap
                overlap: list[str] = []
                overlap_tok = 0
                for s in reversed(current):
                    t = _approx_token_count(s)
                    if overlap_tok + t > overlap_tokens:
                        break
                    overlap.insert(0, s)
                    overlap_tok += t
                current = overlap
                current_tokens = overlap_tok

            current.append(sentence)
            current_tokens += sent_tokens

        if current:
            chunk_text = " ".join(current).strip()
            chunks.append(Chunk(idx, chunk_text, page.page_number, current_tokens))
            idx += 1

    return chunks


def guess_metadata_from_first_page(pages: list[PageText]) -> dict:
    """
    Very lightweight heuristic metadata extraction (FR-03). A real system
    would use a dedicated metadata-extraction model/service; this MVP
    grabs the first non-empty line as a title guess so uploads aren't left
    completely untitled, and leaves everything else for manual entry.
    """
    if not pages or not pages[0].text:
        return {"title_guess": None}
    first_lines = [ln.strip() for ln in pages[0].text.splitlines() if ln.strip()]
    title_guess = first_lines[0][:300] if first_lines else None
    return {"title_guess": title_guess}
