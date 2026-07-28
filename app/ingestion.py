"""
Ingestion pipeline: upload -> extract -> chunk -> embed -> index.

Runs synchronously in-process for this MVP (see README "Known
simplifications" - the SRS's target architecture puts this behind a task
queue so large PDFs don't block the request; that's a drop-in change
later, not a redesign, since this function is already a single
self-contained unit of work).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.db import get_conn, new_id, now
from app.embeddings import get_embedder
from app.pdf_utils import chunk_pages, extract_text, guess_metadata_from_first_page
from app.vectorstore import get_vector_store


class IngestionError(Exception):
    pass


def create_pending_paper(
    storage_path: str,
    title: str | None = None,
    authors: list[str] | None = None,
    year: str | None = None,
    venue: str | None = None,
    source_url: str | None = None,
) -> str:
    paper_id = new_id()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO papers (paper_id, title, authors, year, venue, source, source_url, storage_path, status, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                paper_id, title or "(untitled upload)", json.dumps(authors or []), year, venue,
                "upload", source_url, storage_path, "queued", now(),
            ),
        )
    return paper_id


def ingest_paper(paper_id: str) -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    if row is None:
        raise IngestionError(f"paper_id {paper_id} not found")
    paper = dict(row)

    with get_conn() as conn:
        conn.execute("UPDATE papers SET status = 'processing' WHERE paper_id = ?", (paper_id,))

    try:
        extraction = extract_text(paper["storage_path"])
        if extraction.page_count == 0:
            raise IngestionError("No pages could be extracted from this PDF.")

        chunks = chunk_pages(extraction.pages)
        if not chunks:
            raise IngestionError(
                "No extractable text found (this looks like a scanned/OCR-only PDF, "
                "which is out of scope for this MVP)."
            )

        # best-effort title backfill if none was supplied at upload time
        title_update = None
        if paper["title"] in (None, "", "(untitled upload)"):
            guess = guess_metadata_from_first_page(extraction.pages)
            if guess.get("title_guess"):
                title_update = guess["title_guess"]

        chunk_ids = []
        with get_conn() as conn:
            for c in chunks:
                cid = new_id()
                chunk_ids.append(cid)
                conn.execute(
                    "INSERT INTO chunks (chunk_id, paper_id, chunk_index, text, page_number, token_count, created_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (cid, paper_id, c.chunk_index, c.text, c.page_number, c.token_count, now()),
                )

        embedder = get_embedder()
        vectors = embedder.embed_texts([c.text for c in chunks])
        store = get_vector_store()
        items = [
            {"chunk_id": cid, "paper_id": paper_id, "page_number": c.page_number, "text": c.text}
            for cid, c in zip(chunk_ids, chunks)
        ]
        store.add_batch(items, vectors)

        with get_conn() as conn:
            if title_update:
                conn.execute(
                    "UPDATE papers SET status='ready', page_count=?, title=? WHERE paper_id=?",
                    (extraction.page_count, title_update, paper_id),
                )
            else:
                conn.execute(
                    "UPDATE papers SET status='ready', page_count=? WHERE paper_id=?",
                    (extraction.page_count, paper_id),
                )

        return {
            "paper_id": paper_id,
            "status": "ready",
            "page_count": extraction.page_count,
            "chunk_count": len(chunks),
            "empty_pages": extraction.empty_pages,
        }

    except Exception as e:
        with get_conn() as conn:
            conn.execute(
                "UPDATE papers SET status='failed', error_message=? WHERE paper_id=?",
                (str(e), paper_id),
            )
        raise IngestionError(str(e)) from e


def ingest_uploaded_pdf(
    storage_path: str,
    title: str | None = None,
    authors: list[str] | None = None,
    year: str | None = None,
    venue: str | None = None,
) -> dict[str, Any]:
    """Convenience wrapper: create the paper row, then ingest it immediately."""
    paper_id = create_pending_paper(storage_path, title=title, authors=authors, year=year, venue=venue)
    return ingest_paper(paper_id)
