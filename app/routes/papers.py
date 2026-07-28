from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app.config import settings
from app.db import get_conn
from app.ingestion import IngestionError, create_pending_paper, ingest_paper
from app.vectorstore import get_vector_store

router = APIRouter(prefix="/api/papers", tags=["papers"])

ALLOWED_SUFFIXES = {".pdf"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB, per FR-02


class PaperOut(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    year: str | None = None
    venue: str | None = None
    source: str
    status: str
    page_count: int | None = None
    error_message: str | None = None


def _row_to_paper_out(row: dict) -> PaperOut:
    return PaperOut(
        paper_id=row["paper_id"],
        title=row["title"],
        authors=json.loads(row["authors"] or "[]"),
        year=row["year"],
        venue=row["venue"],
        source=row["source"],
        status=row["status"],
        page_count=row["page_count"],
        error_message=row["error_message"],
    )


@router.post("/upload", response_model=PaperOut)
async def upload_paper(
    file: UploadFile = File(...),
    title: str | None = None,
    year: str | None = None,
    venue: str | None = None,
) -> PaperOut:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 50MB upload limit.")
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        # paper_id must exist before we can name the file after it, so create
        # the DB row first with an empty path, then patch it in.
        paper_id = create_pending_paper(storage_path="", title=title, year=year, venue=venue)
        dest = settings.uploads_dir / f"{paper_id}.pdf"
        dest.write_bytes(contents)
        with get_conn() as conn:
            conn.execute("UPDATE papers SET storage_path = ? WHERE paper_id = ?", (str(dest), paper_id))

        ingest_paper(paper_id)
    except IngestionError as e:
        raise HTTPException(status_code=422, detail=f"Ingestion failed: {e}")

    with get_conn() as conn:
        row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    return _row_to_paper_out(dict(row))


@router.get("", response_model=list[PaperOut])
def list_papers() -> list[PaperOut]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM papers ORDER BY created_at DESC").fetchall()
    return [_row_to_paper_out(dict(r)) for r in rows]


@router.get("/search")
def search_papers(q: str = Query(..., min_length=1), max_results: int = 5) -> list[dict]:
    from app.agent.search import search_papers_external

    try:
        return search_papers_external(q, max_results=max_results)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Search failed: {e}")


@router.get("/{paper_id}", response_model=PaperOut)
def get_paper(paper_id: str) -> PaperOut:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return _row_to_paper_out(dict(row))


@router.delete("/{paper_id}")
def delete_paper(paper_id: str) -> dict:
    with get_conn() as conn:
        row = conn.execute("SELECT storage_path FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Paper not found.")
        conn.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,))  # cascades to chunks/notes

    get_vector_store().delete_paper(paper_id)

    if row["storage_path"]:
        try:
            Path(row["storage_path"]).unlink(missing_ok=True)
        except OSError:
            pass

    return {"paper_id": paper_id, "deleted": True}
