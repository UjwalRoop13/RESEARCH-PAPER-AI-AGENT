from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.agent.tools import ToolError, save_notes
from app.db import get_conn

router = APIRouter(prefix="/api/notes", tags=["notes"])


class NoteIn(BaseModel):
    content: str
    paper_id: str | None = None
    chunk_id: str | None = None
    tags: list[str] = []


@router.post("")
def create_note(note: NoteIn) -> dict:
    try:
        return save_notes(note.model_dump())
    except ToolError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("")
def list_notes(paper_id: str | None = Query(default=None)) -> list[dict]:
    with get_conn() as conn:
        if paper_id:
            rows = conn.execute(
                "SELECT * FROM notes WHERE paper_id = ? ORDER BY created_at DESC", (paper_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM notes ORDER BY created_at DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["tags"] = json.loads(d.get("tags") or "[]")
        out.append(d)
    return out


@router.delete("/{note_id}")
def delete_note(note_id: str) -> dict:
    with get_conn() as conn:
        row = conn.execute("SELECT note_id FROM notes WHERE note_id = ?", (note_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Note not found.")
        conn.execute("DELETE FROM notes WHERE note_id = ?", (note_id,))
    return {"note_id": note_id, "deleted": True}
