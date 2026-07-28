from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from app.agent.tools import generate_report
from app.db import get_conn
from app.exporters import export_docx_bytes, export_markdown_bytes, export_pdf_bytes

router = APIRouter(prefix="/api/reports", tags=["reports"])


class ReportIn(BaseModel):
    title: str
    report_type: str  # summary | comparison | literature_review
    paper_ids: list[str]
    content_md: str
    session_id: str | None = None


@router.post("")
def create_report(report: ReportIn) -> dict:
    return generate_report(report.model_dump())


@router.get("")
def list_reports() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT report_id, session_id, title, report_type, paper_ids, created_at FROM reports "
            "ORDER BY created_at DESC"
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["paper_ids"] = json.loads(d.get("paper_ids") or "[]")
        out.append(d)
    return out


@router.get("/{report_id}")
def get_report(report_id: str) -> dict:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM reports WHERE report_id = ?", (report_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    d = dict(row)
    d["paper_ids"] = json.loads(d.get("paper_ids") or "[]")
    return d


@router.get("/{report_id}/export")
def export_report(report_id: str, format: str = "md") -> Response:
    with get_conn() as conn:
        row = conn.execute("SELECT title, content_md FROM reports WHERE report_id = ?", (report_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    title, content_md = row["title"], row["content_md"]

    if format == "md":
        data, media_type, ext = export_markdown_bytes(content_md), "text/markdown", "md"
    elif format == "docx":
        data = export_docx_bytes(content_md, title)
        media_type, ext = "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"
    elif format == "pdf":
        data, media_type, ext = export_pdf_bytes(content_md, title), "application/pdf", "pdf"
    else:
        raise HTTPException(status_code=400, detail="format must be one of: md, docx, pdf")

    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)[:60].strip() or "report"
    headers = {"Content-Disposition": f'attachment; filename="{safe_title}.{ext}"'}
    return Response(content=data, media_type=media_type, headers=headers)
