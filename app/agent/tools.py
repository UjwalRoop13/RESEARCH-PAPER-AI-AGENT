"""
Tool implementations. Every function here has the signature
(input_dict) -> dict, is pure w.r.t. its inputs (no hidden global state
beyond the DB/vector store), and is registered in TOOL_REGISTRY under the
same name used in tool_schemas.py. The orchestrator looks tools up by name
from this registry - adding a new tool never requires touching the
orchestrator (FR-17).
"""
from __future__ import annotations

import json
from typing import Any, Callable

from app.config import settings
from app.db import get_conn, new_id, now
from app.embeddings import get_embedder
from app.vectorstore import get_vector_store

DEFAULT_COMPARISON_DIMENSIONS = ["objective", "method", "dataset", "key results", "limitations"]


class ToolError(Exception):
    """Raised by a tool for an expected, user-facing failure (bad paper_id, etc.)."""


def _get_paper(paper_id: str) -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    if row is None:
        raise ToolError(f"No paper found with paper_id={paper_id!r}.")
    return dict(row)


def read_pdf(input: dict[str, Any]) -> dict[str, Any]:
    paper_id = input["paper_id"]
    max_pages = input.get("max_pages")
    paper = _get_paper(paper_id)
    if paper["status"] != "ready":
        raise ToolError(f"Paper {paper_id} is not ready yet (status={paper['status']}).")

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT chunk_index, text, page_number FROM chunks WHERE paper_id = ? ORDER BY chunk_index",
            (paper_id,),
        ).fetchall()

    pages: dict[int, list[str]] = {}
    for r in rows:
        pages.setdefault(r["page_number"], []).append(r["text"])

    page_numbers = sorted(pages.keys())
    if max_pages:
        page_numbers = page_numbers[: int(max_pages)]

    full_text = "\n\n".join(f"[Page {pn}]\n" + " ".join(pages[pn]) for pn in page_numbers)
    return {
        "paper_id": paper_id,
        "title": paper["title"],
        "page_count": paper["page_count"],
        "pages_returned": page_numbers,
        "text": full_text,
    }


def retrieve_context(input: dict[str, Any]) -> dict[str, Any]:
    query = input["query"]
    paper_ids = input.get("paper_ids") or None
    top_k = int(input.get("top_k") or settings.retrieval_top_k)

    embedder = get_embedder()
    store = get_vector_store()
    query_vec = embedder.embed_texts([query])[0]
    results = store.search(query_vec, top_k=top_k, paper_ids=paper_ids)

    # attach paper titles for readability/citation
    with get_conn() as conn:
        titles = {
            r["paper_id"]: r["title"]
            for r in conn.execute("SELECT paper_id, title FROM papers").fetchall()
        }

    return {
        "query": query,
        "results": [
            {
                "paper_id": r.paper_id,
                "paper_title": titles.get(r.paper_id, "Unknown"),
                "chunk_id": r.chunk_id,
                "page_number": r.page_number,
                "score": round(r.score, 4),
                "text": r.text,
            }
            for r in results
        ],
    }


def compare_papers(input: dict[str, Any]) -> dict[str, Any]:
    paper_ids = input["paper_ids"]
    if len(paper_ids) < 2:
        raise ToolError("compare_papers requires at least 2 paper_ids.")
    dimensions = input.get("dimensions") or DEFAULT_COMPARISON_DIMENSIONS

    embedder = get_embedder()
    store = get_vector_store()

    with get_conn() as conn:
        titles = {
            r["paper_id"]: r["title"]
            for r in conn.execute("SELECT paper_id, title FROM papers WHERE paper_id IN ({})".format(
                ",".join("?" * len(paper_ids))
            ), paper_ids).fetchall()
        }
    missing = [pid for pid in paper_ids if pid not in titles]
    if missing:
        raise ToolError(f"Unknown paper_id(s): {missing}")

    comparison: dict[str, dict[str, list[dict]]] = {}
    for dim in dimensions:
        comparison[dim] = {}
        for pid in paper_ids:
            query_vec = embedder.embed_texts([f"{dim} of the paper"])[0]
            hits = store.search(query_vec, top_k=2, paper_ids=[pid])
            comparison[dim][pid] = [
                {"page_number": h.page_number, "text": h.text, "score": round(h.score, 4)} for h in hits
            ]

    return {
        "papers": {pid: titles[pid] for pid in paper_ids},
        "dimensions": dimensions,
        "comparison": comparison,
    }


def save_notes(input: dict[str, Any]) -> dict[str, Any]:
    content = input["content"]
    paper_id = input.get("paper_id")
    chunk_id = input.get("chunk_id")
    tags = input.get("tags") or []

    if paper_id:
        _get_paper(paper_id)  # raises ToolError if invalid

    note_id = new_id()
    ts = now()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO notes (note_id, paper_id, chunk_id, content, tags, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (note_id, paper_id, chunk_id, content, json.dumps(tags), ts, ts),
        )
    return {"note_id": note_id, "paper_id": paper_id, "chunk_id": chunk_id, "content": content, "tags": tags}


def generate_report(input: dict[str, Any]) -> dict[str, Any]:
    title = input["title"]
    report_type = input["report_type"]
    paper_ids = input.get("paper_ids") or []
    content_md = input["content_md"]

    report_id = new_id()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO reports (report_id, session_id, title, report_type, paper_ids, content_md, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (report_id, input.get("session_id"), title, report_type, json.dumps(paper_ids), content_md, now()),
        )
    return {"report_id": report_id, "title": title, "report_type": report_type, "paper_ids": paper_ids}


TOOL_REGISTRY: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "read_pdf": read_pdf,
    "retrieve_context": retrieve_context,
    "compare_papers": compare_papers,
    "save_notes": save_notes,
    "generate_report": generate_report,
}
