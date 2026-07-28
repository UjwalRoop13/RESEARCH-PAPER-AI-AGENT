from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agent.orchestrator import run_turn
from app.db import get_conn

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    content: str
    citations: list[dict]
    tool_trace: list[dict]


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty.")
    result = run_turn(req.session_id, req.message)
    return ChatResponse(**result)


@router.get("/sessions")
def list_sessions() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM sessions ORDER BY last_active_at DESC").fetchall()
    return [dict(r) for r in rows]


@router.get("/{session_id}")
def get_session_messages(session_id: str) -> dict:
    with get_conn() as conn:
        session = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        messages = conn.execute(
            "SELECT message_id, role, content, citations, created_at FROM messages "
            "WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
    return {
        "session_id": session_id,
        "title": session["title"],
        "messages": [
            {
                "message_id": m["message_id"],
                "role": m["role"],
                "content": m["content"],
                "citations": json.loads(m["citations"] or "[]"),
                "created_at": m["created_at"],
            }
            for m in messages
        ],
    }
