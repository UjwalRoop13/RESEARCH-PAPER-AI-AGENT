"""
Agent orchestrator - implements the Agent Loop described in the SRS
(Receive -> Reason -> Select Tool -> Act -> Observe -> ... -> Respond).

run_turn() is the single entry point the API layer calls. It:
  1. Loads prior conversation history for the session (state management).
  2. Repeatedly calls the LLM with the tool catalog until it stops
     requesting tools (bounded by settings.max_agent_steps - failure
     recovery against infinite loops).
  3. Executes each requested tool via TOOL_REGISTRY, logging every call
     (name, input, output, latency, status) to the tool_calls table for
     observability (FR-19) and feeds the result back as a tool_result.
  4. Persists the final user-visible message and returns it along with a
     best-effort extraction of citations and the tool-call trace, so the
     API/UI layer can show "how the agent got this answer".
"""
from __future__ import annotations

import json
import re
import time
import traceback
from typing import Any

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tool_schemas import ALL_TOOL_SCHEMAS, WEB_SEARCH_TOOL
from app.agent.tools import TOOL_REGISTRY, ToolError
from app.config import settings
from app.db import get_conn, new_id, now
from app.llm import get_llm_client
from app.llm.base import LLMClient, TextBlock, ToolUseBlock

_CITATION_RE = re.compile(r"\[([a-f0-9]{6,32})(?:,)?\s*p\.?\s*(\d+)\]", re.IGNORECASE)


def _to_api_blocks(content: list[Any]) -> list[dict[str, Any]]:
    blocks = []
    for b in content:
        if isinstance(b, TextBlock):
            blocks.append({"type": "text", "text": b.text})
        elif isinstance(b, ToolUseBlock):
            blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return blocks


def _extract_citations(text: str) -> list[dict[str, Any]]:
    out = []
    for match in _CITATION_RE.finditer(text):
        out.append({"paper_id": match.group(1), "page_number": int(match.group(2))})
    return out


def ensure_session(session_id: str | None) -> str:
    if session_id:
        with get_conn() as conn:
            row = conn.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row:
            return session_id
    sid = new_id()
    ts = now()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, title, created_at, last_active_at) VALUES (?,?,?,?)",
            (sid, "New session", ts, ts),
        )
    return sid


def _load_history(session_id: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def _persist_message(session_id: str, role: str, content: str, citations: list[dict] | None = None) -> str:
    mid = new_id()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (message_id, session_id, role, content, citations, created_at) VALUES (?,?,?,?,?,?)",
            (mid, session_id, role, content, json.dumps(citations or []), now()),
        )
        conn.execute("UPDATE sessions SET last_active_at = ? WHERE session_id = ?", (now(), session_id))
    return mid


def _log_tool_call(
    session_id: str,
    message_id: str | None,
    tool_name: str,
    input_dict: dict,
    output_dict: dict,
    status: str,
    error: str | None,
    latency_ms: int,
) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO tool_calls "
            "(tool_call_id, session_id, message_id, tool_name, input_json, output_json, status, error_message, latency_ms, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                new_id(), session_id, message_id, tool_name,
                json.dumps(input_dict)[:4000], json.dumps(output_dict)[:4000],
                status, error, latency_ms, now(),
            ),
        )


def run_turn(session_id: str | None, user_text: str, llm_client: LLMClient | None = None) -> dict[str, Any]:
    sid = ensure_session(session_id)
    llm = llm_client or get_llm_client()

    history = _load_history(sid)
    messages: list[dict[str, Any]] = [*history, {"role": "user", "content": user_text}]
    _persist_message(sid, "user", user_text)

    tools = [*ALL_TOOL_SCHEMAS, WEB_SEARCH_TOOL]
    tool_trace: list[dict[str, Any]] = []
    final_text = ""

    for _ in range(settings.max_agent_steps):
        response = llm.create(messages=messages, system=SYSTEM_PROMPT, tools=tools)

        if response.stop_reason != "tool_use" or not response.tool_uses:
            final_text = response.text or "(the agent returned no content)"
            break

        messages.append({"role": "assistant", "content": _to_api_blocks(response.content)})

        tool_result_blocks = []
        for tu in response.tool_uses:
            start = time.time()
            status, error, output = "ok", None, None
            try:
                if tu.name not in TOOL_REGISTRY:
                    raise ToolError(f"Unknown tool requested: {tu.name}")
                output = TOOL_REGISTRY[tu.name](tu.input)
            except ToolError as e:
                status, error = "error", str(e)
                output = {"error": str(e)}
            except Exception as e:  # unexpected internal failure - never crash the loop
                status, error = "error", f"internal error: {e}"
                traceback.print_exc()
                raise
            latency_ms = int((time.time() - start) * 1000)

            _log_tool_call(sid, None, tu.name, tu.input, output, status, error, latency_ms)
            tool_trace.append(
                {"tool_name": tu.name, "input": tu.input, "status": status, "latency_ms": latency_ms}
            )
            tool_result_blocks.append(
                {"type": "tool_result", "tool_use_id": tu.id, "content": json.dumps(output)[:8000]}
            )

        messages.append({"role": "user", "content": tool_result_blocks})
    else:
        final_text = (
            "I wasn't able to reach a final answer within the allotted reasoning steps. "
            "Try narrowing your question or asking about fewer papers at once."
        )

    citations = _extract_citations(final_text)
    message_id = _persist_message(sid, "assistant", final_text, citations)

    return {
        "session_id": sid,
        "message_id": message_id,
        "content": final_text,
        "citations": citations,
        "tool_trace": tool_trace,
    }
