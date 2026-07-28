"""
Mock LLM client.

Two modes:

1. Scripted mode (script=[...]) - returns a fixed sequence of LLMResponse
   objects, one per call to .create(), regardless of input. This is for
   precise unit tests of the orchestrator loop (e.g. "assert exactly two
   tool calls happen, in this order, and the loop terminates").

2. Heuristic mode (default, script=None) - a small rule-based responder
   that inspects the latest message and picks a plausible tool to call
   based on keywords, then on the next turn synthesizes a final answer
   from whatever tool_result it sees. This lets the *entire* app (API,
   routes, UI) be exercised end-to-end with PAPERPILOT_MOCK=1 and no
   API key, for demos and manual QA without spending real API credits.

Neither mode calls the network.
"""
from __future__ import annotations

import json
import re
from typing import Any

from app.llm.base import LLMClient, LLMResponse, TextBlock, ToolUseBlock

_ID_RE = re.compile(r"\b[0-9a-f]{8,32}\b")


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg["role"] != "user":
            continue
        content = msg["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            if texts:
                return " ".join(texts)
    return ""


def _last_message_has_tool_result(messages: list[dict[str, Any]]) -> list[dict] | None:
    if not messages:
        return None
    last = messages[-1]
    if last["role"] != "user" or not isinstance(last["content"], list):
        return None
    results = [b for b in last["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]
    return results or None


class MockLLMClient(LLMClient):
    def __init__(self, script: list[LLMResponse] | None = None):
        self._script = list(script) if script else None
        self._call_count = 0

    def create(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self._call_count += 1

        if self._script is not None:
            if not self._script:
                return LLMResponse(content=[TextBlock(text="(mock script exhausted)")], stop_reason="end_turn")
            return self._script.pop(0)

        return self._heuristic_response(messages, tools or [])

    def _heuristic_response(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        tool_results = _last_message_has_tool_result(messages)
        available = {t["name"] for t in tools}

        if tool_results:
            snippets = []
            for r in tool_results:
                content = r.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                    )
                snippets.append(str(content)[:220])
            joined = " | ".join(snippets)
            return LLMResponse(
                content=[
                    TextBlock(
                        text=(
                            "Based on the retrieved information, here is a grounded answer "
                            f"[mock synthesis]: {joined}"
                        )
                    )
                ],
                stop_reason="end_turn",
            )

        user_text = _last_user_text(messages)
        lower = user_text.lower()
        ids_found = _ID_RE.findall(user_text)

        if available == {"web_search"}:
            # Standalone search_papers_external() call: the real API would run
            # web_search server-side and return synthesized JSON text directly.
            # The mock fabricates a couple of plausible-looking results instead.
            topic = user_text.strip().strip('"').strip(".") or "the requested topic"
            mock_results = [
                {
                    "title": f"A Survey of Recent Approaches to {topic}",
                    "authors": "A. Researcher, B. Scholar",
                    "year": "2025",
                    "venue": "arXiv preprint",
                    "summary": f"[mock] An overview of recent methods related to {topic}.",
                    "url": "https://arxiv.org/abs/0000.00000",
                },
                {
                    "title": f"Empirical Advances in {topic}",
                    "authors": "C. Investigator et al.",
                    "year": "2024",
                    "venue": "Preprint",
                    "summary": f"[mock] Reports empirical results relevant to {topic}.",
                    "url": "https://arxiv.org/abs/0000.00001",
                },
            ]
            return LLMResponse(content=[TextBlock(text=json.dumps(mock_results))], stop_reason="end_turn")

        def tool_call(name: str, input_dict: dict) -> LLMResponse:
            return LLMResponse(
                content=[ToolUseBlock(id=f"toolu_mock_{self._call_count}", name=name, input=input_dict)],
                stop_reason="tool_use",
            )

        if "compare_papers" in available and "compare" in lower and len(ids_found) >= 2:
            return tool_call("compare_papers", {"paper_ids": ids_found[:4]})
        if "save_notes" in available and any(k in lower for k in ("note", "remember", "save this")):
            return tool_call("save_notes", {"content": user_text, "paper_id": ids_found[0] if ids_found else None})
        if "generate_report" in available and any(k in lower for k in ("literature review", "report", "write up")):
            return tool_call(
                "generate_report",
                {"title": "Auto-generated report", "report_type": "literature_review", "paper_ids": ids_found},
            )
        if "read_pdf" in available and "read" in lower and ids_found:
            return tool_call("read_pdf", {"paper_id": ids_found[0]})
        if "retrieve_context" in available:
            return tool_call("retrieve_context", {"query": user_text, "paper_ids": ids_found or None})

        return LLMResponse(content=[TextBlock(text=f"[mock] I don't have a tool to handle: {user_text}")], stop_reason="end_turn")
