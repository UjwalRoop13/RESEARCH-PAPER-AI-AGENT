"""
Standalone "Search Papers" capability (FR-04 / GET /api/papers/search).

This is deliberately independent of the chat orchestrator: it's a single
request/response utility, not a multi-turn conversation, so it doesn't
need session state or the full tool loop. It asks Claude to use the
built-in web_search tool and return strict JSON, then parses that JSON.
"""
from __future__ import annotations

import json
from typing import Any

from app.agent.tool_schemas import WEB_SEARCH_TOOL
from app.llm import get_llm_client

_SEARCH_SYSTEM_PROMPT = """You are a research paper discovery assistant. Search the web for real, \
currently findable academic or research papers matching the user's query. Respond with ONLY a raw \
JSON array, no markdown fences, no prose before or after. Each element must have exactly these keys: \
title, authors (short comma-separated string), year, venue, summary (max 25 words), url."""


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Model did not return a JSON array.")
    return json.loads(text[start : end + 1])


def search_papers_external(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    llm = get_llm_client()
    prompt = f'Find up to {max_results} real papers about: "{query}".'
    response = llm.create(
        messages=[{"role": "user", "content": prompt}],
        system=_SEARCH_SYSTEM_PROMPT,
        tools=[WEB_SEARCH_TOOL],
    )
    return _extract_json_array(response.text)
