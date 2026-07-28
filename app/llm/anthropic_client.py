"""
Anthropic-backed LLM client. This is what runs in production. It requires
the `anthropic` package and a valid ANTHROPIC_API_KEY - neither is
available in this sandbox, so this module is exercised by integration
testing on the developer's own machine (see README), not by the offline
test suite, which uses MockLLMClient instead.
"""
from __future__ import annotations

from typing import Any

from app.config import settings
from app.llm.base import LLMClient, LLMResponse, TextBlock, ToolUseBlock


class AnthropicLLMClient(LLMClient):
    def __init__(self, model: str | None = None):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "The 'anthropic' package is not installed. Run: pip install anthropic"
            ) from e
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Set it in your environment or .env file, "
                "or set PAPERPILOT_MOCK=1 to run against the mock LLM instead."
            )
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = model or settings.anthropic_model

    def create(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=2000,
            system=system,
            messages=messages,
        )
        if tools:
            kwargs["tools"] = tools

        response = self._client.messages.create(**kwargs)

        content: list[Any] = []
        for block in response.content:
            if block.type == "text":
                content.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                content.append(ToolUseBlock(id=block.id, name=block.name, input=block.input))
            # server_tool_use / web_search_tool_result blocks (from the
            # built-in web_search tool) are intentionally passed through
            # unmodeled here - Anthropic manages that tool's execution
            # server-side, so the orchestrator never needs to act on it.

        usage = {}
        if getattr(response, "usage", None):
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        return LLMResponse(content=content, stop_reason=response.stop_reason, usage=usage)
