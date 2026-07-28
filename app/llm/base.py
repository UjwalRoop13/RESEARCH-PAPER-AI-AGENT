"""
LLM client abstraction. The agent orchestrator only depends on this
interface, never on the Anthropic SDK directly - which is what makes it
possible to run the full tool-calling loop against MockLLMClient in tests
and CI without network access or an API key, and against
AnthropicLLMClient in production with zero changes to orchestrator code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class TextBlock:
    text: str


@dataclass
class LLMResponse:
    # content is an ordered list of TextBlock | ToolUseBlock, mirroring the
    # Anthropic Messages API content-block model.
    content: list[Any]
    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens" | ...
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.content if isinstance(b, TextBlock)).strip()

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


class LLMClient(ABC):
    @abstractmethod
    def create(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        raise NotImplementedError
