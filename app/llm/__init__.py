from __future__ import annotations

from app.config import settings
from app.llm.base import LLMClient


def get_llm_client() -> LLMClient:
    if settings.mock_llm or not settings.anthropic_api_key:
        from app.llm.mock_client import MockLLMClient

        return MockLLMClient()
    from app.llm.anthropic_client import AnthropicLLMClient

    return AnthropicLLMClient()
