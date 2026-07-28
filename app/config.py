"""
Central configuration for PaperPilot.

All runtime settings are read from environment variables (optionally loaded
from a .env file by the process that starts uvicorn). Nothing here requires
network access at import time, so this module is safe to import in tests.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv_if_present() -> None:
    """Minimal .env loader (avoids a hard dependency on python-dotenv)."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv_if_present()


@dataclass(frozen=True)
class Settings:
    # --- LLM ---
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    # When true (or when no API key is set), the agent uses a deterministic
    # in-memory mock LLM instead of calling the real Anthropic API. This lets
    # the whole tool-calling loop be exercised in tests / CI without network
    # access or an API key.
    mock_llm: bool = field(default_factory=lambda: os.getenv("PAPERPILOT_MOCK", "").lower() in ("1", "true", "yes"))
    max_agent_steps: int = field(default_factory=lambda: int(os.getenv("PAPERPILOT_MAX_AGENT_STEPS", "6")))

    # --- Embeddings ---
    # "local"  -> zero-dependency deterministic hashing embedder (default; works offline)
    # "voyage" -> Voyage AI embeddings (Anthropic's recommended embedding partner)
    embedding_backend: str = field(default_factory=lambda: os.getenv("PAPERPILOT_EMBEDDING_BACKEND", "local"))
    voyage_api_key: str = field(default_factory=lambda: os.getenv("VOYAGE_API_KEY", ""))
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("PAPERPILOT_EMBEDDING_DIM", "256")))

    # --- Storage paths ---
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("PAPERPILOT_DATA_DIR", "data")).resolve())

    # --- Retrieval ---
    chunk_target_tokens: int = field(default_factory=lambda: int(os.getenv("PAPERPILOT_CHUNK_TOKENS", "400")))
    chunk_overlap_tokens: int = field(default_factory=lambda: int(os.getenv("PAPERPILOT_CHUNK_OVERLAP", "60")))
    retrieval_top_k: int = field(default_factory=lambda: int(os.getenv("PAPERPILOT_TOP_K", "8")))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "paperpilot.db"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def vectorstore_dir(self) -> Path:
        return self.data_dir / "vectorstore"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.uploads_dir, self.vectorstore_dir, self.exports_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
