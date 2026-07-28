"""
Embedding layer (RAG Design > Embedding generation).

Anthropic does not itself serve an embeddings endpoint, so this module
exposes a small Embedder interface with two implementations:

- LocalHashingEmbedder (default): a deterministic, dependency-free
  feature-hashing bag-of-words embedder. It captures lexical/term overlap
  well enough to make retrieval, chunking, and the agent loop fully
  testable offline, but it is NOT a substitute for a real semantic
  embedding model in production.
- VoyageEmbedder: calls Voyage AI's embeddings API (Anthropic's recommended
  embedding partner). Requires VOYAGE_API_KEY and network access. This is
  what a production deployment should switch to via
  PAPERPILOT_EMBEDDING_BACKEND=voyage.

Swapping backends never touches calling code - everything goes through
get_embedder().embed_texts(...).
"""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np

from app.config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array, L2-normalized row-wise."""
        raise NotImplementedError


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    unigrams = _TOKEN_RE.findall(text)
    bigrams = [f"{a}_{b}" for a, b in zip(unigrams, unigrams[1:])]
    return unigrams + bigrams


def _stable_hash_index(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


class LocalHashingEmbedder(Embedder):
    def __init__(self, dim: int | None = None):
        self.dim = dim or settings.embedding_dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _tokenize(text)
            if not tokens:
                continue
            for tok in tokens:
                idx = _stable_hash_index(tok, self.dim)
                # sign trick reduces hash-collision bias (standard feature hashing)
                sign = 1.0 if (hash(tok) & 1) == 0 else -1.0
                # use blake2b for the sign too, so it's stable across processes
                sign_digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=1).digest()[0]
                sign = 1.0 if sign_digest % 2 == 0 else -1.0
                out[row, idx] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class VoyageEmbedder(Embedder):
    def __init__(self, model: str = "voyage-3"):
        try:
            import voyageai
        except ImportError as e:
            raise RuntimeError(
                "voyageai package not installed. Run: pip install voyageai"
            ) from e
        if not settings.voyage_api_key:
            raise RuntimeError("VOYAGE_API_KEY is not set.")
        self._client = voyageai.Client(api_key=settings.voyage_api_key)
        self._model = model
        self.dim = 1024  # voyage-3 default output dimension

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        result = self._client.embed(texts, model=self._model, input_type="document")
        arr = np.array(result.embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


_embedder_instance: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder_instance
    if _embedder_instance is not None:
        return _embedder_instance
    if settings.embedding_backend == "voyage":
        _embedder_instance = VoyageEmbedder()
    else:
        _embedder_instance = LocalHashingEmbedder()
    return _embedder_instance
