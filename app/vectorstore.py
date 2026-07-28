"""
Vector store (Database Layer > Vector DB in the SRS).

This is a minimal, dependency-free vector index backed by a single numpy
array plus a JSON sidecar for metadata, persisted to disk under
data/vectorstore/. It supports add, delete-by-paper, and cosine-similarity
top-k search with optional metadata filtering (paper_id allow-list).

This is intentionally swappable: the interface (add, search, delete_paper)
is what the rest of the app depends on. A production deployment can
replace this with pgvector, Pinecone, or Chroma behind the same interface
without touching agent/tool code.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import settings


@dataclass
class SearchResult:
    chunk_id: str
    paper_id: str
    score: float
    page_number: int | None
    text: str


class LocalVectorStore:
    def __init__(self, persist_dir: Path | None = None):
        self._dir = persist_dir or settings.vectorstore_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._vectors_path = self._dir / "vectors.npy"
        self._meta_path = self._dir / "meta.json"
        self._lock = threading.Lock()
        self._vectors: np.ndarray | None = None  # (n, dim)
        self._meta: list[dict] = []  # parallel list: chunk_id, paper_id, page_number, text
        self._load()

    def _load(self) -> None:
        if self._vectors_path.exists() and self._meta_path.exists():
            self._vectors = np.load(self._vectors_path)
            self._meta = json.loads(self._meta_path.read_text())
        else:
            self._vectors = None
            self._meta = []

    def _save(self) -> None:
        if self._vectors is not None:
            np.save(self._vectors_path, self._vectors)
        self._meta_path.write_text(json.dumps(self._meta))

    def add(self, chunk_id: str, paper_id: str, page_number: int | None, text: str, vector: np.ndarray) -> None:
        with self._lock:
            vector = vector.reshape(1, -1).astype(np.float32)
            if self._vectors is None:
                self._vectors = vector
            else:
                self._vectors = np.vstack([self._vectors, vector])
            self._meta.append(
                {"chunk_id": chunk_id, "paper_id": paper_id, "page_number": page_number, "text": text}
            )
            self._save()

    def add_batch(self, items: list[dict], vectors: np.ndarray) -> None:
        """items: list of {chunk_id, paper_id, page_number, text}, aligned with vectors rows."""
        with self._lock:
            vectors = vectors.astype(np.float32)
            if self._vectors is None:
                self._vectors = vectors
            else:
                self._vectors = np.vstack([self._vectors, vectors])
            self._meta.extend(items)
            self._save()

    def delete_paper(self, paper_id: str) -> int:
        with self._lock:
            if self._vectors is None:
                return 0
            keep_idx = [i for i, m in enumerate(self._meta) if m["paper_id"] != paper_id]
            removed = len(self._meta) - len(keep_idx)
            if removed == 0:
                return 0
            self._vectors = self._vectors[keep_idx] if keep_idx else None
            self._meta = [self._meta[i] for i in keep_idx]
            self._save()
            return removed

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 8,
        paper_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        with self._lock:
            if self._vectors is None or len(self._meta) == 0:
                return []
            query_vector = query_vector.reshape(-1).astype(np.float32)
            qnorm = np.linalg.norm(query_vector)
            if qnorm == 0:
                return []
            query_vector = query_vector / qnorm

            if paper_ids:
                allow = set(paper_ids)
                idxs = [i for i, m in enumerate(self._meta) if m["paper_id"] in allow]
                if not idxs:
                    return []
                sub_vectors = self._vectors[idxs]
                sub_meta = [self._meta[i] for i in idxs]
            else:
                sub_vectors = self._vectors
                sub_meta = self._meta

            scores = sub_vectors @ query_vector
            top_k = min(top_k, len(sub_meta))
            if top_k <= 0:
                return []
            top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

            results = []
            for i in top_idx:
                m = sub_meta[int(i)]
                results.append(
                    SearchResult(
                        chunk_id=m["chunk_id"],
                        paper_id=m["paper_id"],
                        score=float(scores[int(i)]),
                        page_number=m.get("page_number"),
                        text=m["text"],
                    )
                )
            return results

    def count(self) -> int:
        return len(self._meta)


_store_instance: LocalVectorStore | None = None


def get_vector_store() -> LocalVectorStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = LocalVectorStore()
    return _store_instance
