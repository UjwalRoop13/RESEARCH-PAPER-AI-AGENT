"""
Relational metadata layer (Database Layer in the SRS).

Uses stdlib sqlite3 directly rather than an ORM: it's zero-dependency, easy
to reason about, and this schema is small enough that raw SQL stays
readable. Swapping to Postgres later means changing this module only -
callers interact through the functions below, not raw connections.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id      TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    authors       TEXT,              -- JSON-encoded list
    year          TEXT,
    venue         TEXT,
    source        TEXT NOT NULL,     -- 'upload' | 'web'
    source_url    TEXT,
    storage_path  TEXT,              -- path to the PDF on disk, if uploaded
    status        TEXT NOT NULL DEFAULT 'queued',  -- queued|processing|ready|failed
    page_count    INTEGER,
    error_message TEXT,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    paper_id      TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    text          TEXT NOT NULL,
    page_number   INTEGER,
    token_count   INTEGER,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);

CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    title         TEXT,
    created_at    REAL NOT NULL,
    last_active_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    message_id    TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role          TEXT NOT NULL,     -- user | assistant
    content       TEXT NOT NULL,
    citations     TEXT,              -- JSON-encoded list of {paper_id, chunk_id, page_number}
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

CREATE TABLE IF NOT EXISTS tool_calls (
    tool_call_id  TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    message_id    TEXT,
    tool_name     TEXT NOT NULL,
    input_json    TEXT,
    output_json   TEXT,
    status        TEXT NOT NULL,     -- ok | error
    error_message TEXT,
    latency_ms    INTEGER,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_toolcalls_session ON tool_calls(session_id);

CREATE TABLE IF NOT EXISTS notes (
    note_id       TEXT PRIMARY KEY,
    paper_id      TEXT REFERENCES papers(paper_id) ON DELETE CASCADE,
    chunk_id      TEXT,
    content       TEXT NOT NULL,
    tags          TEXT,              -- JSON-encoded list
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_paper ON notes(paper_id);

CREATE TABLE IF NOT EXISTS reports (
    report_id     TEXT PRIMARY KEY,
    session_id    TEXT,
    title         TEXT NOT NULL,
    report_type   TEXT NOT NULL,     -- summary | comparison | literature_review
    paper_ids     TEXT,              -- JSON-encoded list
    content_md    TEXT NOT NULL,
    created_at    REAL NOT NULL
);
"""


def new_id() -> str:
    return uuid.uuid4().hex


def now() -> float:
    return time.time()


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(settings.db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    target = str(db_path) if db_path else str(settings.db_path)
    conn = sqlite3.connect(target)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def rows_to_list(rows) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]
