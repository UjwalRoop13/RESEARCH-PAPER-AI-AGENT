from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db import init_db
from app.routes import chat, notes, papers, reports

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="PaperPilot",
    description="An autonomous AI research agent for discovering, understanding, comparing, and summarizing papers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP: same-origin UI is served by this app itself; tighten for production multi-origin use.
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    settings.ensure_dirs()
    init_db()


app.include_router(papers.router)
app.include_router(chat.router)
app.include_router(notes.router)
app.include_router(reports.router)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "mock_llm": settings.mock_llm or not bool(settings.anthropic_api_key),
        "embedding_backend": settings.embedding_backend,
    }


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))
