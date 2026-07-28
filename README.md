# PaperPilot

An autonomous AI research agent that helps you discover, read, compare, and
summarize academic papers - grounded in real retrieved text, with citations.

This is a working implementation of the PaperPilot SRS: a FastAPI backend
with a Claude-powered tool-calling agent, a RAG pipeline over uploaded
PDFs, and a small web UI, all runnable on a single machine with no external
infrastructure (no Postgres, no managed vector DB, no task queue).

## What's actually implemented

All six tools from the SRS work end to end:

| Tool | How |
|---|---|
| **Search Papers** | Claude's built-in `web_search` tool, used both inside chat and via a standalone `/api/papers/search` endpoint |
| **Read PDF** | `pypdf` + `pdfplumber` extraction, chunked and stored per page |
| **Retrieve Context** | Local vector store (numpy, cosine similarity) over chunk embeddings |
| **Compare Papers** | Structured multi-paper retrieval across standard comparison dimensions |
| **Save Notes** | Persisted to SQLite, linkable to a paper or chunk |
| **Generate Report** | Agent-authored Markdown, exportable as Markdown / DOCX / PDF |

Plus: multi-turn conversation memory, session management, tool-call
logging/observability, and graceful error handling on bad input, failed
ingestion, and tool failures.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
# (or leave it blank to run in mock mode - see below)

uvicorn app.main:app --reload
```

Open **http://localhost:8000**. Upload a PDF, wait a few seconds for it to
show a green "ready" dot in the sidebar, then ask it questions in the Chat
tab. Use the Search tab to discover papers on the open web. Ask the agent
to "compare paper X and paper Y", "save a note that...", or "write a
literature review across these papers" to exercise the other tools.

## Running without an API key (mock mode)

Every layer below the LLM call itself - ingestion, chunking, embeddings,
retrieval, the SQLite schema, the agent loop's control flow, tool
execution, and report export - is real and already tested (see
`./run_tests.sh`). To explore the *whole app*, including the UI and API,
without spending API credits or having a key yet:

```bash
PAPERPILOT_MOCK=1 uvicorn app.main:app --reload
```

The UI banner will read "MOCK MODE". A small rule-based mock LLM
(`app/llm/mock_client.py`) picks a plausible tool from your message's
keywords, calls the *real* tool implementation against your *real*
uploaded papers, and narrates the result. It's useful for demos and for
verifying the plumbing, but it does not reason the way Claude does - swap
in a real `ANTHROPIC_API_KEY` for actual answer quality.

## Running the tests

```bash
./run_tests.sh
```

37 tests across 6 modules, covering PDF extraction/chunking, embeddings,
the vector store, the full ingestion pipeline (including failure paths),
every individual tool, report export in all three formats, and - most
importantly - the agent orchestrator loop itself: single tool calls,
multi-step tool sequences, tool errors, the max-steps guard against
infinite loops, and cross-turn conversation memory. These run against
`MockLLMClient` in "scripted" mode, so they exercise the *exact* message
plumbing (tool_use/tool_result block construction) that the real Anthropic
API expects, without needing network access.

## Architecture

```
app/
├── config.py         Settings (env-driven, no network at import time)
├── db.py              SQLite schema + connection helper (relational metadata layer)
├── pdf_utils.py         PDF text extraction + sentence-aware chunking
├── embeddings.py         Embedder interface: local hashing (default) or Voyage AI
├── vectorstore.py          Local numpy-backed vector store (the "vector DB" layer)
├── ingestion.py              Upload -> extract -> chunk -> embed -> index pipeline
├── exporters.py                Markdown -> Markdown/DOCX/PDF report export
├── llm/
│   ├── base.py                   LLMClient interface (provider-agnostic)
│   ├── anthropic_client.py         Real Claude API implementation
│   └── mock_client.py               Deterministic offline mock (scripted + heuristic)
├── agent/
│   ├── tool_schemas.py               Tool JSON schemas Claude sees
│   ├── tools.py                       Tool implementations + TOOL_REGISTRY
│   ├── prompts.py                      System prompt (grounding, citations, anti-injection)
│   ├── orchestrator.py                  The agent loop itself
│   └── search.py                         Standalone external paper search (FR-04)
├── routes/                                 Thin FastAPI routers (papers, chat, notes, reports)
├── static/                                   Vanilla JS/CSS/HTML UI
└── main.py                                     FastAPI app assembly
```

Every layer is swappable behind the interface the rest of the app depends
on: change `LocalVectorStore` for pgvector/Pinecone, `LocalHashingEmbedder`
for Voyage/OpenAI, `MockLLMClient`/`AnthropicLLMClient` for a different
provider, or `sqlite3` for Postgres - without touching agent or tool code.
This directly implements the SRS's extensibility (FR-17) and
maintainability requirements.

### The agent loop

`app/agent/orchestrator.py::run_turn()` implements: receive -> reason ->
select tool -> act -> observe -> (repeat, bounded by
`PAPERPILOT_MAX_AGENT_STEPS`) -> respond. Every tool call is logged to the
`tool_calls` table with input, output, status, and latency for
observability. Citations are extracted from the final answer via the
`[paper_id p.N]` pattern the system prompt asks the model to use.

## Known simplifications (vs. the full production SRS)

This is a real, working MVP, not a production deployment. Specifically:

- **Ingestion runs synchronously** on upload rather than via a task queue.
  Fine for single-user/demo use; large PDFs or concurrent uploads would
  benefit from the Celery/queue design in the SRS. The ingestion function
  is already a self-contained unit of work, so this is a wrapper change,
  not a redesign.
- **The default embedder is a zero-dependency local hashing embedder**,
  not a true semantic model - Anthropic doesn't serve embeddings directly.
  It's good enough for lexical-overlap retrieval and for running
  everything offline/free, but production semantic quality needs
  `PAPERPILOT_EMBEDDING_BACKEND=voyage` (or swap in OpenAI embeddings
  behind the same `Embedder` interface).
- **The vector store is a single local file**, not a distributed index -
  fine up to tens of thousands of chunks, not built for multi-tenant scale.
- **No authentication** - every user shares one paper library, matching
  the SRS's "auth optional for MVP" framing (FR-01 is marked Should, not
  Must).
- **Scanned/OCR-only PDFs are out of scope** - ingestion fails gracefully
  with a clear error rather than silently producing empty chunks.
- **Conversation memory is full-history replay**, not summarized -fine for
  short sessions, would need trimming/summarization for very long ones.

None of these are hidden: `ingest_paper()` raises `IngestionError` with a
specific message on failure, `/api/health` reports which embedding backend
and LLM mode are active, and this section exists so you know exactly what
you're getting.
