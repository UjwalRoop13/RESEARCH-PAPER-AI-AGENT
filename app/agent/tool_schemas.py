"""
Tool schemas exposed to the LLM, in Anthropic's tool-use JSON Schema
format. Adding a new tool means: (1) add its schema here, (2) implement it
in tools.py, (3) register it in TOOL_REGISTRY. The orchestrator and API
layer never need to change - this satisfies FR-17 (tool extensibility).

"Search Papers" (FR-04) is deliberately NOT a custom client tool: it is
implemented using Anthropic's built-in server-side web_search tool, which
Claude calls directly to find real, current papers on the open web. See
agent/orchestrator.py for how it's merged into the tools list sent to the
API.
"""
from __future__ import annotations

READ_PDF = {
    "name": "read_pdf",
    "description": (
        "Read the full extracted text of a paper that has already been uploaded and "
        "ingested into PaperPilot, identified by its paper_id. Use this when you need "
        "more context than the short chunks returned by retrieve_context - for example "
        "to check the full methodology section or verify a claim in context."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "paper_id": {"type": "string", "description": "The paper_id of an already-uploaded paper."},
            "max_pages": {
                "type": "integer",
                "description": "Optional cap on number of pages to return, to control context size. Defaults to all pages.",
            },
        },
        "required": ["paper_id"],
    },
}

RETRIEVE_CONTEXT = {
    "name": "retrieve_context",
    "description": (
        "Semantic search over the chunks of uploaded papers. Returns the most relevant "
        "passages for a natural-language query, each tagged with its paper_id and page "
        "number so you can cite it. Use this as your default way to ground answers - "
        "prefer it over read_pdf unless you specifically need full-document context."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The natural-language question or topic to search for."},
            "paper_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of paper_ids to restrict the search to. Omit to search all uploaded papers.",
            },
            "top_k": {"type": "integer", "description": "Number of chunks to return. Defaults to 8."},
        },
        "required": ["query"],
    },
}

COMPARE_PAPERS = {
    "name": "compare_papers",
    "description": (
        "Retrieve structured, side-by-side context for comparing two or more uploaded "
        "papers along standard dimensions (objective, method, dataset, key results, "
        "limitations). Returns the most relevant chunk per dimension per paper so you can "
        "synthesize a comparison table in your final answer. Requires at least 2 paper_ids."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "paper_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "description": "paper_ids of the papers to compare (2 or more).",
            },
            "dimensions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional custom comparison dimensions. Defaults to objective, method, dataset, results, limitations.",
            },
        },
        "required": ["paper_ids"],
    },
}

SAVE_NOTES = {
    "name": "save_notes",
    "description": (
        "Persist a note for the user, optionally linked to a specific paper or chunk. "
        "Use this when the user explicitly asks you to save, remember, or note something "
        "down for later reference."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The note text to save."},
            "paper_id": {"type": "string", "description": "Optional paper_id this note relates to."},
            "chunk_id": {"type": "string", "description": "Optional chunk_id this note is anchored to."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional free-form tags."},
        },
        "required": ["content"],
    },
}

GENERATE_REPORT = {
    "name": "generate_report",
    "description": (
        "Compile a formatted report (a summary, a comparison, or a full literature review) "
        "across one or more papers and persist it so the user can export it later as "
        "Markdown, PDF, or DOCX. Use this only when the user explicitly asks for a report, "
        "write-up, or literature review to be produced - not for ordinary Q&A."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Title for the report."},
            "report_type": {
                "type": "string",
                "enum": ["summary", "comparison", "literature_review"],
                "description": "The kind of report to produce.",
            },
            "paper_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "paper_ids to include in the report.",
            },
            "content_md": {
                "type": "string",
                "description": (
                    "The full Markdown content of the report, written by you. Write this "
                    "AFTER you have gathered enough grounded context via retrieve_context / "
                    "compare_papers - include citations like [paper_id p.N] inline."
                ),
            },
        },
        "required": ["title", "report_type", "paper_ids", "content_md"],
    },
}

ALL_TOOL_SCHEMAS = [READ_PDF, RETRIEVE_CONTEXT, COMPARE_PAPERS, SAVE_NOTES, GENERATE_REPORT]

# Anthropic's built-in server-side web search tool (implements FR-04, Search Papers).
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}
