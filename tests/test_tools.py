from tests import _env  # noqa: F401  must be first

import unittest
from pathlib import Path

from app.agent.tools import ToolError, compare_papers, generate_report, read_pdf, retrieve_context, save_notes
from app.db import init_db
from app.ingestion import ingest_uploaded_pdf

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


class TestTools(unittest.TestCase):
    def setUp(self):
        init_db()
        self.paper = ingest_uploaded_pdf(str(FIXTURE), title="Paper A")
        self.paper_id = self.paper["paper_id"]
        self.other_paper = ingest_uploaded_pdf(str(FIXTURE), title="Paper B")
        self.other_paper_id = self.other_paper["paper_id"]

    def test_retrieve_context_returns_ranked_results(self):
        out = retrieve_context({"query": "beamforming complexity reduction", "paper_ids": [self.paper_id]})
        self.assertGreater(len(out["results"]), 0)
        self.assertEqual(out["results"][0]["paper_id"], self.paper_id)

    def test_read_pdf_returns_full_text_with_page_tags(self):
        out = read_pdf({"paper_id": self.paper_id})
        self.assertIn("[Page 1]", out["text"])
        self.assertEqual(out["title"], "Paper A")

    def test_read_pdf_unknown_paper_raises_tool_error(self):
        with self.assertRaises(ToolError):
            read_pdf({"paper_id": "nonexistent"})

    def test_compare_papers_requires_two_ids(self):
        with self.assertRaises(ToolError):
            compare_papers({"paper_ids": [self.paper_id]})

    def test_compare_papers_returns_all_dimensions(self):
        out = compare_papers({"paper_ids": [self.paper_id, self.other_paper_id]})
        self.assertIn("objective", out["comparison"])
        self.assertIn(self.paper_id, out["comparison"]["objective"])
        self.assertIn(self.other_paper_id, out["comparison"]["objective"])

    def test_compare_papers_unknown_id_raises(self):
        with self.assertRaises(ToolError):
            compare_papers({"paper_ids": [self.paper_id, "nonexistent"]})

    def test_save_notes_persists_and_returns_note_id(self):
        out = save_notes({"content": "Interesting finding", "paper_id": self.paper_id, "tags": ["important"]})
        self.assertIn("note_id", out)
        self.assertEqual(out["tags"], ["important"])

    def test_save_notes_invalid_paper_id_raises(self):
        with self.assertRaises(ToolError):
            save_notes({"content": "orphan note", "paper_id": "nonexistent"})

    def test_generate_report_persists_report(self):
        out = generate_report(
            {
                "title": "My Review",
                "report_type": "summary",
                "paper_ids": [self.paper_id],
                "content_md": "# My Review\n\nSome content.",
            }
        )
        self.assertIn("report_id", out)
        self.assertEqual(out["report_type"], "summary")


if __name__ == "__main__":
    unittest.main()
