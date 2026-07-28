from tests import _env  # noqa: F401  must be first

import unittest
from pathlib import Path

from app.db import get_conn, init_db
from app.ingestion import IngestionError, ingest_uploaded_pdf
from app.vectorstore import get_vector_store

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


class TestIngestion(unittest.TestCase):
    def setUp(self):
        init_db()

    def test_ingest_creates_paper_chunks_and_vectors(self):
        result = ingest_uploaded_pdf(str(FIXTURE), title="Test Paper")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["page_count"], 2)
        self.assertGreater(result["chunk_count"], 0)

        with get_conn() as conn:
            paper = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (result["paper_id"],)).fetchone()
            chunks = conn.execute("SELECT * FROM chunks WHERE paper_id = ?", (result["paper_id"],)).fetchall()
        self.assertEqual(paper["status"], "ready")
        self.assertEqual(paper["title"], "Test Paper")
        self.assertEqual(len(chunks), result["chunk_count"])
        self.assertGreaterEqual(get_vector_store().count(), result["chunk_count"])

    def test_ingest_backfills_title_when_none_given(self):
        result = ingest_uploaded_pdf(str(FIXTURE))
        with get_conn() as conn:
            paper = conn.execute("SELECT title FROM papers WHERE paper_id = ?", (result["paper_id"],)).fetchone()
        self.assertNotEqual(paper["title"], "(untitled upload)")
        self.assertIn("Sparse Beamforming", paper["title"])

    def test_ingest_failure_marks_paper_failed_not_exception_swallowed(self):
        bad_path = "/tmp/paperpilot_test_not_a_pdf.pdf"
        Path(bad_path).write_text("not a real pdf")
        with self.assertRaises(IngestionError):
            ingest_uploaded_pdf(bad_path, title="Bad")
        with get_conn() as conn:
            row = conn.execute("SELECT status, error_message FROM papers WHERE title = 'Bad'").fetchone()
        self.assertEqual(row["status"], "failed")
        self.assertIsNotNone(row["error_message"])


if __name__ == "__main__":
    unittest.main()
