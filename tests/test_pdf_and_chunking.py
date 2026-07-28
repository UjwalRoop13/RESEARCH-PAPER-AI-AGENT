from tests import _env  # noqa: F401  must be first

import unittest
from pathlib import Path

from app.pdf_utils import chunk_pages, extract_text, guess_metadata_from_first_page

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


class TestPdfExtraction(unittest.TestCase):
    def test_extracts_all_pages_with_text(self):
        result = extract_text(FIXTURE)
        self.assertEqual(result.page_count, 2)
        self.assertEqual(result.empty_pages, [])
        self.assertIn("Sparse Beamforming", result.pages[0].text)
        self.assertIn("Related Work", result.pages[1].text)

    def test_metadata_guess_uses_first_line(self):
        result = extract_text(FIXTURE)
        guess = guess_metadata_from_first_page(result.pages)
        self.assertTrue(guess["title_guess"].startswith("Sparse Beamforming"))

    def test_missing_file_returns_empty_result_not_exception(self):
        result = extract_text("/tmp/does_not_exist_at_all.pdf")
        self.assertEqual(result.page_count, 0)


class TestChunking(unittest.TestCase):
    def setUp(self):
        self.pages = extract_text(FIXTURE).pages

    def test_chunks_never_cross_page_boundary(self):
        chunks = chunk_pages(self.pages, target_tokens=30, overlap_tokens=8)
        page1_chunks = [c for c in chunks if c.page_number == 1]
        page2_chunks = [c for c in chunks if c.page_number == 2]
        self.assertTrue(page1_chunks)
        self.assertTrue(page2_chunks)
        for c in chunks:
            self.assertIn(c.page_number, (1, 2))

    def test_chunk_indices_are_sequential(self):
        chunks = chunk_pages(self.pages, target_tokens=30, overlap_tokens=8)
        indices = [c.chunk_index for c in chunks]
        self.assertEqual(indices, list(range(len(chunks))))

    def test_larger_target_tokens_produces_fewer_chunks(self):
        small = chunk_pages(self.pages, target_tokens=15, overlap_tokens=4)
        large = chunk_pages(self.pages, target_tokens=200, overlap_tokens=20)
        self.assertGreater(len(small), len(large))

    def test_empty_pages_produce_no_chunks(self):
        from app.pdf_utils import PageText

        chunks = chunk_pages([PageText(page_number=1, text="")])
        self.assertEqual(chunks, [])


if __name__ == "__main__":
    unittest.main()
