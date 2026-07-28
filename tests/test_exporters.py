from tests import _env  # noqa: F401  must be first

import unittest

from app.exporters import export_docx_bytes, export_markdown_bytes, export_pdf_bytes

SAMPLE_MD = """# Title

## Section
Some paragraph text.

- bullet one
- bullet two
"""


class TestExporters(unittest.TestCase):
    def test_markdown_export_is_utf8_bytes(self):
        data = export_markdown_bytes(SAMPLE_MD)
        self.assertEqual(data.decode("utf-8"), SAMPLE_MD)

    def test_docx_export_produces_valid_zip_magic_bytes(self):
        data = export_docx_bytes(SAMPLE_MD, "Title")
        self.assertEqual(data[:2], b"PK")  # docx is a zip archive
        self.assertGreater(len(data), 1000)

    def test_pdf_export_produces_valid_pdf_header(self):
        data = export_pdf_bytes(SAMPLE_MD, "Title")
        self.assertTrue(data.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
