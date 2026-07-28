"""
Import this module FIRST, before any `app.*` import, in every test file.

It points PaperPilot at a fresh temporary data directory and forces mock
LLM mode, so tests never touch a real database/vector store shared with
other runs, and never require network access or an API key.

Each test file should be run in its own process (see run_tests.sh) since
app.config.settings is a module-level singleton created at first import.
"""
import os
import tempfile

_tmp_dir = tempfile.mkdtemp(prefix="paperpilot_test_")
os.environ["PAPERPILOT_DATA_DIR"] = _tmp_dir
os.environ["PAPERPILOT_MOCK"] = "1"
os.environ["PAPERPILOT_EMBEDDING_BACKEND"] = "local"
os.environ.setdefault("PAPERPILOT_EMBEDDING_DIM", "128")
