from tests import _env  # noqa: F401  must be first

import unittest
from pathlib import Path

from app.agent.orchestrator import run_turn
from app.db import get_conn, init_db
from app.ingestion import ingest_uploaded_pdf
from app.llm.base import LLMClient, LLMResponse, TextBlock, ToolUseBlock
from app.llm.mock_client import MockLLMClient

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


class InfiniteToolClient(LLMClient):
    """Always requests a tool call - used to test the max-steps guard."""

    def create(self, messages, system, tools=None):
        return LLMResponse(content=[ToolUseBlock(id="x", name="retrieve_context", input={"query": "x"})], stop_reason="tool_use")


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        init_db()
        self.paper = ingest_uploaded_pdf(str(FIXTURE), title="Beamforming Paper")
        self.paper_id = self.paper["paper_id"]

    def test_simple_end_turn_no_tools(self):
        script = [LLMResponse(content=[TextBlock(text="Hello there.")], stop_reason="end_turn")]
        result = run_turn(None, "hi", llm_client=MockLLMClient(script=script))
        self.assertEqual(result["content"], "Hello there.")
        self.assertEqual(result["tool_trace"], [])

    def test_single_tool_call_then_final_answer(self):
        script = [
            LLMResponse(
                content=[ToolUseBlock(id="t1", name="retrieve_context", input={"query": "complexity", "paper_ids": [self.paper_id]})],
                stop_reason="tool_use",
            ),
            LLMResponse(content=[TextBlock(text=f"Found it [{self.paper_id} p.1].")], stop_reason="end_turn"),
        ]
        result = run_turn(None, "What does the paper say?", llm_client=MockLLMClient(script=script))
        self.assertEqual(len(result["tool_trace"]), 1)
        self.assertEqual(result["tool_trace"][0]["tool_name"], "retrieve_context")
        self.assertEqual(result["tool_trace"][0]["status"], "ok")
        self.assertEqual(result["citations"], [{"paper_id": self.paper_id, "page_number": 1}])

    def test_multi_step_tool_sequence(self):
        script = [
            LLMResponse(content=[ToolUseBlock(id="t1", name="retrieve_context", input={"query": "x"})], stop_reason="tool_use"),
            LLMResponse(content=[ToolUseBlock(id="t2", name="save_notes", input={"content": "note"})], stop_reason="tool_use"),
            LLMResponse(content=[TextBlock(text="Done.")], stop_reason="end_turn"),
        ]
        result = run_turn(None, "do two things", llm_client=MockLLMClient(script=script))
        names = [t["tool_name"] for t in result["tool_trace"]]
        self.assertEqual(names, ["retrieve_context", "save_notes"])

    def test_tool_error_does_not_crash_loop(self):
        script = [
            LLMResponse(content=[ToolUseBlock(id="t1", name="read_pdf", input={"paper_id": "nonexistent"})], stop_reason="tool_use"),
            LLMResponse(content=[TextBlock(text="I could not find that paper.")], stop_reason="end_turn"),
        ]
        result = run_turn(None, "read paper nonexistent", llm_client=MockLLMClient(script=script))
        self.assertEqual(result["tool_trace"][0]["status"], "error")
        self.assertEqual(result["content"], "I could not find that paper.")

    def test_max_steps_guard_prevents_infinite_loop(self):
        from app.config import settings

        result = run_turn(None, "loop forever", llm_client=InfiniteToolClient())
        self.assertEqual(len(result["tool_trace"]), settings.max_agent_steps)
        self.assertIn("allotted reasoning steps", result["content"])

    def test_conversation_memory_persists_across_turns(self):
        r1 = run_turn(
            None, "My name is Priya.",
            llm_client=MockLLMClient(script=[LLMResponse(content=[TextBlock(text="Hi Priya!")], stop_reason="end_turn")]),
        )
        sid = r1["session_id"]
        r2 = run_turn(
            sid, "What is my name?",
            llm_client=MockLLMClient(script=[LLMResponse(content=[TextBlock(text="Your name is Priya.")], stop_reason="end_turn")]),
        )
        self.assertEqual(r2["session_id"], sid)
        with get_conn() as conn:
            rows = conn.execute("SELECT role FROM messages WHERE session_id = ? ORDER BY created_at", (sid,)).fetchall()
        self.assertEqual([r["role"] for r in rows], ["user", "assistant", "user", "assistant"])

    def test_unknown_session_id_creates_new_session_instead_of_crashing(self):
        result = run_turn(
            "does-not-exist",
            "hello",
            llm_client=MockLLMClient(script=[LLMResponse(content=[TextBlock(text="hi")], stop_reason="end_turn")]),
        )
        self.assertNotEqual(result["session_id"], "does-not-exist")


if __name__ == "__main__":
    unittest.main()
