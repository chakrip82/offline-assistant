"""Orchestrator logic tested against a fake engine - no ollama server,
no network, so this runs anywhere including CI."""
import tempfile
from pathlib import Path

import pytest

from assistant.core.orchestrator import Orchestrator
from assistant.llm.engine import ChatResult
from assistant.memory.db import MemoryDB


class FakeEngine:
    """Returns a canned reply with no tool calls, and a fixed embedding."""

    base_url = "fake://local"

    def __init__(self, reply: str = "Hello! How can I help?"):
        self.reply = reply
        self.calls = 0

    def health_check(self) -> bool:
        return True

    def chat(self, messages, tools=None, retries=2) -> ChatResult:
        self.calls += 1
        return ChatResult(content=self.reply, tool_calls=[])

    def embed(self, text: str) -> list[float]:
        return [0.0] * 8


class FakeVectorMemory:
    def recall(self, query, top_k=None):
        return []

    def remember(self, text, metadata=None):
        pass


@pytest.fixture
def tmp_db(tmp_path):
    return MemoryDB(path=str(tmp_path / "test.db"))


def test_handle_turn_returns_reply_and_persists_history(tmp_db):
    engine = FakeEngine(reply="Sure, I can help with that.")
    orch = Orchestrator(engine=engine, db=tmp_db, vector_memory=FakeVectorMemory(), session_id="s1")

    result = orch.handle_turn("What's the weather like offline?")

    assert result.text == "Sure, I can help with that."
    assert result.tool_calls_made == []

    history = tmp_db.recent_turns("s1", 10)
    roles = [t.role for t in history]
    assert roles == ["user", "assistant"]


def test_handle_turn_isolates_sessions(tmp_db):
    engine = FakeEngine()
    orch1 = Orchestrator(engine=engine, db=tmp_db, vector_memory=FakeVectorMemory(), session_id="a")
    orch2 = Orchestrator(engine=engine, db=tmp_db, vector_memory=FakeVectorMemory(), session_id="b")

    orch1.handle_turn("hello from session a")
    orch2.handle_turn("hello from session b")

    assert len(tmp_db.recent_turns("a", 10)) == 2
    assert len(tmp_db.recent_turns("b", 10)) == 2
