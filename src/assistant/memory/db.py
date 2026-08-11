"""Structured local memory: conversation turns + durable key/value facts.

SQLite, single file, no server. This is the source of truth for exact
lookups ("what did I say my flight number was"); semantic/fuzzy recall
lives in vector_store.py instead.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass

from assistant.config import get_settings
from assistant.logging_config import get_logger

log = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversation_session ON conversation(session_id);
"""


@dataclass
class Turn:
    role: str
    content: str
    created_at: float


class MemoryDB:
    def __init__(self, path: str | None = None):
        settings = get_settings()
        self.path = str(settings.resolved_path(path or settings.memory.sqlite_path))
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    # -- conversation history -------------------------------------------------

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO conversation (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, time.time()),
            )

    def recent_turns(self, session_id: str, limit: int) -> list[Turn]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM conversation "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [Turn(r["role"], r["content"], r["created_at"]) for r in reversed(rows)]

    # -- durable facts ----------------------------------------------------------

    def set_fact(self, key: str, value: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO facts (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (key, value, time.time()),
            )

    def get_fact(self, key: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM facts WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def all_facts(self) -> dict[str, str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM facts").fetchall()
        return {r["key"]: r["value"] for r in rows}
