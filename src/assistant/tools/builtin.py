"""A small set of genuinely offline-safe built-in tools.

Import this module once at startup (orchestrator does it) so the
@registry.register decorators run and populate the tool registry.
"""
from __future__ import annotations

import ast
import datetime as dt
import operator
from pathlib import Path

from assistant.config import get_settings
from assistant.memory.db import MemoryDB
from assistant.tools.registry import registry
from assistant.memory.vector_store import VectorMemory

_db = MemoryDB()
_fact_memory = VectorMemory(
    collection_name="assistant_facts",
    collection_metadata={"hnsw:space": "cosine"},
)


def _notes_path() -> Path:
    settings = get_settings()
    p = settings.resolved_path(settings.memory.data_dir) / "notes.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@registry.register(
    name="take_note",
    description="Save a short note to the user's local notes file for later reference.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "The note content to save."}},
        "required": ["text"],
    },
)
def take_note(text: str) -> str:
    path = _notes_path()
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"- [{timestamp}] {text}\n")
    return f"Saved note: {text}"


@registry.register(
    name="read_notes",
    description="Read back the most recent saved notes.",
    parameters={
        "type": "object",
        "properties": {"limit": {"type": "integer", "description": "How many recent notes to return (default 10)."}},
        "required": [],
    },
)
def read_notes(limit: int | str = 10) -> str:
    """Return recent notes while tolerating string arguments from local models."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 10

    limit = max(1, min(limit, 100))
    path = _notes_path()
    if not path.exists():
        return "No notes saved yet."
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return "\n".join(lines[-limit:]) if lines else "No notes saved yet."


@registry.register(
    name="remember_fact",
    description=(
        "Store one durable fact about the user under a short key for exact recall later. "
        "When the user provides multiple distinct facts, call this tool separately "
        "for each fact."
    ),
    parameters={
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Short identifier, e.g. 'favorite_editor'."},
            "value": {"type": "string", "description": "The fact to remember."},
        },
        "required": ["key", "value"],
    },
)
def remember_fact(key: str, value: str) -> str:
    normalized_key = key.strip().lower().replace(" ", "_").replace("-", "_")
    clean_value = value.strip()

    _db.set_fact(normalized_key, clean_value)
    _fact_memory.upsert_fact(normalized_key, clean_value)

    readable_key = normalized_key.replace("_", " ")
    return f"I'll remember that your {readable_key} is {clean_value}."


@registry.register(
    name="recall_fact",
    description=(
        "Recall a previously remembered fact. Pass the user's full natural-language "
        "question, for example 'Who is my son?' or 'What is my favorite color?'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's natural-language memory question."
            }
        },
        "required": ["query"],
    },
)
def recall_fact(query: str) -> str:
    normalized_query = query.strip().lower().replace(" ", "_").replace("-", "_")

    # First: fast exact lookup for a real key such as "son_name".
    value = _db.get_fact(normalized_query)
    if value is not None:
        return value

    # Second: semantic lookup for questions such as "Who is my son?"
    hits = _fact_memory.recall(query, top_k=1)
    if hits and hits[0].score > get_settings().memory.fact_retrieval_threshold:
        fact_key = hits[0].metadata.get("key")
        if fact_key:
            value = _db.get_fact(fact_key)
            if value is not None:
                return value

    return "I don't have a matching remembered fact."


@registry.register(
    name="search_facts",
    description=(
        "Find multiple remembered facts related to a natural-language query. "
        "Use this for list questions, such as 'Who are my teammates at Atlassian?' "
        "Do not use it when the user asks for one specific fact."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The natural-language description of the facts to find.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of facts to return; default 10.",
            },
            "key_prefix": {
                "type": "string",
                "description": (
                    "Optional normalized fact-key prefix for an exact group filter, "
                    "for example 'atlassian_team' for Atlassian teammates."
                ),
            },
        },
        "required": ["query"],
    },
)
def search_facts(
    query: str,
    limit: int | str = 10,
    key_prefix: str | None = None,
) -> str:
    """Return all confidently matching durable facts for a list-style question."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 10

    limit = max(1, min(limit, 20))

    # A key prefix is the most reliable way to list a known group. It avoids
    # semantically related but incorrect results, such as a manager appearing
    # in a request for teammates.
    if key_prefix:
        normalized_prefix = key_prefix.strip().lower().replace(" ", "_").replace("-", "_")
        matches = [
            value
            for key, value in _db.all_facts().items()
            if key.startswith(normalized_prefix)
        ][:limit]
        if matches:
            return "\n".join(f"- {value}" for value in matches)
        return "I don't have matching remembered facts."

    threshold = get_settings().memory.fact_retrieval_threshold
    hits = _fact_memory.recall(query, top_k=limit)
    matches: list[str] = []
    seen_keys: set[str] = set()

    for hit in hits:
        if hit.score < threshold:
            continue

        fact_key = hit.metadata.get("key")
        if not fact_key or fact_key in seen_keys:
            continue

        value = _db.get_fact(fact_key)
        if value is not None:
            matches.append(value)
            seen_keys.add(fact_key)

    if not matches:
        return "I don't have matching remembered facts."

    return "\n".join(f"- {value}" for value in matches)


_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos, ast.FloorDiv: operator.floordiv,
}


def _safe_eval(node) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


@registry.register(
    name="calculator",
    description="Evaluate a basic arithmetic expression (+, -, *, /, %, **). Use for any math instead of guessing.",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "e.g. '(12 + 8) * 3'"}},
        "required": ["expression"],
    },
)
def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval").body
        return str(_safe_eval(tree))
    except Exception:
        return "Error: could not evaluate that expression."


@registry.register(
    name="current_datetime",
    description="Get the current local date and time.",
    parameters={"type": "object", "properties": {}, "required": []},
)
def current_datetime() -> str:
    return dt.datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")
