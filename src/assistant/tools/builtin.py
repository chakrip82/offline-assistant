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

_db = MemoryDB()


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
def read_notes(limit: int = 10) -> str:
    path = _notes_path()
    if not path.exists():
        return "No notes saved yet."
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return "\n".join(lines[-limit:]) if lines else "No notes saved yet."


@registry.register(
    name="remember_fact",
    description="Store a durable fact about the user (e.g. preferences) under a short key, for exact recall later.",
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
    _db.set_fact(key, value)
    return f"Remembered: {key} = {value}"


@registry.register(
    name="recall_fact",
    description="Look up a previously remembered fact by its key.",
    parameters={
        "type": "object",
        "properties": {"key": {"type": "string", "description": "The fact's key."}},
        "required": ["key"],
    },
)
def recall_fact(key: str) -> str:
    value = _db.get_fact(key)
    return value if value is not None else f"No fact stored under '{key}'."


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
