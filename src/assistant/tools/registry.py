"""Lightweight tool-calling registry.

Tools are plain Python functions decorated with @tool(...). We generate
the Ollama/OpenAI-style function schema from the decorator args (kept
explicit rather than introspected from type hints, since offline models
are far more reliable with a hand-written schema than a guessed one).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from assistant.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str]

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, name: str, description: str, parameters: dict[str, Any]):
        def decorator(func: Callable[..., str]) -> Callable[..., str]:
            self._tools[name] = ToolSpec(name, description, parameters, func)
            return func
        return decorator

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]

    def has(self, name: str) -> bool:
        return name in self._tools

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Error: unknown tool '{name}'"
        try:
            return self._tools[name].func(**arguments)
        except Exception as e:  # noqa: BLE001 - a tool failure should never crash the assistant
            log.exception("Tool '%s' raised an error", name)
            return f"Error running tool '{name}': {e}"


registry = ToolRegistry()
