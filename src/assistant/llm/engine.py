"""Local LLM engine.

Talks to a locally running Ollama server (http://127.0.0.1:11434) — this
is the only "backend" needed on the machine: no cloud calls, ever. Ollama
itself downloads and runs models fully offline once pulled.

If you'd rather not run the ollama daemon, backend="llama_cpp" is stubbed
below for swapping in llama-cpp-python directly.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

from assistant.config import get_settings
from assistant.logging_config import get_logger

log = get_logger(__name__)


class LLMError(RuntimeError):
    """Raised when the local model backend is unreachable or errors out."""


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: list[dict] | None = None
    name: str | None = None  # tool name, when role == "tool"


@dataclass
class ChatResult:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    raw: dict[str, Any] | None = None


class OllamaEngine:
    """Thin, resilient client for Ollama's /api/chat endpoint."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.llm.host.rstrip("/")

    def health_check(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        retries: int = 2,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.settings.llm.model,
            "messages": [self._to_wire(m) for m in messages],
            "stream": False,
            "options": {
                "temperature": self.settings.llm.temperature,
                "num_ctx": self.settings.llm.context_window,
                "num_predict": self.settings.llm.max_tokens,
            },
        }
        if tools:
            payload["tools"] = tools

        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.settings.llm.request_timeout_s,
                )
                resp.raise_for_status()
                data = resp.json()
                msg = data.get("message", {})
                return ChatResult(
                    content=msg.get("content", ""),
                    tool_calls=msg.get("tool_calls", []) or [],
                    raw=data,
                )
            except requests.RequestException as e:
                last_err = e
                wait = 1.5 * (attempt + 1)
                log.warning("LLM call failed (attempt %d/%d): %s - retrying in %.1fs",
                            attempt + 1, retries + 1, e, wait)
                time.sleep(wait)

        raise LLMError(
            f"Could not reach local model at {self.base_url}. "
            f"Is `ollama serve` running and has the model been pulled? "
            f"(model={self.settings.llm.model}) Last error: {last_err}"
        )

    def embed(self, text: str) -> list[float]:
        try:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.settings.llm.embedding_model, "prompt": text},
                timeout=self.settings.llm.request_timeout_s,
            )
            resp.raise_for_status()
            return resp.json().get("embedding", [])
        except requests.RequestException as e:
            raise LLMError(f"Embedding call failed: {e}") from e

    @staticmethod
    def _to_wire(m: ChatMessage) -> dict[str, Any]:
        out: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            out["tool_calls"] = m.tool_calls
        if m.name:
            out["name"] = m.name
        return out


def get_engine() -> OllamaEngine:
    settings = get_settings()
    if settings.llm.backend != "ollama":
        raise NotImplementedError(
            f"Backend '{settings.llm.backend}' not implemented yet — only 'ollama' is wired up."
        )
    return OllamaEngine(settings)
