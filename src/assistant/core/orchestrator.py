"""The conversation orchestrator.

Responsibilities per turn:
  1. Pull rolling recent history (SQLite) for continuity
  2. Pull semantically relevant older memory (Chroma) for recall
  3. Build the prompt (system + memory context + recent history + user turn)
  4. Call the local LLM, with tools offered
  5. If the model calls a tool, execute it locally and loop back in
  6. Persist the turn to both memory stores
  7. Return the final assistant text
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

from assistant.config import get_settings
from assistant.llm.engine import ChatMessage, ChatResult, LLMError, OllamaEngine, get_engine
from assistant.logging_config import get_logger
from assistant.memory.db import MemoryDB
from assistant.memory.vector_store import VectorMemory
from assistant.tools import builtin  # noqa: F401 - populates the registry on import
from assistant.tools.registry import registry

log = get_logger(__name__)

MAX_TOOL_HOPS = 4  # hard cap so a confused model can't loop forever


@dataclass
class TurnResult:
    text: str
    tool_calls_made: list[str]


class Orchestrator:
    def __init__(
        self,
        engine: OllamaEngine | None = None,
        db: MemoryDB | None = None,
        vector_memory: VectorMemory | None = None,
        session_id: str | None = None,
    ):
        self.settings = get_settings()
        self.engine = engine or get_engine()
        self.db = db or MemoryDB()
        self.vector_memory = vector_memory or VectorMemory(self.engine)
        self.session_id = session_id or str(uuid.uuid4())

    def _build_messages(self, user_text: str) -> list[ChatMessage]:
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self.settings.assistant.system_prompt)
        ]

        # Semantic recall from long-term memory, if anything relevant exists
        hits = self.vector_memory.recall(user_text)
        if hits:
            context = "\n".join(f"- {h.text}" for h in hits if h.score > 0.3)
            if context:
                messages.append(ChatMessage(
                    role="system",
                    content=f"Relevant things you remember about the user:\n{context}",
                ))

        # Rolling recent history for conversational continuity
        for turn in self.db.recent_turns(self.session_id, self.settings.memory.max_history_turns):
            messages.append(ChatMessage(role=turn.role, content=turn.content))

        messages.append(ChatMessage(role="user", content=user_text))
        return messages

    def handle_turn(self, user_text: str) -> TurnResult:
        self.db.add_turn(self.session_id, "user", user_text)
        messages = self._build_messages(user_text)
        tools_used: list[str] = []
        remembered_facts: list[str] = []

        for _ in range(MAX_TOOL_HOPS):
            try:
                result = self.engine.chat(messages, tools=registry.schemas())
            except LLMError as e:
                log.error("LLM call failed: %s", e)
                return TurnResult(text=f"[offline model unreachable] {e}", tool_calls_made=tools_used)

            if not result.tool_calls:
                self.db.add_turn(self.session_id, "assistant", result.content)
                self._maybe_store_long_term(user_text, result.content)
                return TurnResult(text=result.content, tool_calls_made=tools_used)

            # Model wants to call one or more tools - execute locally and loop
            messages.append(ChatMessage(role="assistant", content=result.content, tool_calls=result.tool_calls))
            for call in result.tool_calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments", {})
                args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
                log.info("Tool call: %s(%s)", name, args)
                tool_output = registry.call(name, args)
                tools_used.append(name)

                # Recalled facts are exact user-provided data. Return them directly
                # so the language model cannot alter their spelling or replace them
                # with unrelated world knowledge. Saved facts continue through the
                # loop, allowing the model to save additional facts from one turn.
                if name in {"recall_fact", "search_facts"}:
                    self.db.add_turn(self.session_id, "assistant", tool_output)
                    self._maybe_store_long_term(user_text, tool_output)
                    return TurnResult(text=tool_output, tool_calls_made=tools_used)

                if name == "remember_fact":
                    remembered_facts.append(tool_output)

                messages.append(ChatMessage(role="tool", name=name, content=tool_output))

            # The model can issue multiple remember_fact calls in one response.
            # Return their exact confirmations directly instead of allowing a
            # follow-up model response to invent a notes summary.
            if remembered_facts:
                confirmation = "\n".join(remembered_facts)
                self.db.add_turn(self.session_id, "assistant", confirmation)
                self._maybe_store_long_term(user_text, confirmation)
                return TurnResult(text=confirmation, tool_calls_made=tools_used)

        fallback = "I tried a few tool calls but couldn't land on an answer - could you rephrase?"
        self.db.add_turn(self.session_id, "assistant", fallback)
        return TurnResult(text=fallback, tool_calls_made=tools_used)

    def _maybe_store_long_term(self, user_text: str, assistant_text: str) -> None:
        """Store a compact summary of the exchange for future semantic recall.
        Simple heuristic for now: skip trivial small talk, store substantive turns."""
        if len(user_text.split()) < 4:
            return
        self.vector_memory.remember(
            f"User said: {user_text}\nAssistant replied: {assistant_text}",
            metadata={"session_id": self.session_id},
        )
