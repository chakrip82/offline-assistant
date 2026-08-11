"""Semantic memory via a local, embedded Chroma vector store.

No server, no network — persists to disk under data/chroma. Embeddings
come from the local model via the same Ollama engine (nomic-embed-text
by default), so this stays fully offline too.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import chromadb

from assistant.config import get_settings
from assistant.llm.engine import OllamaEngine
from assistant.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class MemoryHit:
    text: str
    score: float
    metadata: dict


class VectorMemory:
    def __init__(self, engine: OllamaEngine | None = None):
        settings = get_settings()
        self.settings = settings
        self.engine = engine or OllamaEngine(settings)
        self._client = chromadb.PersistentClient(
            path=str(settings.resolved_path(settings.memory.vector_dir))
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.memory.collection_name
        )

    def remember(self, text: str, metadata: dict | None = None) -> None:
        """Store a piece of text (a fact, a summarized exchange, a note) for later recall."""
        try:
            embedding = self.engine.embed(text)
        except Exception as e:  # noqa: BLE001 - degrade gracefully, never crash the chat loop
            log.warning("Skipping vector store write, embedding failed: %s", e)
            return
        self._collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{**(metadata or {}), "ts": time.time()}],
        )

    def recall(self, query: str, top_k: int | None = None) -> list[MemoryHit]:
        top_k = top_k or self.settings.memory.retrieval_top_k
        try:
            embedding = self.engine.embed(query)
        except Exception as e:  # noqa: BLE001
            log.warning("Skipping recall, embedding failed: %s", e)
            return []

        if self._collection.count() == 0:
            return []

        result = self._collection.query(query_embeddings=[embedding], n_results=min(top_k, self._collection.count()))
        hits: list[MemoryHit] = []
        docs = result.get("documents", [[]])[0]
        dists = result.get("distances", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        for doc, dist, meta in zip(docs, dists, metas):
            hits.append(MemoryHit(text=doc, score=1 - dist, metadata=meta or {}))
        return hits
