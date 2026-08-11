"""Typed configuration for the assistant.

Loads config/default.yaml, then layers an optional local config.yaml,
then env vars prefixed with ASSISTANT_ (nested via double underscore,
e.g. ASSISTANT_LLM__MODEL=llama3.1:8b). This is the single source of
truth every other module imports from.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class LLMConfig(BaseModel):
    backend: str = "ollama"
    host: str = "http://127.0.0.1:11434"
    model: str = "llama3.1:8b-instruct-q4_K_M"
    embedding_model: str = "nomic-embed-text"
    temperature: float = 0.4
    max_tokens: int = 1024
    context_window: int = 8192
    request_timeout_s: int = 120


class MemoryConfig(BaseModel):
    data_dir: str = "data"
    sqlite_path: str = "data/assistant.db"
    vector_dir: str = "data/chroma"
    collection_name: str = "assistant_memory"
    max_history_turns: int = 20
    retrieval_top_k: int = 5


class AudioConfig(BaseModel):
    enabled: bool = False
    stt_model_size: str = "base.en"
    sample_rate: int = 16000
    vad_aggressiveness: int = 2
    tts_voice_path: str = "data/voices/en_US-lessac-medium.onnx"
    wake_word_enabled: bool = False
    wake_word: str = "assistant"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "data/logs"


class AssistantMeta(BaseModel):
    name: str = "Nova"
    system_prompt: str = "You are a helpful offline assistant."


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ASSISTANT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    assistant: AssistantMeta = Field(default_factory=AssistantMeta)

    def resolved_path(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@lru_cache
def get_settings() -> Settings:
    merged = _deep_merge(_load_yaml(DEFAULT_CONFIG_PATH), _load_yaml(LOCAL_CONFIG_PATH))
    settings = Settings(**merged)
    # ensure data dirs exist
    for rel in (settings.memory.data_dir, settings.logging.log_dir,
                str(Path(settings.memory.vector_dir))):
        settings.resolved_path(rel).mkdir(parents=True, exist_ok=True)
    return settings
