# Offline Assistant

A personal AI assistant that runs **entirely on your machine** — no
internet required at inference time. Text + voice, local memory,
local tool-calling.

## Architecture

```
Interface (CLI text / voice loop)
        │
Orchestrator  ── rolling history (SQLite) + semantic recall (Chroma)
        │
Local LLM (Ollama, e.g. Llama 3.1 8B, quantized)
        │
Tools (notes, facts, calculator, datetime — all local)
```

Nothing here calls out to a cloud API. The only network access needed
is a **one-time download** of the model weights and voices during setup.

## Requirements

- Python 3.10+
- ~8GB free RAM for an 8B quantized model (adjust `llm.model` in
  `config/default.yaml` if your machine is smaller/larger)
- [Ollama](https://ollama.com) for local inference

## Setup

```bash
git clone <this repo>  # or just use this folder as-is
cd offline-assistant
bash scripts/setup.sh
```

This installs Ollama, pulls the chat + embedding models, creates local
data directories, and installs the Python package.

## Run (text)

```bash
assistant
```

You'll get a plain chat loop. Ctrl+C or type `exit` to quit. It
remembers conversation history and durable facts across sessions,
and can call built-in tools (notes, calculator, fact storage) — try:

```
you > remember that my favorite editor is neovim
you > take a note: renew car insurance by end of month
you > what's 234 * 18?
```

## Run (voice)

```bash
pip install -e ".[voice]"
```

Then download a Piper voice (see `scripts/setup.sh` for exact URLs),
set `audio.enabled: true` in `config/config.yaml`, and run:

```bash
assistant-voice
```

It listens continuously, detects when you stop talking (VAD-based, no
push-to-talk needed), transcribes with Whisper, thinks, and speaks the
reply with Piper — all offline.

## Configuration

Defaults live in `config/default.yaml`. To override without editing
that file, create `config/config.yaml` (gitignored) with just the keys
you want to change, e.g.:

```yaml
llm:
  model: "qwen2.5:14b-instruct-q4_K_M"
audio:
  enabled: true
```

Or use environment variables: `ASSISTANT_LLM__MODEL=mistral:7b`.

## Project layout

```
src/assistant/
  config.py            # typed settings (YAML + env)
  logging_config.py    # rotating file + console logging
  llm/engine.py         # Ollama client: chat, embeddings, retries
  memory/db.py          # SQLite: conversation history + durable facts
  memory/vector_store.py# Chroma: semantic recall
  tools/registry.py     # tool-calling framework
  tools/builtin.py      # notes, calculator, facts, datetime
  core/orchestrator.py  # the actual conversation loop
  audio/stt.py           # mic capture + VAD + Whisper transcription
  audio/tts.py           # Piper speech synthesis
  interfaces/cli.py      # text entrypoint
  interfaces/voice_loop.py # voice entrypoint
tests/                  # unit tests, run with `pytest` (no ollama needed)
scripts/setup.sh        # one-time machine setup
```

## Adding a new tool

```python
# in tools/builtin.py or a new module imported by orchestrator.py
from assistant.tools.registry import registry

@registry.register(
    name="my_tool",
    description="What this does and when the model should use it.",
    parameters={
        "type": "object",
        "properties": {"arg": {"type": "string", "description": "..."}},
        "required": ["arg"],
    },
)
def my_tool(arg: str) -> str:
    return f"did something with {arg}"
```

The model sees the schema automatically on the next run — no other
wiring needed.

## Troubleshooting

- **"Cannot reach local model server"** → run `ollama serve` in another
  terminal, and confirm the model in `config/default.yaml` has been
  pulled (`ollama list`).
- **Voice mode complains about missing dependencies** → `pip install -e
  ".[voice]"`.
- **Slow responses** → drop to a smaller/more-quantized model
  (`llm.model` in config), or a smaller Whisper size (`audio.stt_model_size`).

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Tests use a fake LLM engine — no Ollama server or network needed to run them.
