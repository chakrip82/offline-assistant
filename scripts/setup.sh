#!/usr/bin/env bash
# Run this once on the machine that will actually run the assistant.
# It is NOT meant to run inside a sandboxed build environment - it
# installs real binaries and downloads multi-GB model weights.
set -euo pipefail

echo "== Offline Assistant setup =="

if ! command -v ollama &> /dev/null; then
  echo "-> Installing Ollama..."
  if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "   On macOS: download from https://ollama.com/download and install, then re-run this script."
    exit 1
  else
    curl -fsSL https://ollama.com/install.sh | sh
  fi
else
  echo "-> Ollama already installed."
fi

echo "-> Starting ollama server in the background (if not already running)..."
(ollama serve > /tmp/ollama.log 2>&1 &) || true
sleep 2

MODEL="llama3.1:8b-instruct-q4_K_M"
EMBED_MODEL="nomic-embed-text"

echo "-> Pulling chat model: ${MODEL} (several GB, one-time download)..."
ollama pull "${MODEL}"

echo "-> Pulling embedding model: ${EMBED_MODEL}..."
ollama pull "${EMBED_MODEL}"

echo "-> Creating local data directories..."
mkdir -p data/chroma data/logs data/voices

echo "-> Installing Python package (text mode)..."
python3.11 -m pip install -e .

echo ""
echo "Setup complete for TEXT mode. Try: assistant"
echo ""
echo "For VOICE mode, additionally run:"
echo "  pip install -e \".[voice]\""
echo "  # then download a Piper voice, e.g.:"
echo "  curl -L -o data/voices/en_US-lessac-medium.onnx \\"
echo "    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
echo "  curl -L -o data/voices/en_US-lessac-medium.onnx.json \\"
echo "    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
echo "  # then set audio.enabled: true in config/config.yaml"
echo "  # then run: assistant-voice"
