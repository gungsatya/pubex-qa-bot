#!/bin/bash
set -euo pipefail

if ! command -v ollama >/dev/null 2>&1; then
  echo "Perintah 'ollama' tidak ditemukan. Install Ollama terlebih dahulu." >&2
  exit 1
fi

VLM_MODEL="${VLM_MODEL:-qwen2.5vl:3b}"
LLM_MODEL="${LLM_MODEL:-qwen2.5:3b}"
EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"

echo "Memastikan model tersedia di Ollama..."
ollama pull "$VLM_MODEL"
ollama pull "$LLM_MODEL"
ollama pull "$EMBED_MODEL"

if [ "${1:-}" = "--pull-only" ]; then
  echo "Semua model sudah tersedia."
  exit 0
fi

echo "Menjalankan Ollama server di default endpoint (http://127.0.0.1:11434)."
exec ollama serve
