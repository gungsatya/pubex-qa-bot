#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_BIN="$BASE_DIR/llama/build/bin"

export LD_LIBRARY_PATH="$LLAMA_BIN:${LD_LIBRARY_PATH:-}"

HF_MODEL="Qwen/Qwen3-VL-2B-Instruct-GGUF"

HOST="127.0.0.1"
PORT="8080"
GPU_LAYERS="32"
CTX_SIZE="5120"
THREADS="6"
PARALLEL="1"
CACHE_SIZE="2048"

echo "======================================"
echo " Starting LLaMA Server (Multimodal)"
echo " Model   : $HF_MODEL"
echo " Host    : $HOST:$PORT"
echo " GPU Lyr : $GPU_LAYERS"
echo " Ctx     : $CTX_SIZE"
echo " Threads : $THREADS"
echo "======================================"

exec "$LLAMA_BIN/llama-server" \
  --host "$HOST" \
  --port "$PORT" \
  -hf "$HF_MODEL" \
  --mmproj-offload \
  --n-gpu-layers "$GPU_LAYERS" \
  --ctx-size "$CTX_SIZE" \
  --threads "$THREADS" \
  --parallel "$PARALLEL" \
  --no-warmup \
  --cache-ram "$CACHE_SIZE" 