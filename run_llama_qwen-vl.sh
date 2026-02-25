#!/usr/bin/env bash
set -euo pipefail

HOST="0.0.0.0"
PORT="8081"

# Quant paling balance untuk 6GB
HF_REPO="unsloth/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M"

# Context aman untuk VLM di 6GB
CTX_SIZE="8192"

# Fokus single request (lebih stabil daripada auto=4 slot)
N_PARALLEL="1"

# Batch tuning (aman, bisa kamu naikkan nanti)
BATCH="2048"
UBATCH="128"

# Sampling (meniru preferensi kamu)
TOP_P="0.8"
TOP_K="20"
TEMP="0.7"
MIN_P="0.0"
PRES_PEN="1.5"

# KV cache quant biar ctx muat
CACHE_K="q8_0"
CACHE_V="q8_0"

# PENTING untuk Qwen-VL: minimum image tokens
IMAGE_MIN_TOKENS="1024"

# Batasi image max tokens biar tidak "meledak" untuk slide resolusi tinggi
# Jika kamu merasa detail masih kurang, naikkan ke 2048
IMAGE_MAX_TOKENS="2048"

# Prompt cache RAM (opsional)
# 0 = disable, atau kecilkan mis. 2048 untuk hemat RAM
CACHE_RAM="0"

echo "========================================"
echo " llama-server Qwen3-VL-2B"
echo " Repo   : ${HF_REPO}"
echo " Listen : http://${HOST}:${PORT}"
echo " Ctx    : ${CTX_SIZE}"
echo " Slots  : ${N_PARALLEL}"
echo " ImgTok : min=${IMAGE_MIN_TOKENS}, max=${IMAGE_MAX_TOKENS}"
echo " KV     : K=${CACHE_K}, V=${CACHE_V}"
echo "========================================"

exec llama-server \
  --host "${HOST}" \
  --port "${PORT}" \
  --hf-repo "${HF_REPO}" \
  --mmproj-auto \
  --mmproj-offload \
  --n-gpu-layers all \
  --jinja \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --temp "${TEMP}" \
  --min-p "${MIN_P}" \
  --presence-penalty "${PRES_PEN}" \
  --fit on \
  --flash-attn on \
  --ctx-size "${CTX_SIZE}" \
  --parallel "${N_PARALLEL}" \
  --batch-size "${BATCH}" \
  --ubatch-size "${UBATCH}" \
  --cache-type-k "${CACHE_K}" \
  --cache-type-v "${CACHE_V}" \
  --image-min-tokens "${IMAGE_MIN_TOKENS}" \
  --image-max-tokens "${IMAGE_MAX_TOKENS}" \
  --cache-ram "${CACHE_RAM}" \
  --no-perf \
  --cont-batching