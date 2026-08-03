#!/usr/bin/env bash
# OpenAI-compatible server for DeepSeek-V4-Flash-0731 on one RTX 4090.
#
#   ./serve-dsv4-4090.sh                 # 127.0.0.1:8000, 16k context
#   DSV4_PORT=9000 ./serve-dsv4-4090.sh
#
# Point any OpenAI client at http://127.0.0.1:8000/v1 with model "deepseek-v4-flash".
# Startup takes ~30 s (trunk load + pinning the RAM cache); after that the caches and
# the KV state stay resident, so there is no per-invocation warm-up and a conversation
# that extends the previous one skips prefill entirely.
#
#   DSV4_VRAM_CACHE_GIB   default 0 = auto, all free VRAM after the trunk
#   DSV4_RAM_CACHE_GIB    default 48; shrinks automatically if it cannot pin that much
#   DSV4_SEQ_LEN          default 16384 (context is nearly free: MLA + 128 window)
#   DSV4_IO_THREADS       default 32
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export DSV4_MODEL=${DSV4_MODEL:-$HERE/DeepSeek-V4-Flash-0731}
export DSV4_VRAM_CACHE_GIB=${DSV4_VRAM_CACHE_GIB:-0}
export DSV4_RAM_CACHE_GIB=${DSV4_RAM_CACHE_GIB:-48}
export DSV4_IO_THREADS=${DSV4_IO_THREADS:-32}
export DSV4_SEQ_LEN=${DSV4_SEQ_LEN:-16384}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
PY=${DSV4_PYTHON:-$HERE/.venv-dsv4/bin/python}

[ -x "$PY" ] || { echo "no venv at $PY (needs torch 2.12 + triton + fastapi + uvicorn)"; exit 1; }
[ -d "$DSV4_MODEL" ] || { echo "no checkpoint at $DSV4_MODEL"; exit 1; }

exec "$PY" "$HERE/dsv4/serve.py" "$@"
