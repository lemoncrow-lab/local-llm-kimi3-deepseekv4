#!/usr/bin/env bash
# Run DeepSeek-V4-Flash-0731 on one RTX 4090, streaming the routed experts from disk.
#
#   ./run-dsv4-4090.sh 'why is the sky blue?'                  # <=4096 tokens, temp 1.0
#   DSV4_TOKENS=200 ./run-dsv4-4090.sh --chat 'explain MoE'    # post-trained format
#   DSV4_TEMPERATURE=0.0001 ./run-dsv4-4090.sh --chat 'hi'     # greedy, reproducible
#   ./run-dsv4-4090.sh --ids 2778                              # skip the tokenizer
#
# If serve-dsv4-4090.sh is already running (DSV4_PORT, default 8000) this forwards the
# prompt to it and streams the reply -- the GPU fits exactly one copy of the trunk, and
# the server's caches are already warm.  DSV4_NO_SERVER=1 forces a local load.
#
# The 7.8 GiB trunk sits in VRAM.  Each token routes to 43 x 6 fp4 experts = 3.21 GiB,
# but routing is skewed enough that most of it comes from the two caches below rather
# than off disk, so the rate climbs over the first ~30 tokens and then holds.
#
#   DSV4_VRAM_CACHE_GIB   experts kept in VRAM; a hit is free      (default 10)
#   DSV4_RAM_CACHE_GIB    experts kept in pinned RAM; a hit is a   (default 48,
#                         shrinks automatically if the box cannot pin that much)
#                         DMA at 18.9 GB/s instead of a 9.2 GB/s read
#   DSV4_IO_THREADS       pread queue depth for the disk misses    (default 32)
#   DSV4_TRITON=0         fall back to pure torch: same numbers, ~2.5x slower
#
# Raise DSV4_VRAM_CACHE_GIB until it OOMs (11.9 GiB of the 24 is trunk + KV) and
# DSV4_RAM_CACHE_GIB as far as free RAM allows -- both buy tok/s directly.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export DSV4_MODEL=${DSV4_MODEL:-$HERE/DeepSeek-V4-Flash-0731}
export DSV4_VRAM_CACHE_GIB=${DSV4_VRAM_CACHE_GIB:-0}   # 0 = auto: all free VRAM after the trunk
export DSV4_RAM_CACHE_GIB=${DSV4_RAM_CACHE_GIB:-48}
export DSV4_IO_THREADS=${DSV4_IO_THREADS:-32}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
PY=${DSV4_PYTHON:-$HERE/.venv-dsv4/bin/python}

[ -x "$PY" ] || { echo "no venv at $PY (needs torch 2.12 + triton + transformers)"; exit 1; }
[ -d "$DSV4_MODEL" ] || { echo "no checkpoint at $DSV4_MODEL"; exit 1; }

# tilelang is deliberately NOT used: its sparse_attn wants 141 KB of dynamic shared
# memory and Ada caps at 100 KB.  dsv4/{kernels_torch,tk,fp4_triton}.py stand in.
exec "$PY" "$HERE/dsv4/run.py" \
    --max-new-tokens "${DSV4_TOKENS:-4096}" \
    --seq-len "${DSV4_SEQ_LEN:-16384}" \
    --temperature "${DSV4_TEMPERATURE:-1.0}" \
    "$@"
