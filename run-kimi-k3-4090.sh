#!/usr/bin/env bash
# Modified 2026-08: portable RTX 4090 launcher. See MODIFICATIONS.md.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="${K3_ENGINE_DIR:-$SCRIPT_DIR}"
MODEL="${K3_MODEL_DIR:-$HOME/Models/Kimi-K3}"
TRUNK="${K3_TRUNK_DIR:-$HOME/Models/Kimi-K3-trunk}"
ENGINE_BIN="${K3_ENGINE_BIN:-$ENGINE_DIR/bin/k3-cuda}"
TOKENS="${K3_GEN:-64}"
MODE="${K3_MODE:-fast}"
RESULT="${K3_RESULT:-$PWD/k3-last-run.json}"

if (( $# == 0 )); then
    echo "usage: $0 \"your prompt\"" >&2
    echo "modes: K3_MODE=fast (default), balanced, exact" >&2
    exit 2
fi

for required in "$ENGINE_BIN" "$MODEL/config.json" "$TRUNK/trunk.bin"; do
    if [[ ! -e "$required" ]]; then
        echo "missing required path: $required" >&2
        exit 1
    fi
done

case "$MODE" in
    fast)
        TOPK=1
        CUDA_ENV=(
            K3_TRUNK_PERSIST_META=1
            K3_CUDA_MIN_BYTES=0
            K3_CUDA_CACHE_FORMAT=q4
            K3_CUDA_Q3_GROUP=128
            K3_CUDA_CACHE_GB=20
            K3_CUDA_HOST_CACHE_GB=40
        )
        TRUNK_GB=4
        ;;
    balanced)
        TOPK=4
        CUDA_ENV=(
            K3_TRUNK_PERSIST_META=1
            K3_CUDA_MIN_BYTES=0
            K3_CUDA_CACHE_FORMAT=q4
            K3_CUDA_Q3_GROUP=128
            K3_CUDA_CACHE_GB=20
            K3_CUDA_HOST_CACHE_GB=40
        )
        TRUNK_GB=4
        ;;
    exact)
        TOPK=16
        CUDA_ENV=(K3_CUDA_CACHE_GB=18)
        TRUNK_GB=16
        ;;
    *)
        echo "unknown K3_MODE=$MODE (use fast, balanced, or exact)" >&2
        exit 2
        ;;
esac

USER_TEXT=$*
PROMPT='<|open|>message role="user"<|sep|>'"$USER_TEXT"'<|close|>message<|sep|><|end_of_msg|><|open|>message role="assistant"<|sep|><|open|>response<|sep|>'

exec env \
    -u K3_TRUNK_PERSIST_META \
    -u K3_CUDA_MIN_BYTES \
    -u K3_CUDA_CACHE_FORMAT \
    -u K3_CUDA_Q3_GROUP \
    -u K3_CUDA_CACHE_GB \
    -u K3_CUDA_HOST_CACHE_GB \
    "${CUDA_ENV[@]}" "$ENGINE_BIN" "$MODEL" \
    --trunk "$TRUNK" \
    --prompt "$PROMPT" --tok "$MODEL" \
    --gen "$TOKENS" --incremental --cuda \
    --topk "$TOPK" --trunk-gb "$TRUNK_GB" --cache-gb 4 \
    --out "$RESULT"
