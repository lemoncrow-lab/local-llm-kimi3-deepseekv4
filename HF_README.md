---
license: apache-2.0
base_model: moonshotai/Kimi-K3
pipeline_tag: text-generation
library_name: custom
tags:
- kimi-k3
- mixture-of-experts
- cuda
- rtx-4090
- mxfp4
- q4
- nvme
- c
---

# Kimi K3 on one RTX 4090 — experimental CUDA inference

This repository is a modified fork of
[ FareedKhan-dev/kimi-k3-in-c ](https://github.com/FareedKhan-dev/kimi-k3-in-c).
It adds an optional CUDA/NVRTC backend and experimental runtime changes intended to make
the Kimi K3 text model usable on a single 24 GB RTX 4090.

No model weights are included. Download the official
[ moonshotai/Kimi-K3 ](https://huggingface.co/moonshotai/Kimi-K3) checkpoint separately
and comply with its license.

## Measured result

Batch 1, greedy incremental decode on one RTX 4090:

| Mode | Layers | Routed experts/token | Trunk | Steady decode |
|---|---:|---:|---|---:|
| exact | 93 | 16 | checkpoint BF16 | 0.033 tok/s |
| balanced | 93 | 4 | runtime Q4/group-128 | 0.408–0.451 tok/s |
| fast | 93 | 1 | runtime Q4/group-128 | 0.673–1.033 tok/s |

The fast and balanced presets are modified models. They retain all 93 language layers,
but post-quantize the always-active BF16 trunk and reduce routed MoE top-k. They are not
checkpoint-equivalent Kimi K3.

The first forward pass reads and compresses the 108.81 GB packed trunk. With a one-token
input, measured warm-up was 27.57 seconds in fast mode and 32.79 seconds in balanced mode.
A longer XTML prompt took 69.11 and 79.83 seconds respectively. Later tokens in the same
process perform zero trunk reads. The cache is currently lost when the process exits.

## What changed

- New CUDA Driver API/NVRTC backend compiled for Ada `compute_89`.
- BF16 GEMV for higher-precision model matrices.
- Native Kimi MXFP4/E8M0 expert GEMV directly from packed nibbles.
- Runtime symmetric block-Q4 trunk packing and GEMV.
- Two-tier compressed cache: 20 GB in VRAM plus about 9.2 GB in pinned host memory.
- Source-layer-scoped cache keys so streamed layer buffers cannot alias.
- Persistent layer metadata so the trunk is read once per process instead of once/token.
- Configurable routed expert `--topk`.
- Experimental depth-spaced layer mapping; rejected for release presets because it
  produced incoherent output.
- Correct K3 XTML non-thinking chat prompt wrapper.
- Portable exact, balanced, and fast launcher.

Read [the inference report](docs/INFERENCE.md) for implementation details, numeric gates,
negative results, benchmark methodology, and limitations.

## Requirements

Tested target:

- Linux x86-64.
- NVIDIA RTX 4090 with 24 GB VRAM.
- An NVIDIA driver supporting the installed CUDA runtime.
- Sufficient system RAM for the engine and pinned overflow.
- About 1.7 TB free storage for the official checkpoint and packed trunk.
- Fast NVMe storage for warm-up.
- C/C++ compilers, OpenMP, Make, Python 3, and a virtual environment.

PyTorch, vLLM, SGLang, and a full system CUDA toolkit are not required.

## Build

```bash
git clone git@github.com:lemoncrow-lab/local-llm-kimi3.git
cd local-llm-kimi3

python3 -m venv .venv
.venv/bin/pip install -r requirements-cuda.txt
make cuda -j"$(nproc)"
make test -j"$(nproc)"
```

The produced GPU binary is `bin/k3-cuda`.

## Download Kimi K3 separately

```bash
export K3_MODEL_DIR="$HOME/Models/Kimi-K3"

HF_XET_HIGH_PERFORMANCE=1 \
  .venv/bin/hf download moonshotai/Kimi-K3 \
  --local-dir "$K3_MODEL_DIR" \
  --max-workers 16
```

The download is resumable. Reduce the worker count if the connection is unstable.

Expected official checkpoint payload:

- 96 safetensors shards.
- 1,560,936,091,448 bytes.

## Pack the always-active trunk

```bash
export K3_TRUNK_DIR="$HOME/Models/Kimi-K3-trunk"

.venv/bin/python tools/pack_trunk.py \
  "$K3_MODEL_DIR" "$K3_TRUNK_DIR" 93
```

Expected `trunk.bin` size: `108811952128` bytes.

## Run

```bash
K3_MODE=fast K3_GEN=64 ./run-kimi-k3-4090.sh \
  "Explain why the sky is blue."

K3_MODE=balanced K3_GEN=64 ./run-kimi-k3-4090.sh \
  "Write a small C hash table."

# This retains checkpoint BF16 trunk and all 16 routed experts, but is very slow.
K3_MODE=exact K3_GEN=4 ./run-kimi-k3-4090.sh \
  "Say hello."
```

Override locations if needed:

```bash
K3_MODEL_DIR=/mnt/models/Kimi-K3 \
K3_TRUNK_DIR=/mnt/models/Kimi-K3-trunk \
K3_MODE=balanced K3_GEN=32 \
./run-kimi-k3-4090.sh "Your prompt"
```

The result defaults to `./k3-last-run.json`; override it with `K3_RESULT`.

## Validation

- Weightless operation, streaming-cache, safetensors, config, tokenizer, scale, and
  full-model oracle suites pass.
- Full-model oracle: teacher forcing 32/32, greedy 20/20, incremental 20/20.
- Real one-layer BF16 CPU/CUDA logits: max absolute error `2.384186e-06`,
  Pearson correlation `1.0`, same argmax.
- Real one-layer Q4/group-128 versus BF16: Pearson correlation
  `0.9892295088799061`, same argmax.
- CPU exact and CUDA exact selected the same first token in the measured run.

The Q4 check is a numeric kernel gate, not a complete model-quality evaluation. No
perplexity, MMLU, coding, long-context, or safety evaluation has yet been completed for
the top-1/top-4 presets.

Machine-readable results are in [docs/BENCHMARKS.json](docs/BENCHMARKS.json).

## Licenses

Engine source and these modifications are Apache-2.0; see `LICENSE`, `NOTICE`, and
`MODIFICATIONS.md`.

Kimi K3 weights are not distributed here. They are under Moonshot AI's separate
[Kimi K3 License](https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE).
