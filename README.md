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
| balanced | 93 | 4 | resident mixed Q4/Q3, group 128 | 0.63–0.89 tok/s |
| fast | 93 | 1 | resident mixed Q4/Q3, group 128 | 2.08–2.77 tok/s |
| `--topk 0` | 93 | 0 | resident mixed Q4/Q3, group 128 | 6.07–6.57 tok/s |

Earlier releases packed the trunk at Q4/group-128, which is 28.9 GB: 20 GB fitted in
VRAM and the remaining 9.2 GB was copied host-to-device **every token**, 0.54 s of the
1.2 s a token took. The mixed format packs the large matrices at three bits and the
routing-critical ones at four, 22.6 GB in total, so nothing crosses PCIe per token
(measured 0.408–0.451 → 0.63–0.89 balanced, 0.673–1.033 → 2.08–2.77 fast).

The fast, balanced and `--topk 0` presets are modified models. They retain all 93
language layers, but post-quantize the always-active BF16 trunk and reduce or remove
routed MoE top-k. They are not checkpoint-equivalent Kimi K3, and the mixed trunk is a
larger departure than the Q4 one was: on the reference probe the argmax is unchanged
but the logit vector correlates 0.64 with the Q4 path.

The first forward pass reads and packs the 108.81 GB trunk: measured warm-up is now
~44 s for a 28-token prompt in fast mode (69.11 s before). Later tokens in the same
process perform zero trunk reads. The cache is still lost when the process exits.

## Also here: DeepSeek-V4-Flash-0731 on the same 4090

`deepseek-v4-flash/` is a second, unrelated runner in this repository — same box, same
constraint, different model and a different technique. It runs the released checkpoint
at **full precision** (fp4 e2m1 routed experts, fp8 e4m3 trunk, exactly as shipped, with
output token ids identical to the reference implementation) by streaming the ~140 GiB
expert pool from NVMe through a two-level VRAM + pinned-RAM LRU cache.

| | |
|---|---:|
| steady decode, batch 1 | **~4.5 tok/s** (0.35 tok/s before optimisation) |
| precision | released fp4/fp8 — no requantisation |
| resident in VRAM | 7.8 GiB trunk + KV; experts stream |

It works because routing is skewed: over a real trace only 28.4% of the 11 008
(layer, expert) pairs are ever touched and the hottest 5% serve 32% of accesses, so most
of the 3.21 GiB a token needs is already cached. The rest is a fused grouped fp4 GEMM
that unpacks e2m1 in registers, and triton replacements for the Sinkhorn router,
activation quantisation and fp8 GEMM (26 460 → 6 357 kernel launches per token).

There is also a persistent OpenAI **and** Anthropic compatible server, so Claude Code
and Codex CLI can be pointed at it. See
[deepseek-v4-flash/README.md](deepseek-v4-flash/README.md).

## What changed

- New CUDA Driver API/NVRTC backend compiled for Ada `compute_89`.
- BF16 GEMV for higher-precision model matrices.
- Native Kimi MXFP4/E8M0 expert GEMV directly from packed nibbles.
- Runtime symmetric block-Q3/Q4 trunk packing and GEMV, with bf16 per-group scales and
  an error-minimising scale search.
- Mixed-precision resident cache (`K3_CUDA_CACHE_FORMAT=mix`): the whole trunk in VRAM
  in one arena, no per-token host-to-device tier.
- Page-locked expert arena, router GEMV on the GPU, `--topk 0`, and per-step CPU/GPU
  profiling under `K3_CUDA_PROFILE=1`.
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
