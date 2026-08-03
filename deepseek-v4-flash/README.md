# DeepSeek-V4-Flash-0731 on one RTX 4090 — what it costs, what it would run at

Measured 2026-08-02 on this box (i9-14900K, 125 GiB RAM, 32 GiB swap, RTX 4090 24 GB,
FireCuda 530 NVMe). All numbers below are measured, not estimated, except where marked.

## What this is

A runner for the released `DeepSeek-V4-Flash-0731` checkpoint on a single 24 GB GPU, at
the precision it shipped in — fp4 e2m1 routed experts, fp8 e4m3 trunk, no requantisation
— reaching **~4.5 tok/s** by streaming the expert pool from NVMe instead of shrinking it.

It exists because nothing else loads this checkpoint: `config.json` declares
`DeepseekV4ForCausalLM` with no `auto_map`, so transformers cannot build it, and no
vLLM / SGLang / llama.cpp build has that architecture either. Only the engine is
hand-written; FastAPI, uvicorn, triton and the checkpoint's own `encoding_dsv4` do
everything above it.

```
dsv4/run.py          CLI: loads the trunk, installs the streaming MoE, generates
dsv4/serve.py        persistent OpenAI + Anthropic compatible server
dsv4/stream.py       two-level VRAM + pinned-RAM expert cache, pread/DMA loader
dsv4/fp4_triton.py   fused grouped fp4 e2m1 GEMM (unpacks in registers)
dsv4/tk.py           triton Sinkhorn router, act_quant, fp8 GEMM
dsv4/kernels_torch.py  reference torch implementations the triton ones are checked against
dsv4/test_*.py       correctness tests, all asserting on relative error
run-dsv4-4090.sh     one-shot CLI wrapper (forwards to the server if one is up)
serve-dsv4-4090.sh   server wrapper
```

## Requirements

* NVIDIA GPU with **>= 20 GB** VRAM (7.8 GiB trunk + KV + expert cache; measured on a 4090)
* **>= 64 GB** RAM — 48 GiB of it is pinned for the host expert cache by default
* the checkpoint on a fast NVMe (155 GiB; a SATA SSD roughly halves the rate)
* python 3.12+, torch 2.12 with CUDA, triton, transformers, fastapi, uvicorn

## Quick start

```bash
# 1. weights (155 GiB) -- accept the licence on the model page first
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-0731 --local-dir ./DeepSeek-V4-Flash-0731

# 2. environment
python -m venv .venv-dsv4 && .venv-dsv4/bin/pip install \
    torch --index-url https://download.pytorch.org/whl/cu129
.venv-dsv4/bin/pip install triton transformers fastapi uvicorn

# 3. tell the scripts where both live (defaults assume they sit next to them)
export DSV4_MODEL=$PWD/DeepSeek-V4-Flash-0731
export DSV4_PYTHON=$PWD/.venv-dsv4/bin/python

# 4. one-shot
./run-dsv4-4090.sh --chat 'write fizzbuzz in rust'

# 5. or leave it resident -- worth far more here than for a normal model
./serve-dsv4-4090.sh                       # 127.0.0.1:8000
curl -N localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"write fizzbuzz"}],"stream":true}'
```

With the server up, `run-dsv4-4090.sh` forwards to it automatically (`DSV4_NO_SERVER=1`
opts out) — the GPU fits exactly one copy of the trunk, so the two cannot coexist.

Verify the kernels against the torch reference at any time:

```bash
cd dsv4 && ../.venv-dsv4/bin/python test_kernels_torch.py && \
  ../.venv-dsv4/bin/python test_tk.py && ../.venv-dsv4/bin/python test_fp4_triton.py
```

## The checkpoint

155.42 GiB, 48 shards, 43 layers + 3 MTP (DSpark speculative) blocks.

| part | size | dtype | resident? |
| --- | ---: | --- | --- |
| routed experts (46 blocks x 256) | 146.62 GiB | fp4 packed in I8 + E8M0 scales | streamed |
| attention (MLA + indexer + compressor) | 5.40 GiB | fp8 e4m3 + bf16 | GPU |
| shared expert (1 per layer) | 1.08 GiB | fp8 e4m3 | GPU |
| head | 0.99 GiB | bf16 | GPU |
| embed | 0.99 GiB | bf16 | host (row lookup) |
| router / hash-cluster / MTP | 0.35 GiB | bf16 + I64 | GPU |

**Always-active trunk = 7.83 GiB → it fits in 24 GB VRAM with ~15 GB to spare.**
That is the whole difference from Kimi K3, whose trunk is 108.81 GiB.

Per expert: one contiguous **12.00 MiB** fp4 weight run + one contiguous **0.75 MiB**
scale run — two large sequential reads, the best possible layout for streaming.

Per decode token: 43 layers x 6 experts x 12.75 MiB = **3.21 GiB (3.45 GB)**.
K3 at top-16 is 25.83 GB/token — 7.5x more.

## Measured rates (bench-dsv4-io.py, 258 real expert chunks per token)

| path | rate | s/token | tok/s |
| --- | ---: | ---: | ---: |
| cold NVMe, O_DIRECT QD16 | 6.27 GB/s | 0.550 | **1.82** |
| warm page cache QD16 | 21.01 GB/s | 0.164 | **6.10** |
| pinned H2D over PCIe 4 x16 | 17.40 GB/s | 0.198 | **5.05** |

Read the table as a ladder, because the 146.62 GiB expert pool does not fit in
125 GiB of RAM:

* **hard ceiling 5.05 tok/s** — PCIe, unavoidable if experts are multiplied on the GPU
  and every byte is already in RAM.
* **realistic 2.5–4 tok/s** (estimated blend) — with ~60–70 % of the pool in page
  cache and the rest off NVMe, reads overlapping the upload.
* **floor 1.8 tok/s** — cold cache, every expert byte off disk.
* ~15 GB of leftover VRAM holds ~1 100 experts (9 % of the pool), which skip both
  the read and the PCIe hop; hot-expert skew makes that worth more than 9 %.

For scale: Kimi K3 in exact mode on this same box measured **0.055 tok/s**.
DeepSeek-V4-Flash is a ~50–90x better fit for this hardware, at full released
precision (fp4 experts + fp8 trunk is what DeepSeek shipped — nothing is being
downgraded).

## It runs

```bash
cd /home/pankaj/Models
DSV4_TOKENS=200 DSV4_TEMPERATURE=0.0001 ./run-dsv4-4090.sh --chat 'Explain in one paragraph why the sky is blue.'
```

```
trunk: 1615 tensors in 0.8s, 11.92 GiB on gpu
cache: 803 experts in vram (10.0 GiB), 3855 pinned in ram (48.0 GiB)
step 77:  0.23s  4.361 tok/s  3.21 GiB experts  cache 186v/ 62r/ 10d  [fetch 0.04s  expert-gemm 0.02s  rest 0.16s]
text: "The sky appears blue because of a phenomenon called Rayleigh scattering, where
sunlight--which contains all colors--interacts with the tiny molecules of nitrogen and
oxygen in Earth's atmosphere. ..."
```

Full released precision: fp4 e2m1 routed experts, fp8 e4m3 trunk, nothing requantised,
nothing dropped.

| | |
|---|---|
| cold, first token | 0.09 tok/s (prefill pulls 17 GiB of experts) |
| tokens 1–30 | 1.3 → 3.0 tok/s, cache filling |
| steady state | **3.5–4.7 tok/s** (0.21–0.28 s/token), peak 5.4 |

Knobs (all optional, see the header of `run-dsv4-4090.sh`):

| env | default | effect |
|---|---|---|
| `DSV4_VRAM_CACHE_GIB` | 10 | experts held in VRAM; a hit is free. Raise until OOM. |
| `DSV4_RAM_CACHE_GIB` | 48 | experts held in pinned RAM; a hit is an 18.9 GB/s DMA |
| `DSV4_IO_THREADS` | 32 | pread queue depth for disk misses |
| `DSV4_TOKENS` / `DSV4_SEQ_LEN` / `DSV4_TEMPERATURE` | 8 / 1024 / 1.0 | |
| `DSV4_TRITON=0` | — | pure-torch fallback: same numbers, ~2.5x slower |

## Why routing skew is the whole game

Traced over 47 generated tokens (12 126 expert accesses):

* only **28.4%** of the 11 008 (layer, expert) pairs are ever touched;
* the hottest 5% of those — **1.94 GiB** — serve **32%** of all accesses;
* **37%** of a token's experts were already used by the previous token.

| LRU cache | 2 GiB | 4 | 8 | 16 | 32 | 48 |
|---|---|---|---|---|---|---|
| hit rate | 0% | 36% | 51% | 64% | 74% | 74% |

(74% is the compulsory-miss floor for a 47-token trace; over a long generation the
observed rate is ~92%, of which ~60 points are VRAM hits that cost nothing at all.)

Three layers (`n_hash_layers: 3`) route by token id alone via `gate.tid2eid`, so their
experts are knowable before any compute — 7% of the traffic, not yet exploited.

## What had to be built

`inference/` as shipped cannot run here: it assumes the whole 155 GiB is resident
(4xGB300), `convert.py` needs another 155 GB of scratch, and tilelang's `sparse_attn`
asks for 141 312 B of dynamic shared memory against Ada's 101 376 B cap.

| file | what it is |
|---|---|
| `dsv4/kernels_torch.py` | pure-torch reference for all six tilelang kernels + `hadamard_transform`. Correctness oracle. |
| `dsv4/fp4_triton.py` | fused **grouped** fp4 GEMM. Unpacks e2m1 in registers — never materialises a dequantised weight — and does a layer's 6 experts in one launch, gathering them from the cache pool by slot index. |
| `dsv4/tk.py` | triton `act_quant`, `fp4_act_quant`, `fp8_gemm`, `hc_split_sinkhorn`. |
| `dsv4/stream.py` | two-level (VRAM + pinned RAM) LRU expert cache; `pread` from a 32-thread pool straight into pinned slots. |
| `dsv4/run.py` | weight index over the 48 shards, trunk loader, streaming `MoE.forward`. |
| `dsv4/test_*.py` | assert on **relative error**, not correlation — a constant-factor-off result must fail. |

Two bugs cost the most:

* `convert.py` dequantises `attn.wo_a` to bf16 at conversion time and `Attention.forward`
  then uses the weight raw, so the per-128x128 block scale has to be folded in at load
  or the model emits fluent nonsense. `dsv4/run.py:load_trunk` does that.
* the fp4 rounding boundaries are **inclusive-upper** (`torch.bucketize(right=False)`).
  After dividing by a power-of-two scale, exact midpoint hits are common enough that
  `<` instead of `<=` shifts 4.5% of the weights.

## What made it ~9x faster

Baseline was 1.9–2.9 s/token (0.35–0.52 tok/s).

| change | before | after |
|---|---|---|
| grouped fp4 GEMM in registers, 2 launches/layer instead of 18 | 0.40 s | 0.02 s |
| `pread`+pinned+32 threads instead of pageable `mmap`→GPU | 1.20 s | 0.25 s |
| VRAM + pinned-RAM expert cache | 0.25 s | 0.04–0.10 s |
| triton sinkhorn / act_quant / fp8_gemm (26 460 → 6 357 launches/token) | 0.35 s | 0.14 s |
| **total** | **2.4 s** | **0.23 s** |

## The ceiling on this box, honestly

A token needs 43 x 6 x 12.75 MiB = 3.21 GiB of expert weights. 20 tok/s therefore needs
**64 GB/s** of expert bandwidth. What this machine has:

| path | measured |
|---|---|
| VRAM (a cache hit) | ~900 GB/s — but only 10 GiB is free, 803 of 11 008 experts |
| PCIe 4.0 x16, pinned H2D | **18.9 GB/s** |
| `pread` from the dm-crypt'd NVMe pair, QD32 | **9.2 GB/s** (QD4: 3.1) |

With ~60% of accesses served from VRAM, the other 1.28 GiB still has to cross PCIe:
68 ms/token floor, i.e. **~7 tok/s even if every non-VRAM access hit pinned RAM and the
remaining torch work were free.** Measured 4.7 against that is close.

20+ tok/s single-stream needs the working set in VRAM — roughly 40 GiB of it, so a
48 GB card or two 4090s. The alternative is aggregate throughput: at batch B the layer's
expert set is shared across B tokens, and B≈512 reaches ~20 tok/s aggregate off one
sweep of the pool per step. Neither is a tuning change to this runner.

## What is left on the table here

* `rest` (0.14 s) is now the largest term — ~6 400 kernel launches of torch attention,
  RMSNorm and hyper-connections. CUDA graphs over the non-MoE part would take most of it.
* the 3 hash-routed layers can be prefetched at token start (7% of traffic, free overlap).
* `DSV4_RAM_CACHE_GIB` above 48 keeps paying until the pool (137 GiB) or RAM runs out.

## Do not use `DSV4_TILELANG`

It compiles (with `CUDA_HOME=/home/pankaj/Models/.cuda13`) and is faster, but it is
numerically wrong on this hardware: same prompt and seed, the torch/triton path answers
"2, 3, and 5" and the tilelang path answers "dep dep dep". It exists only as a bisect aid.

## Using it for coding

```bash
DSV4_SEQ_LEN=16384 DSV4_RAM_CACHE_GIB=56 DSV4_TOKENS=800 DSV4_TEMPERATURE=0.2 \
  ./run-dsv4-4090.sh --chat "$(cat task.md)"
```

Measured on a 1068-token code prompt (a module + "write a unit test"), 16 k context:

| phase | cost |
|---|---|
| prefill | **30–40 s**, and it is *flat* for any prompt over ~200 tokens |
| generation | **3.5–4.2 tok/s** ≈ 220 tokens/min |
| 800-token answer | ~4 min after prefill |

Three things follow from that.

**Context is free, prefill is not.** MLA + the 128-token sliding window mean seq_len
1024 → 16384 costs 0.1 GiB of VRAM, so use the long context. But a prompt of more than
~200 tokens routes to essentially all 256 experts in every layer, so prefill sweeps the
whole 137 GiB pool once no matter how long the prompt is. Prefer **few long turns over
many short ones** — a 200-token question and a 4000-token question cost the same to
ingest, and every new turn re-pays it.

**Every process start re-pays the warm-up**, so use the server, not the CLI. Starting
`run.py` rebuilds both expert caches from cold: the first token is ~35 s and the rate
only reaches steady state after ~30 tokens. `serve-dsv4-4090.sh` pays that once.

**Do not lower the precision.** The routed experts are *already* fp4 e2m1 — half a byte
per weight, the format the checkpoint was released in — and the trunk is fp8 e4m3. There
is no lower tier to drop to without inventing one, and it would not buy what it looks
like it should:

* per token ~55–70% of expert reads are free VRAM hits and only ~5% reach disk. The
  remaining time is PCIe for the RAM-cache hits (0.07–0.13 s) plus 0.14 s of per-layer
  torch work — neither is bytes-of-precision bound;
* halving the expert size would double the VRAM cache (669 → ~1340 of 11 008 experts),
  which the trace says moves the hit rate ~55% → ~70%. That is **~15–20% faster** in
  exchange for 2-bit weights on a model trained for 4 — a bad trade;
* the only lossless size lever is the block scales, 0.75 MiB of each expert's 12.75 MiB
  (6%), and coarsening them is a precision cut in disguise.

Spend the effort on cache size instead: `DSV4_RAM_CACHE_GIB` as high as free RAM allows
is worth more than any quantisation change, and it costs nothing in quality.

## The server

```bash
./serve-dsv4-4090.sh                     # 127.0.0.1:8000, 16k context, ~90 s to start
```

OpenAI-compatible, so any client works — point it at `http://127.0.0.1:8000/v1`, model
`deepseek-v4-flash`. `POST /v1/chat/completions` (streaming and not, `tools` passed
through to the checkpoint's own DSML tool-call format), `POST /v1/completions`,
`GET /v1/models`, `GET /health`.

### Pointing coding agents at it

The server speaks both wire formats, so the OpenAI-native and Anthropic-native CLIs
both work unchanged: `POST /v1/messages` (+ `/v1/messages/count_tokens`) shares the
engine and the chat template with `/v1/chat/completions`, differing only in the
content-block schema and the named SSE events.

```bash
# Claude Code
ANTHROPIC_BASE_URL=http://127.0.0.1:8000 ANTHROPIC_AUTH_TOKEN=local \
ANTHROPIC_MODEL=deepseek-v4-flash claude

# Codex CLI -- ~/.codex/config.toml
#   model = "deepseek-v4-flash"
#   model_provider = "dsv4"
#   [model_providers.dsv4]
#   name = "dsv4"
#   base_url = "http://127.0.0.1:8000/v1"
#   wire_api = "chat"                       # not the Responses API
```

Temper expectations: at ~4.5 tok/s a 600-token patch is ~2 minutes, and every tool
result re-prefills unless it extends the previous state exactly.

Only the *engine* is hand-written, and only because it has to be: the checkpoint declares
`DeepseekV4ForCausalLM` with no `auto_map`, so transformers cannot load it, and no
vLLM / SGLang / llama.cpp build has that architecture (hyper-connections, Sinkhorn
combination weights, sqrtsoftplus routing with hash layers, per-layer KV compression,
fp4 e2m1 experts, DSpark MTP). Everything above it is off the shelf — FastAPI + uvicorn
for routing, SSE and validation, and `encoding_dsv4` for the chat template.

What being resident buys:

| | CLI | server |
|---|---|---|
| expert caches | rebuilt every run | resident |
| first token | ~35 s, every invocation | ~35 s, once |
| conversation turn 2+ | full re-prefill | **prefix reuse**: only the new tokens |

Prefix reuse is limited by the reference model, not by choice: `Attention.forward` has
exactly two modes, a chunked prefill at `start_pos == 0` and single-token decode
(model.py:534 squeezes `seqlen`), and the KV compressor writes at `start_pos % ratio`, so
state can be extended but never rewound. A continuation is therefore walked one token at
a time — 0.25 s each against a flat ~35 s prefill, so it pays up to `DSV4_CONTINUE_MAX`
(256) new tokens, beyond which the server resets and prefills. That covers the agent loop
exactly: resend the conversation with the assistant's verbatim reply plus a new turn and
only the new turn is processed.

One request at a time by construction (`max_batch_size` 1, and the expert cache is shared
mutable state); concurrent callers queue on a lock.

**`run-dsv4-4090.sh` uses the server when one is up.** The GPU fits exactly one copy of
the trunk, so the CLI probes `DSV4_PORT` (8000) first and, if it finds this server,
forwards the prompt and streams the reply instead of loading a second 12 GiB model --
which also skips the cold start. `DSV4_NO_SERVER=1` forces a local load. If neither can
get the GPU, both now exit naming the process that holds it rather than raising a torch
OOM from inside `Transformer.__init__`.
