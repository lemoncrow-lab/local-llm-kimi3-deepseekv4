# Inference implementation and benchmarks

## Scope

This fork implements the Kimi K3 language/text graph. It does not implement the vision
tower. The official checkpoint files remain unchanged.

Kimi K3's text configuration has 93 layers, hidden width 7,168, 896 routed experts,
16 routed experts per token in the original graph, two shared experts, and a vocabulary
of 163,840. Expert weights are already packed MXFP4/E2M1 with E8M0 scales. The
always-active trunk is BF16. The official checkpoint payload is about 1.56 TB; the
engine's packed 93-layer trunk is 108.81 GB.

## Exact path versus practical presets

The exact CUDA path retains:

- all 93 layers;
- checkpoint BF16 trunk weights;
- all 16 routed experts;
- original routing and shared-expert computation.

It measured 0.033 tok/s. The practical presets trade fidelity for speed:

| Preset | Changed trunk | Routed top-k |
|---|---|---:|
| balanced | runtime symmetric Q4, group 128 | 4 |
| fast | runtime symmetric Q4, group 128 | 1 |

Both still execute all 93 layers. Neither practical preset should be described as exact
Kimi K3.

## CUDA backend

`src/cuda/k3_cuda.cu` is host C++ linked against the CUDA Driver API and NVRTC.
Embedded kernels are compiled for `compute_89` at process startup.

Implemented kernels:

1. BF16 GEMV.
2. Native checkpoint MXFP4/E8M0 GEMV.
3. Block scale generation.
4. Signed Q4 pack and Q4 GEMV.
5. Experimental Q3 pack and Q3 GEMV.

Each matrix-vector operation computes output rows in CUDA blocks and reduces partial sums
with warp shuffles. The MXFP4 kernel reads packed nibbles and checkpoint scale bytes
directly, avoiding a full dequantized expert buffer.

## Compressed trunk cache

A first Q4 implementation remained slow because it reread the 108.81 GB trunk metadata
for every generated token. Over 16 steps it read roughly 1.74 TB and achieved only about
0.052 tok/s.

The current cache has two parts:

- compressed matrices: 20 GB in VRAM and roughly 9.2 GB in CUDA-pinned host overflow;
- persistent CPU metadata: roughly 2.47 GB for norms, gates, biases, scales, and other
  layer vectors still consumed on the CPU.

The first forward reads each of the 93 trunk layers once, packs eligible matrices, and
copies persistent metadata. Cache entries include their checkpoint source-layer scope;
this prevents address reuse in the streamed trunk ring from aliasing two layers. Once
the cache is complete, later decode bypasses trunk rebinding and performs zero trunk
reads in the same process.

Host-overflow matrices are copied into reusable device staging storage on demand. A future
optimization should eliminate or overlap this PCIe traffic.

## Routed experts

`--topk N` accepts values from 1 through the checkpoint top-k of 16. Reducing it changes
the model. The shared-expert path is not removed by this flag.

The CLI prints a modification warning whenever top-k is below the checkpoint value.

## Prompt rendering

The portable launcher wraps user text with K3's non-thinking XTML prompt:

```text
<|open|>message role="user"<|sep|>USER<|close|>message<|sep|><|end_of_msg|><|open|>message role="assistant"<|sep|><|open|>response<|sep|>
```

Without this renderer, raw prompts frequently continued control/JSON scaffolding.

## Measured throughput

All measurements are greedy, batch-1, incremental decode. “Steady” excludes the first
cache-building step.

| Run | Layers | Top-k | Warm-up | Steady tok/s |
|---|---:|---:|---:|---:|
| CUDA exact, one-ID prompt | 93 | 16 | not separated | 0.033 |
| balanced, one-ID prompt | 93 | 4 | 32.79 s | 0.416–0.431 |
| fast, one-ID prompt | 93 | 1 | 27.57 s | 0.872–1.033 |
| balanced, 32-token XTML prompt | 93 | 4 | 79.83 s | 0.408–0.451 |
| fast, 32-token XTML prompt | 93 | 1 | 69.11 s | 0.673–0.937 |

Five one-ID steady steps averaged 0.4216 tok/s for balanced and 0.949 tok/s for fast.
Practical-mode peak RSS was about 23.6–23.7 GB.

The cache is process-local. Each launcher invocation rebuilds it. These results therefore
favor longer generations in one process and do not yet represent a low-latency server.

## Numeric gates

Exact real-layer BF16 CPU versus CUDA:

- max absolute logit error: `2.384186e-06`;
- mean absolute logit error: `3.085209e-07`;
- Pearson correlation: `1.0`;
- same argmax.

Runtime Q4/group-128 versus BF16 for one real layer:

- max absolute logit error: `1.357734`;
- mean absolute logit error: `0.20808391`;
- Pearson correlation: `0.9892295088799061`;
- same argmax token `22436`.

Device-resident Q4 and pinned-host-overflow Q4 produced bit-identical logit vectors.
Persistent and non-persistent metadata produced bit-identical Q4 logits.

These are implementation checks, not full-model quality evaluations.

## Rejected experiments

### Q3 as the default

One-layer logit correlations versus BF16 were:

| Q3 format | Pearson correlation |
|---|---:|
| row-scale | 0.738 |
| group 256 | 0.932 |
| group 128 | 0.944 |

Q3/group-128 is implemented, but was not made the published default because error can
accumulate through all 93 layers.

### Depth pruning

Eight-layer and 24-layer depth-spaced variants decoded incoherent text. They were rejected.
The published exact, balanced, and fast presets all retain 93 layers.

## Decoded-text smoke test

For `Explain why the sky is blue.`, balanced began:

> The sky appears blue because Earth's atmosphere scatters shorter blue wavelengths of sunlight more than

Fast began:

> The sky is blue because sunlight is scattered in all directions by the Earth’s atmosphere

This verifies the tokenizer/prompt/decode path. It is not a quality benchmark.

## The resident mixed cache (2026-08-02)

A steady token in fast mode was 1.19-1.33 s. Per-step attribution (`K3_CUDA_PROFILE=1`,
which prints a CUDA row and a host row after every token) put it here:

| item | s/token |
|---|---:|
| host->device copy of the 9.2 GB Q4 overflow tier | 0.54 |
| routed expert uploads, 1.61 GB out of pageable memory | 0.20 |
| router GEMV: 2.4 GB of f32 weights, f64 accumulate, on the CPU | 0.25 |
| expert disk reads | 0.25 |
| resident trunk GEMVs (815 calls) | 0.06 |

The overflow tier existed because Q4/group-128 packs this trunk to 28.9 GB and the card
holds 24.5. `K3_CUDA_CACHE_FORMAT=mix` packs matrices above `K3_CUDA_Q4_MAX_ELEMS`
(default 26M elements: the attention projections and shared experts) at three bits and
leaves the rest at four, for 22.6 GB. Three things had to be fixed before that fitted:

- **bf16 per-group scales.** f32 scales at group 128 are 0.25 bits per weight, 1.7 GB
  across the trunk. A scale needs range, not mantissa.
- **One arena, bump-allocated.** The cache holds ~2,400 tensors and `cuMemAlloc` rounds
  each request to a 2 MB granule; the rounding alone was over a gigabyte, and it showed
  up as an out-of-memory 90 layers into warm-up rather than as a number anywhere.
- **Chunked quantisation staging.** Packing a matrix used to stage the whole BF16
  source on the device, so 2.35 GB had to be reserved all run for `lm_head` alone.
  `K3_CUDA_STAGE_MB` (default 256) caps it.

With the trunk resident, the remaining PCIe traffic is the routed experts themselves.
Page-locking the expert arena (`cuMemHostRegister` over `K3Cache.arena`) took their
upload from 8 GB/s to PCIe speed, and moving the router GEMV onto the GPU removed the
last large CPU matmul.

Measured after, same prompt, 8-token runs:

| preset | before | after |
|---|---:|---:|
| fast (top-1) | 0.673-1.033 tok/s | 2.08-2.77 tok/s |
| balanced (top-4) | 0.408-0.451 tok/s | 0.63-0.89 tok/s |
| `--topk 0` | - | 6.07-6.57 tok/s |

### What it costs

About 88% of trunk parameters moved from four bits to three. On the reference probe the
next-token argmax is unchanged and the decoded text stays coherent, but the logit vector
correlates 0.64 with the Q4 path (0.57 with the GPU router as well). `K3_ROUTER_CPU=1`
restores the exact f64 router; `K3_CUDA_Q4_MAX_ELEMS` raises the Q4 share until the
overflow tier reappears; `K3_CUDA_CACHE_FORMAT=q4` restores the old behaviour entirely.

### What is left

0.45 s/token in fast mode is 0.17 s of expert path, 0.07 s of trunk GEMVs and 0.21 s of
host time. That last figure is dominated by ~1,500 upload-launch-download-synchronise
round trips per token, because the graph runs on the CPU and only the matmuls are
offloaded. Device-resident activations, the KDA recurrence on the GPU (0.045 s/token,
626 MB of state) and an asynchronous expert prefetch are the next three items, in that
order.

## exact mode: multiply in RAM, do not upload

Exact keeps the BF16 trunk, so 86 of its 108.81 GB cannot be resident on a 24 GB card.
Uploading those bytes costs PCIe at a measured 18.3 GB/s; multiplying them where they
already are costs a host-RAM sweep. On an 8-layer slice with a deliberately small 4 GB
VRAM cache and `--topk 16`, the lane measured 0.78-0.85 s/token against 1.60-1.63 with
the upload path, and produced the same three tokens: 138932, 126023, 153224.

Full-model exact speed is then set by how much trunk fits in RAM, because an unpinned
layer costs 1.17 GB of NVMe at 6.2 GB/s every token. With ~60 GB available (this
measurement) only about a third of the trunk pins and a step measures 32 s, unchanged.
With ~115 GB free the whole trunk pins and the arithmetic says 8-13 s/token.

Neither the lane nor anything else lifts exact past **0.71 tok/s** on one card: top-16
routing moves 25.83 GB of expert bytes per token and PCIe delivers 18.3 GB/s.

## Current bottleneck and next target

The practical Q4 cache is larger than available VRAM, leaving about 9.2 GB in pinned host
memory. That overflow is transferred over PCIe every token and is the main candidate for
the next order-of-magnitude optimization.

Potential approaches must be measured against coherent decode and numeric gates:

- fit the complete active cache in VRAM with a mixed precision policy;
- skip or approximate selected always-active matrices only with explicit model-change
  disclosure;
- overlap host transfers with compute using double buffering;
- persist the packed cache across process launches;
- fuse small operations and reduce CUDA launch/synchronization overhead.

## Limitations

- Top-1/top-4 quality has not been evaluated on a standard suite.
- Runtime Q4 is post-training quantization, not QAT.
- No continuous batching, API server, or long-lived multi-prompt loop.
- No persistent cache file.
- Text only.
- No claim that practical presets reproduce official Kimi K3 quality.
