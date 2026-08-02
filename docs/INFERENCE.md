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
