# r/LocalLLaMA post draft

## Title

[Experimental] Kimi K3 on one RTX 4090: 0.42–0.95 tok/s after warm-up (93 layers, Q4 trunk, top-4/top-1 MoE)

## Body

I added a CUDA/NVRTC backend and compressed warm-cache path to the C99 Kimi K3 engine.

First, the honest baseline: offloading the unchanged graph alone is still slow.

| mode | layers | routed top-k | trunk | measured decode |
|---|---:|---:|---|---:|
| CUDA exact | 93 | 16 | checkpoint BF16 | 0.033 tok/s |
| CUDA balanced | 93 | 4 | runtime Q4/128 | ~0.422 tok/s average |
| CUDA fast | 93 | 1 | runtime Q4/128 | ~0.949 tok/s average |

Balanced and fast are not exact K3. They keep all 93 language layers, but post-quantize
the always-active BF16 trunk to symmetric block Q4/group-128 and reduce the 16 routed
experts to top-4/top-1. The shared-expert path remains.

Low-level changes:

- custom BF16 GEMV on CUDA;
- native Kimi MXFP4/E8M0 expert GEMV directly from packed nibbles;
- runtime Q4 trunk pack/GEMV;
- 20 GB VRAM cache plus ~9.2 GB CUDA-pinned host overflow;
- cache keys scoped by source layer;
- ~2.47 GB persistent CPU metadata, so the 108.81 GB trunk is read once/process;
- configurable routed top-k;
- official XTML non-thinking prompt wrapper.

One-token-input warm-up was 27.57 s fast and 32.79 s balanced. A 32-token XTML prompt took
69.11/79.83 s. After warm-up, balanced measured 0.408–0.451 tok/s and fast measured
0.673–1.033 tok/s. The cache is process-local, so restarting repeats warm-up.

Quality caveat: there is no perplexity/MMLU/coding/long-context evaluation yet. One real
Q4/128 layer had 0.98923 logit correlation versus BF16 with the same argmax. Both practical
modes produced coherent text for “Explain why the sky is blue.” Eight- and 24-layer
pruning were faster but produced garbage, so I rejected them.

The exact path has stronger validation: CPU and CUDA selected the same first token; a real
BF16 layer measured max logit error 2.38e-6, correlation 1.0, and the same argmax. The
weightless operation/full-model oracle suite passes.

No weights are in the repository. Download the official 96-shard, ~1.56 TB Moonshot Kimi
K3 checkpoint separately and pack the 108.81 GB trunk.

Source: https://github.com/lemoncrow-lab/local-llm-kimi3

```bash
K3_MODE=balanced K3_GEN=64 ./run-kimi-k3-4090.sh \
  "Explain why the sky is blue."
```

The current bottleneck is the ~9.2 GB cache overflow crossing PCIe each token. The next
goal is to fit the active cache in VRAM or overlap that traffic while retaining coherent
decode.
