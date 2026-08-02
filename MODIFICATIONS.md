# Modification notice

This distribution modifies `kimi-k3-in-c` from upstream commit
`85ab2cd901aa81b70caac7711f06864d594b8ff3`.

Changes made in August 2026:

- optional CUDA Driver API/NVRTC backend for BF16, native checkpoint MXFP4/E8M0,
  and experimental block-scaled Q3/Q4 GEMV;
- two-tier VRAM/CUDA-pinned-host compressed matrix cache;
- source-layer-scoped cache keys;
- persistent streamed-trunk metadata for zero trunk reads after warm-up;
- `--cuda`, `--topk`, and experimental `--layer-spacing` controls;
- portable exact, balanced, and fast RTX 4090 launcher;
- CUDA build/link support using Python CUDA runtime and NVRTC wheels;
- inference benchmarks, disclosures, and reproduction documentation.

Balanced and fast alter the model by post-quantizing the BF16 trunk to block
Q4/group-128 and reducing routed expert top-k. Exact does neither. Official checkpoint
files are not modified or distributed.

This notice is provided for Apache License 2.0 section 4(b).
