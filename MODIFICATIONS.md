# Modification notice

This distribution modifies `kimi-k3-in-c` from upstream commit
`85ab2cd901aa81b70caac7711f06864d594b8ff3`.

Changes made in August 2026:

- optional CUDA Driver API/NVRTC backend for BF16, native checkpoint MXFP4/E8M0,
  and experimental block-scaled Q3/Q4 GEMV;
- two-tier VRAM/CUDA-pinned-host compressed matrix cache;
- mixed Q4/Q3 resident cache (`K3_CUDA_CACHE_FORMAT=mix`) that packs the whole trunk
  into VRAM, removing the per-token host-to-device copy of the overflow tier;
- bf16 per-group scales, error-minimising scale search, single-arena suballocation,
  and chunked quantisation staging, which is what makes the trunk fit;
- router GEMV on the GPU and a page-locked expert arena;
- `--topk 0` (shared experts only) and per-step CPU/GPU profiling under
  `K3_CUDA_PROFILE=1`;
- source-layer-scoped cache keys;
- persistent streamed-trunk metadata for zero trunk reads after warm-up;
- `--cuda`, `--topk`, and experimental `--layer-spacing` controls;
- portable exact, balanced, and fast RTX 4090 launcher;
- CUDA build/link support using Python CUDA runtime and NVRTC wheels;
- inference benchmarks, disclosures, and reproduction documentation.

Balanced and fast alter the model by post-quantizing the BF16 trunk (now mixed
Q4/Q3 at group 128) and reducing routed expert top-k; `--topk 0` removes the routed
experts altogether. Exact does none of it. Official checkpoint files are not modified
or distributed.

This notice is provided for Apache License 2.0 section 4(b).
