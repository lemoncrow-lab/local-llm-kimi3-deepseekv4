# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `K3_CUDA_CACHE_FORMAT=mix`: mixed Q4/Q3 resident cache that packs the whole 108.81 GB
  trunk into 22.6 GB of VRAM, removing the 9.2 GB per-token host-to-device tier.
  Threshold via `K3_CUDA_Q4_MAX_ELEMS`, staging via `K3_CUDA_STAGE_MB`,
  error-minimising scale search via `K3_CUDA_QOPT`.
- Router GEMV on the GPU (`K3_ROUTER_CPU=1` restores the exact f64 CPU path) and a
  page-locked expert arena.
- `--topk 0`: shared experts only, no routed path. A different model; measured as the
  ceiling of the current graph at 6.07-6.57 tok/s.
- `K3_CUDA_PROFILE=1`: per-step attribution of CPU and GPU time.

### Changed

- Packed per-group scales are bf16 rather than f32, and packed tensors are
  suballocated from one arena instead of two `cuMemAlloc`s each.
- Steady decode, one RTX 4090: fast 0.673-1.033 -> 2.08-2.77 tok/s, balanced
  0.408-0.451 -> 0.63-0.89 tok/s, warm-up 69 s -> 44 s. The mixed trunk moves the logit
  vector (Pearson 0.64 against the Q4 path); the argmax on the reference probe does not
  change.

## [0.1.0] - 2026-07-31

First public release.

### Added

- Full 93-layer Kimi K3 inference: 69 KDA + 24 Gated MLA layers, 896 routed experts with
  top-16 selection, SiTU-GLU, Attention Residuals, native MXFP4 expert weights.
- **Trunk streaming**, which turns the memory budget into a dial rather than a floor. The
  model runs in 8 GB and in 224 GB and produces byte-identical output at every budget
  measured in between.
- MXFP4 matmul that consumes packed nibbles directly, never materialising a dequantised
  expert.
- BPE tokenizer in C, reading the released `tiktoken.model` directly, text in, text out
  with no external step.
- Config reader that loads the checkpoint's own `config.json` and **refuses** a config it
  cannot fully understand rather than defaulting missing fields.
- Incremental decode with a KV cache and carried recurrent state, verified to produce the
  same tokens as full recompute.
- Named memory presets (`--preset laptop|desktop|workstation|server|max`) derived from
  the measured memory ladder.
- `scripts/k3-doctor.sh`, reports whether a machine can run the model, which preset
  fits, and how fast its storage is.
- `scripts/download-model.sh`, fetches the checkpoint and verifies it byte-exactly
  against the published total, because a partial download produces wrong output silently.
- Test suite that runs entirely without model weights: op fixtures, expert cache,
  safetensors reader, config reader, and end-to-end oracle gates (teacher forcing,
  greedy decode, and incremental decode).
- CI: build matrix across GCC and Clang, warnings-as-errors, ASan and UBSan, Python and
  shell lint. Tokenizer parity is built and reported but CANNOT gate on a clean
  checkout, because it needs the vocabulary that ships with the model weights; run
  `make tok` locally against a downloaded checkpoint.

### Known limitations

- No chunked prefill, so long prompts are impractical despite a 32k context ceiling.
- Greedy decoding only; no chat template; no serving layer; no vision; CPU only.

See [docs/ROADMAP.md](docs/ROADMAP.md).

[Unreleased]: https://github.com/FareedKhan-dev/kimi-k3-in-c/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FareedKhan-dev/kimi-k3-in-c/releases/tag/v0.1.0
