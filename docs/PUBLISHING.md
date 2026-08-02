# Hugging Face publishing

Create a source-only model repository. Do not upload the official checkpoint or
`trunk.bin`.

```bash
cd /path/to/local-llm-kimi3
.venv/bin/hf auth login

export HF_REPO="YOUR_HF_USERNAME/local-llm-kimi3"
.venv/bin/hf repos create "$HF_REPO" --repo-type model

.venv/bin/hf upload "$HF_REPO" HF_README.md README.md \
  --repo-type model \
  --commit-message "Add Kimi K3 RTX 4090 model card"

.venv/bin/hf upload "$HF_REPO" . . \
  --repo-type model \
  --include "Makefile" \
  --include "LICENSE" \
  --include "NOTICE" \
  --include "MODIFICATIONS.md" \
  --include "HF_README.md" \
  --include "requirements-cuda.txt" \
  --include "run-kimi-k3-4090.sh" \
  --include "include/**" \
  --include "src/**" \
  --include "third_party/**" \
  --include "tools/**" \
  --include "tests/**" \
  --include "docs/*.md" \
  --include "docs/BENCHMARKS.json" \
  --exclude "README.md" \
  --exclude ".git/**" \
  --exclude ".venv/**" \
  --exclude "bin/**" \
  --exclude "build/**" \
  --exclude "build-cuda/**" \
  --commit-message "Publish reproducible CUDA inference source"
```

After upload, verify that the repository contains no safetensors, packed trunk, virtual
environment, build artifact, or credential.
