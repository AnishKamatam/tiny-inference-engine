#!/usr/bin/env bash
# Downloads the test models into models/ and generates HuggingFace reference fixtures.
# Revisions are pinned so fixtures are reproducible. Requires the `hf` CLI and `uv`.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models

hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir models/Qwen3-0.6B
hf download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-BF16.gguf \
  --revision 50968a4468ef4233ed78cd7c3de230dd1d61a56b --local-dir models
hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf \
  --revision 23749fefcc72300e3a2ad315e1317431b06b590a --local-dir models

uv run tools/dump_reference.py models/Qwen3-0.6B models/fixtures
