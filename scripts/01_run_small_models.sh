#!/usr/bin/env bash
# Small tiers, both sizes, all four datasets. Needs a CUDA GPU.
# Resumable: re-run after any interruption and it continues from the checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.run_inference --models qwen3-1.7b qwen3-4b
