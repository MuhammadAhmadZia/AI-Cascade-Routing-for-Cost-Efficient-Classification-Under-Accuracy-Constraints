#!/usr/bin/env bash
# Large tier (4-bit). Run as a separate process so VRAM is clean.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.run_inference --models qwen3-14b
