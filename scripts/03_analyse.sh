#!/usr/bin/env bash
# Tables and figures. CPU only, seconds.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.analysis
python -m src.figures
echo "tables in results/, figures in results/figures/"
