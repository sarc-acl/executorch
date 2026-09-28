#!/usr/bin/env bash
# Regenerate all e2e prefill figures (PDF + 300-dpi PNG) from ../cells.csv and ../runs_all.csv.
set -euo pipefail
cd "$(dirname "$0")"
for f in fig1_speedup fig2_throughput fig3_heatmap; do
  uv run --with matplotlib --with pandas --with numpy python "$f.py"
done
