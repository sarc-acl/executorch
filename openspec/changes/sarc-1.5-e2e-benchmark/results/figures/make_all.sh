#!/usr/bin/env bash
# Regenerate all e2e prefill figures (PDF + 300-dpi PNG) for the six-GPU set from
# ../raw6/cells.csv and ../raw6/runs_all.csv (paths in style.py).
# fig*_5gpu.{pdf,png} are the earlier five-GPU renders, kept for provenance; not regenerated.
set -euo pipefail
cd "$(dirname "$0")"
for f in fig1_speedup fig2_throughput fig3_heatmap; do
  uv run --with matplotlib --with pandas --with numpy python "$f.py"
done
