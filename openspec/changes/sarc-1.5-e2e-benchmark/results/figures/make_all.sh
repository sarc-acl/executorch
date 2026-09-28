#!/usr/bin/env bash
# Regenerate the e2e prefill figures (PDF + 300-dpi PNG). Paths are in style.py.
#   fig1_speedup, fig3_heatmap: speedup-only; 7 GPUs from ../raw7/cells.csv plus the
#     Xclipse (M51) speedup rows (internal device: relative speedups only).
#   fig2_throughput: absolute tok/s; the 7 GPUs from ../raw7/ only (never the M51).
# fig*_5gpu.*, fig*_6gpu.* and fig*_6gpu_m51.* are earlier renders, kept for provenance; not regenerated.
set -euo pipefail
cd "$(dirname "$0")"
for f in fig1_speedup fig2_throughput fig3_heatmap; do
  uv run --with matplotlib --with pandas --with numpy python "$f.py"
done
