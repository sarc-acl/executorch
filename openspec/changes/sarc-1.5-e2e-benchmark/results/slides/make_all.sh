#!/usr/bin/env bash
# Regenerate the talk slide figures (SVG + PDF + 200-dpi PNG, 13.333 x 7.5 in).
# Default: 6 GPUs from ../raw6/cells.csv (RX 7900 XTX marked pre-release).
# The *_5gpu.* files are the archived renders of the earlier 5-GPU version; to
# re-render that set from ../cells.csv run:  SLIDES_VARIANT=5gpu ./make_all.sh
# (writes *_5gpu_regen.*, never overwriting the archived files).
set -euo pipefail
cd "$(dirname "$0")"
for f in s1_hero s2_ttft s3_scaling s4_convergence s5_heatmap; do
  uv run -q --with matplotlib --with pandas --with numpy python "$f.py"
done
rm -rf __pycache__
