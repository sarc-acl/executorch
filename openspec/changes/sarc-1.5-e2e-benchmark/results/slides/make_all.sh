#!/usr/bin/env bash
# Regenerate all talk slide figures (SVG + PDF + 200-dpi PNG, 13.333 x 7.5 in) from ../cells.csv.
set -euo pipefail
cd "$(dirname "$0")"
for f in s1_hero s2_ttft s3_scaling s4_convergence s5_heatmap; do
  uv run -q --with matplotlib --with pandas --with numpy python "$f.py"
done
