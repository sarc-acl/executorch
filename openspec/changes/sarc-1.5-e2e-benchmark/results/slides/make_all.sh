#!/usr/bin/env bash
# Regenerate the talk slide figures (SVG + PDF + 200-dpi PNG, 13.333 x 7.5 in).
# Default ("7gpu"): s1/s1b/s2/s4 (absolute tok/s, seconds) show 7 GPUs from
# ../raw7/cells.csv ("the" x 2048; RX 7900 XTX and RX 7600 pre-release, †);
# the speedup-only slides s3/s5 also show the internal Xclipse (M51, ‡) from its
# relative-speedup contribution (contrib/m51/speedups.csv; its 4w row not plotted).
# Archived earlier renders (never overwritten):
#   *_5gpu.*      five GPUs
#   *_6gpu.*      six GPUs (+RX 7900 XTX), no M51
#   *_6gpu_m51.*  s3/s5 with six GPUs + M51
# Re-render an older set with the current layout:
#   SLIDES_VARIANT=6gpu_m51|6gpu|5gpu ./make_all.sh   -> *_<variant>_regen.*
set -euo pipefail
cd "$(dirname "$0")"
for f in s1_hero s2_ttft s3_scaling s4_convergence s5_heatmap; do
  uv run -q --with matplotlib --with pandas --with numpy python "$f.py"
done
rm -rf __pycache__
