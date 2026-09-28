#!/usr/bin/env bash
# Regenerate evidence figures e1-e4 for seven GPUs (PDF + 200-dpi PNG) from evidence/combined6/, the RX 7600
# contribution (contrib/rx7600/: roofline.json, efficiency.csv, trace/families.csv; paths in common.py) and raw7/cells.csv.
# Earlier versions are kept: *_5gpu.* (CAPTIONS_5gpu.md) and *_6gpu.* (CAPTIONS_6gpu.md); not regenerated.
# Each script prints the plotted numbers as CSV on stdout for cross-checking.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")"
for f in e1_roofs_vs_kernels e2_time_breakdown e3_amdahl_model_size e4_speedup_decomposition; do
  uv run -q --with matplotlib --with pandas --with numpy python "$f.py" > "$f.values.csv"

done
