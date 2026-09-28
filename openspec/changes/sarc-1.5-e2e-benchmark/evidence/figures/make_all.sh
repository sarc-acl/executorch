#!/usr/bin/env bash
# Regenerate evidence figures e1-e4 for six GPUs (PDF + 200-dpi PNG) from evidence/combined6/ and raw6/cells.csv.
# The five-GPU versions are kept as *_5gpu.* (inputs: evidence/*.csv, cells.csv); see CAPTIONS_5gpu.md.
# Each script prints the plotted numbers as CSV on stdout for cross-checking.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")"
for f in e1_roofs_vs_kernels e2_time_breakdown e3_amdahl_model_size e4_speedup_decomposition; do
  uv run -q --with matplotlib --with pandas --with numpy python "$f.py" > "$f.values.csv"

done
