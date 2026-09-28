#!/usr/bin/env bash
# Regenerate evidence figures e1-e4 (PDF + 200-dpi PNG). Inputs are read-only; see CAPTIONS.md.
# Each script prints the plotted numbers as CSV on stdout for cross-checking.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")"
for f in e1_roofs_vs_kernels e2_time_breakdown e3_amdahl_model_size e4_speedup_decomposition; do
  uv run -q --with matplotlib --with pandas --with numpy python "$f.py" > "$f.values.csv"

done
