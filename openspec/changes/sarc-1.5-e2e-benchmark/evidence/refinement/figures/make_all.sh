#!/usr/bin/env bash
# Regenerate all refinement figures (PDF + 300-dpi PNG) from ../refinement.csv and ../../../refine/cells.csv.
set -euo pipefail
cd "$(dirname "$0")"
for s in r1_kernel_gain.py r2_e2e_measured.py r2_e2e_projection.py r3_steps.py r4_tok_s_before_after.py; do
  uv run --with matplotlib --with pandas --with numpy python "$s"
done
ls -1 *.pdf *.png
