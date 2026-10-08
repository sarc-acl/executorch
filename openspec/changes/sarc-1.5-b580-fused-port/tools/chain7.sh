#!/bin/bash
# chain7.sh: the selection screen of thresholds.txt (kernel_screen): screen5-select on build topic5, 3 rounds,
# every run behind the idle wait and cooled: the parent's three kernels, the two incumbents (the literal 780M
# variants) and, per head_dim, the two fastest variants of the one-round look screen4-fused as challengers.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"; ST=$A/logs/chain7.status
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
say "chain7 start"
P="b580-refine3 b580-fused-d64_t32x32s32m8ro b580-fused-d128_t16x64s32m8ro b580-fused-d64_t16x64s16m8g4roj b580-fused-d64_t16x64s16m8g4oj b580-fused-d128_t16x128s16m8g8oj b580-fused-d128_t16x64s16m8g4oj"
$TOOLS/screen_sdpa.sh screen5-select topic5 3 $P > $A/logs/screen5-select.out 2>&1; rc=$?
python3 $TOOLS/screen_sdpa_summary.py $A/raw/screen5-select/screen.csv b580-refine3 > $A/raw/screen5-select/summary.csv 2>&1
say "screen5-select rc=$rc $(tail -1 $A/raw/screen5-select/env.txt)"
say "CHAIN7_DONE"
