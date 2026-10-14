#!/bin/bash
# chain2.sh: second detached chain, started while chain1 runs; it waits for CHAIN1_DONE.
#   kernel-level screen of the fused variants against the parent's three attention kernels (b580-refine3):
#   screen_sdpa.sh screen1-fused, build topic1, 3 rounds, cooled before every run, resumable.
# The rule that reads it is in thresholds.txt (kernel_screen); nothing is selected here.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"; ST=$A/logs/chain2.status
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
say "chain2 waiting for chain1"
until grep -q 'CHAIN1_DONE\|CHAIN1_STOPPED' $A/logs/chain1.status 2>/dev/null; do sleep 30; done
grep -q CHAIN1_DONE $A/logs/chain1.status || { say "CHAIN2_STOPPED chain1 did not finish"; exit 1; }
V="d64_t16x32s16m8ro d64_t16x32s16m8o d64_t8x32s16m8ro d64_t16x64s16m8ro d64_t8x64s16m8ro d64_t16x32s16m8r d64_t32x32s32m8ro
   d128_t16x64s16m8ro d128_t16x64s16m8o d128_t8x64s16m8ro d128_t8x64s16m8o d128_t16x32s16m8ro d128_t8x32s16m8ro d128_t16x64s16m8r d128_t8x64s16m8r d128_t16x64s32m8ro"
P="b580-refine3 b580-fused1"; for v in $V; do P+=" b580-fused-$v"; done
$TOOLS/screen_sdpa.sh screen1-fused topic1 3 $P > $A/logs/screen1-fused.out 2>&1; rc=$?
python3 $TOOLS/screen_sdpa_summary.py $A/raw/screen1-fused/screen.csv b580-refine3 > $A/raw/screen1-fused/summary.csv 2>&1
say "screen1-fused rc=$rc $(tail -1 $A/raw/screen1-fused/env.txt)"
say "CHAIN2_DONE"
