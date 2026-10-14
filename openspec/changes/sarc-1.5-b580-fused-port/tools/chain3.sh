#!/bin/bash
# chain3.sh: third detached chain (after the desktop went into use during s1-aa).
#   1. logits-probe builds for parent2 and topic1 (container builds, no GPU job runs meanwhile)
#   2. error against the fp32 reference, parent kernels and b580-fused1 on the same binary (not timed)
#   3. baseline + A/A with calibration, session s1-aa2 (every timed run behind the idle wait)
#   4. kernel-level screen screen1-fused: parent kernels, b580-fused1 and the 16 fused variants, 3 rounds,
#      every run behind the idle wait and cooled
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"; ST=$A/logs/chain3.status
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
say "chain3 start"
for t in parent2 topic1; do $TOOLS/build-probe.sh $t > $A/logs/build-probe-$t.out 2>&1; say "build-probe $t rc=$? $(tail -1 $A/logs/build-probe-$t.out)"; done
$TOOLS/sdpa_ref.sh c1-ref topic1 b580-fused1 > $A/logs/c1-ref.out 2>&1; say "c1-ref rc=$? $(tail -1 $A/raw/c1-ref/full.csv) / extended $(tail -1 $A/raw/c1-ref/extended.csv) / peaked $(tail -1 $A/raw/c1-ref/peaked.csv)"
$TOOLS/stage.sh s1-aa2 parent2 "$PARENT_ENV" topic1 "$PARENT_ENV" "baseline + A/A: parent 51d9d757f against the topic build, both with the parent environment; idle desktop" > $A/logs/s1-aa2.stage.out 2>&1 || { say "CHAIN3_STOPPED staging s1-aa2 failed"; exit 1; }
$TOOLS/session.sh s1-aa2 --calibrate > $A/logs/s1-aa2.out 2>&1; say "s1-aa2 rc=$? $(tail -1 $A/stage/s1-aa2/raw/done.txt 2>/dev/null) $(grep calibration: $A/stage/s1-aa2/raw/env.txt)"
V="d64_t32x32s32m8ro d64_t16x32s16m8ro d64_t16x32s16m8o d64_t8x32s16m8ro d64_t16x64s16m8ro d64_t8x64s16m8ro d64_t16x32s16m8r
   d128_t16x64s32m8ro d128_t16x64s16m8ro d128_t16x64s16m8o d128_t8x64s16m8ro d128_t8x64s16m8o d128_t16x32s16m8ro d128_t8x32s16m8ro d128_t16x64s16m8r d128_t8x64s16m8r"
P="b580-refine3 b580-fused1"; for v in $V; do P+=" b580-fused-$v"; done
$TOOLS/screen_sdpa.sh screen1-fused topic1 3 $P > $A/logs/screen1-fused.out 2>&1; rc=$?
python3 $TOOLS/screen_sdpa_summary.py $A/raw/screen1-fused/screen.csv b580-refine3 > $A/raw/screen1-fused/summary.csv 2>&1
say "screen1-fused rc=$rc $(tail -1 $A/raw/screen1-fused/env.txt)"
say "CHAIN3_DONE"
