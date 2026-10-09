#!/bin/bash
# chain13.sh: the timed sessions that ran behind busy_wait on 2026-10-09 (s3b-c2, s4-final, s5-pristine), again
# under the authorized protocol: every timed run starts at once whatever the desktop reports (owner decision
# 2026-10-09 00:22 UTC); BUSYMAX 5.0 %, CLKMIN 2635 MHz, 7 repeats, rejection and replacement unchanged. Build
# topic7 (e1e450530) as before; the earlier sessions stay where they are.
#   1. s7-pristine: pristine (no environment) against topic7 with b580-fused1, timed session only
#   2. s3c-c2: topic7 b580-fused1 against topic7 b580-fused2, timed session only (the gate items of candidate 2
#      that do not depend on the start time are those of s3b-c2)
#   3. s6-final: parent2 with the parent environment against topic7 with b580-fused1, gate_sdpa.sh (full gate)
#   4. collection
# Ends CHAIN13_DONE.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"; TAG=topic7
ST=$A/logs/chain13.status; say() { echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
U="ET_VK_SARC_UNVERIFIED=1"; C1="$U ET_VK_SARC_DEV_PROFILE=b580-fused1"; C2="$U ET_VK_SARC_DEV_PROFILE=b580-fused2"
say "chain13 start, build $TAG $(sed -n 's/^commit=//p' $A/build/$TAG.src.txt), tools $(git -C $ET rev-parse --short HEAD), kernel $(uname -r)"
timed() { # timed <session>: session.sh with 7 repeats, then the summary
  gpu_shared || { say "CHAIN13_STOPPED build lock"; exit 75; }
  $TOOLS/session.sh $1 --reps 7 > $A/stage/$1/e2e5.out 2>&1; say "session $1 rc=$? $(tail -1 $A/stage/$1/e2e5.out | cut -c1-120)"
  python3 $TOOLS/summarize.py $A/stage/$1/raw > $A/stage/$1/raw/summary.csv 2>&1; say "$1: $(grep -h 'geomean\|INCOMPLETE\|UNADJUDICATED' $A/stage/$1/raw/summary.csv | cut -c1-60 | tr '\n' ';')"; }
$TOOLS/stage.sh s7-pristine pristine "" $TAG "$C1" "final stack b580-fused1 on the committed head against the pristine parent of the first campaign (no profile); repeats s5-pristine without busy_wait" > $A/logs/s7-pristine.stage.out 2>&1 || { say "CHAIN13_STOPPED staging s7-pristine failed"; exit 1; }
timed s7-pristine
$TOOLS/stage.sh s3c-c2 $TAG "$C1" $TAG "$C2" "candidate 2 b580-fused2 against candidate 1 (b580-fused1), same build; timed session only, repeats the timing of s3b-c2 without busy_wait" > $A/logs/s3c-c2.stage.out 2>&1 || { say "CHAIN13_STOPPED staging s3c-c2 failed"; exit 1; }
timed s3c-c2
$TOOLS/stage.sh s6-final parent2 "$PARENT_ENV" $TAG "$C1" "final stack b580-fused1 on the committed head against the parent (b580-refine3); repeats the gate of s4-final without busy_wait" > $A/logs/s6-final.stage.out 2>&1 || { say "CHAIN13_STOPPED staging s6-final failed"; exit 1; }
B580_REPS=7 $TOOLS/gate_sdpa.sh s6-final > $A/logs/gate-s6-final.out 2>&1; say "gate s6-final rc=$? $(cat $A/stage/s6-final/gate.done 2>/dev/null)"
$TOOLS/collect.sh > $A/logs/collect-chain13.out 2>&1; say "collect rc=$?"
say "CHAIN13_DONE"
