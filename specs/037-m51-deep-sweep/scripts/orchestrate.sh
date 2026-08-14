#!/usr/bin/env bash
# specs/037 overnight orchestrator: chains Phase 1 (deep sweep) -> Phase 2
# (cross-size validate) -> Phase 3 (final matrix) -> Phase 4 (report).
#
# Phase 0 pre-flight was run manually before this script's first launch
# (driver md5 verified c9861e9906d03fa2c7d48b804e1a1c80 = f14c51b6f8, max
# clock 980000/5333000/934000 pinned+verified, correctness gate 48
# PASSED/0 FAILED, coherence check "Paris", all 12 PTEs staged on-device).
# This script re-verifies+auto-reflashes the driver before every phase
# anyway, since a multi-hour unattended run can outlast any one check.
#
# Each phase is independently resumable (sweep.py's Optuna journal;
# final_matrix.py's JSONL) -- re-running this script picks up where it left
# off. Does NOT use `set -e`: a phase failing should still let the script
# reach Phase 4 and emit a report from whatever's available (falling back to
# the shipped default winner tokens), not abort the whole night silently.
#
# Usage: nohup setsid ./orchestrate.sh </dev/null > ../results/orchestrate.log 2>&1 &
set -uo pipefail

REPO=/local/yanwen.xu/workspace/dev/executorch
SPEC037="$REPO/specs/037-m51-deep-sweep"
SPEC036="$REPO/specs/036-portable-device-sweep"
PY="$REPO/.venv/bin/python3"
HOST=yanwen.xu@sj1-dmckee-d01
SERIAL=0000088f8e579c33
REQUIRED_MD5=c9861e9906d03fa2c7d48b804e1a1c80
REFLASH_SO=/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so

ts() { date +'%Y-%m-%dT%H:%M:%S'; }
log() { echo "[$(ts)] [$1] $2"; }

pin_max_clocks() {
  ssh -o ConnectTimeout=15 "$HOST" "adb -s $SERIAL root" >/dev/null 2>&1
  sleep 1
  ssh -o ConnectTimeout=15 "$HOST" "adb -s $SERIAL shell 'echo 980000 > /sys/class/devfreq/23400000.sgpu/min_freq; echo 980000 > /sys/class/devfreq/23400000.sgpu/max_freq; echo 5333000 > /sys/class/devfreq/17000010.devfreq_mif/scaling_devfreq_min; echo 5333000 > /sys/class/devfreq/17000010.devfreq_mif/scaling_devfreq_max; echo 934000 > /sys/class/devfreq/17000020.devfreq_int/scaling_devfreq_min; echo 934000 > /sys/class/devfreq/17000020.devfreq_int/scaling_devfreq_max'"
  log CLOCK "max pin (re)applied (980000/5333000/934000)"
}

check_and_reflash_driver() {
  local md5
  md5=$(ssh -o ConnectTimeout=15 "$HOST" "adb -s $SERIAL shell md5sum /vendor/lib64/hw/vulkan.samsung.so" | awk '{print $1}')
  if [ "$md5" = "$REQUIRED_MD5" ]; then
    log DRIVER "OK: $md5"
    return 0
  fi
  log WARN "driver drifted to '$md5' -- auto-reflashing to f14c51b6f8 (confirmed overnight policy)"
  ssh -o ConnectTimeout=15 "$HOST" "adb -s $SERIAL root"; sleep 1
  ssh -o ConnectTimeout=15 "$HOST" "adb -s $SERIAL remount"
  ssh -o ConnectTimeout=15 "$HOST" "adb -s $SERIAL shell setenforce 0"
  ssh -o ConnectTimeout=15 "$HOST" "adb -s $SERIAL shell stop"
  ssh -o ConnectTimeout=15 "$HOST" "adb -s $SERIAL push $REFLASH_SO /vendor/lib64/hw/vulkan.samsung.so"
  ssh -o ConnectTimeout=15 "$HOST" "adb -s $SERIAL shell chmod 644 /vendor/lib64/hw/vulkan.samsung.so"
  ssh -o ConnectTimeout=15 "$HOST" "adb -s $SERIAL remount"
  ssh -o ConnectTimeout=15 "$HOST" "adb -s $SERIAL shell start"
  ssh -o ConnectTimeout=15 "$HOST" "adb -s $SERIAL shell setenforce 1"
  sleep 6
  md5=$(ssh -o ConnectTimeout=15 "$HOST" "adb -s $SERIAL shell md5sum /vendor/lib64/hw/vulkan.samsung.so" | awk '{print $1}')
  if [ "$md5" != "$REQUIRED_MD5" ]; then
    log ABORT "reflash did not produce the required driver (got '$md5')."
    return 1
  fi
  log DRIVER "reflash OK: $md5"
  pin_max_clocks
}

mkdir -p "$SPEC037/results"
log START "specs/037 overnight orchestrator starting (pid $$)"

cd "$SPEC036/scripts"

log PHASE1 "deep sweep: q4gsw (budget 180, maxclk)"
check_and_reflash_driver
"$PY" sweep.py --shader q4gsw --group-size 128 --remote android --slug-suffix maxclk \
  --budget 180 --batch-size 16 --early-stop 40 --finalists 8 --reps 3 \
  2>&1 | tee -a "$SPEC037/results/phase1_q4gsw.log"
log PHASE1 "q4gsw sweep exit"

log PHASE1 "deep sweep: dq8ca (budget 180, maxclk, no_int8_wmma_sg32 quirk)"
check_and_reflash_driver
"$PY" sweep.py --shader dq8ca --group-size 128 --remote android --slug-suffix maxclk \
  --quirk no_int8_wmma_sg32 --budget 180 --batch-size 16 --early-stop 40 --finalists 8 --reps 3 \
  2>&1 | tee -a "$SPEC037/results/phase1_dq8ca.log"
log PHASE1 "dq8ca sweep exit"

cd "$SPEC037/scripts"

log PHASE2 "cross-size validation: q4gsw"
check_and_reflash_driver
"$PY" pick_winner.py --shader q4gsw --slug-suffix maxclk --top-n 3 \
  2>&1 | tee -a "$SPEC037/results/phase2_q4gsw.log"

log PHASE2 "cross-size validation: dq8ca"
check_and_reflash_driver
"$PY" pick_winner.py --shader dq8ca --slug-suffix maxclk --top-n 3 --quirk no_int8_wmma_sg32 \
  2>&1 | tee -a "$SPEC037/results/phase2_dq8ca.log"

Q4GSW_WINNER=$("$PY" -c "import json; print(json.load(open('$SPEC037/results/winner_q4gsw.json'))['winner_token'])" 2>/dev/null || echo "tsweep_t128x128k16g22s32")
DQ8CA_WINNER=$("$PY" -c "import json; print(json.load(open('$SPEC037/results/winner_dq8ca.json'))['winner_token'])" 2>/dev/null || echo "tsweep_t64x32k32g12s64")
log PHASE2 "winners: q4gsw=$Q4GSW_WINNER dq8ca=$DQ8CA_WINNER"

log PHASE3 "final 36-run matrix"
check_and_reflash_driver
"$PY" final_matrix.py --results "$SPEC037/results/final_matrix.jsonl" \
  --status-file "$SPEC037/results/final_matrix.status.json" \
  --q4gsw-winner "$Q4GSW_WINNER" --dq8ca-winner "$DQ8CA_WINNER" \
  --auto-reflash --auto-proceed \
  2>&1 | tee -a "$SPEC037/results/phase3_matrix.log"
log PHASE3 "final matrix exit"

log PHASE4 "generating report"
"$PY" make_report.py --results "$SPEC037/results/final_matrix.jsonl" \
  --winner-json-dir "$SPEC037/results" \
  --out "$SPEC037/results/report.md" \
  2>&1 | tee -a "$SPEC037/results/phase4_report.log"

log DONE "orchestration complete -- see $SPEC037/results/report.md"
