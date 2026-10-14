#!/bin/bash
# gate.sh <session> "<cand env>" [sdpa] [reps]: the gate of one staged candidate (stage.sh), one GPU job at a time:
#   1. wait until the GPU is at most idle + 3 C (48 C; at most 30 min), then e2e5.sh parent vs candidate (<reps> valid
#      runs per cell, next token on the timed, the real-text and the unaligned prompt)
#   2. with "sdpa": test_llama_microbench --sdpa-correctness-only, tiers all, extended and full, 12 passes each with the
#      candidate env, one control pass each without it (the parent env)
#   3. unmodified sarc/tools/verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff with the candidate env on the
#      candidate's binaries, compared line by line with s0-parent-verify (rates removed)
#   4. warm ETDump traces of both arms, per-family breakdown
source "$(dirname "$(readlink -f "$0")")/env.sh"
S=$1; ENVS=$2; SDPA=${3:-}; REPS=${4:-$(sed -n 's/^reps=//p' $T/thresholds.txt)}; D=$A/stage/$S
PENV=$(tr '\n' ' ' < $D/parent/env)
st() { echo "$(date -u +%FT%TZ) $*" >> $D/gate.status; }
st "gate start reps=$REPS sdpa=$SDPA env=[$ENVS]"
eval "$($T/hold.sh vars)"
while :; do
  $T/hold.sh wait "gate.sh $S: timed session"
  t0=$SECONDS; while (( $(gtemp) > 48 && SECONDS - t0 < 1800 )); do sleep 10; done
  [[ -e $HOLD ]] || break
done
echo "prestart_temp=$(gtemp) waited=$((SECONDS - t0))s $(date -u +%FT%TZ)" > $D/prestart.txt
$T/e2e5.sh --stage $D --out raw --reps $REPS > $D/e2e5.out 2>&1
/usr/bin/python3 $T/summarize.py $D/raw $REPS > $D/raw/summary.csv 2>&1
st "timed session done"
if [[ $SDPA == sdpa ]]; then
  $T/sdpa_tiers.sh $S "$ENVS" 12 "all extended full" cand
  $T/sdpa_tiers.sh $S "$PENV" 1 "all extended full" table
  st "sdpa tiers done"
fi
$T/verify_stage.sh $S "$ENVS"
/usr/bin/python3 $T/verify_compare.py $A/stage/s0-parent-verify/verify.out $D/verify.out > $D/verify-compare.txt 2>&1
st "verify done: $(tail -1 $D/verify-compare.txt)"
$T/trace.sh $S 1b,3b,8b 4w,8da4w "parent cand" > $D/trace.out 2>&1
st "trace done"; st GATE_DONE
