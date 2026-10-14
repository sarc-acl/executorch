#!/bin/bash
# gate2.sh <session> "<cand env>" [sdpa]: the round-2 gate for one staged candidate (stage.sh), one GPU job at a
# time, the sweep paused meanwhile. Same checks as gate.sh / gate_sdpa.sh, reordered so that the timed session
# starts from a cool device:
#   1. wait until the GPU is at most 43 C (at most 30 min), then e2e5.sh parent vs candidate (5 valid runs per
#      cell, next token on both prompts)
#   2. with "sdpa": test_llama_microbench --sdpa-correctness-only, tiers all, extended and full, 12 passes each
#      with the candidate env, one control pass each without it; the SDPA perf suite with and without it
#   3. unmodified sarc/tools/verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff with the candidate env
#   4. warm ETDump traces of both arms
A=${ART780M:-$HOME/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-04}; S=$1; ENVS=$2; SDPA=${3:-}; D=$A/stage/$S
B=$D/test_llama_microbench; HW=/sys/class/hwmon/hwmon2; LOCK=00000000-c400-0000-0000-000000000000
touch $A/PAUSE; trap 'rm -f $A/PAUSE' EXIT
eval "$($A/tools/hold.sh vars)"   # coordinator hold: wait it out before the cool start, and cool again after one
while :; do
  $A/tools/hold.sh wait "gate2.sh $S: timed session"
  t0=$SECONDS; while (( $(cat $HW/temp1_input) > 43000 && SECONDS - t0 < 1800 )); do sleep 10; done
  [[ -e $HOLD ]] || break
done
echo "prestart_temp=$(( $(cat $HW/temp1_input) / 1000 )) waited=$((SECONDS - t0))s $(date -u +%FT%TZ)" > $D/prestart.txt
$A/tools/e2e5.sh --stage $D --out raw --lock $LOCK > $D/e2e5.out 2>&1
python3 $A/tools/summarize.py $D/raw > $D/raw/summary.csv 2>&1
if [[ $SDPA == sdpa ]]; then
  O=$D/sdpa-correctness; mkdir -p $O; : > $O/summary.txt
  for tier in all extended full; do
    for i in $(seq 1 12); do
      t0=$SECONDS; while (( $(cat $HW/temp1_input) > 55000 && SECONDS - t0 < 300 )); do sleep 5; done
      env $ENVS $A/tools/gl.sh $B --sdpa-correctness-only --sdpa-tier=$tier > $O/cand-$tier-r$i.log 2>&1
      echo "cand $tier r$i rc=$? passed=$(grep -c 'PASSED' $O/cand-$tier-r$i.log) failed=$(grep -c 'FAILED' $O/cand-$tier-r$i.log) pairing_not_ok=$(grep 'sdpa-kernels' $O/cand-$tier-r$i.log | grep -vc 'pairing=ok')" >> $O/summary.txt
    done
    $A/tools/gl.sh $B --sdpa-correctness-only --sdpa-tier=$tier > $O/table-$tier-r1.log 2>&1
    echo "table $tier r1 rc=$? passed=$(grep -c 'PASSED' $O/table-$tier-r1.log) failed=$(grep -c 'FAILED' $O/table-$tier-r1.log)" >> $O/summary.txt
  done
  env $ENVS $A/tools/gl.sh $B --sdpa --json-out=$O/perf-cand.json > $O/perf-cand.log 2>&1; echo "perf cand rc=$?" >> $O/summary.txt
  $A/tools/gl.sh $B --sdpa --json-out=$O/perf-table.json > $O/perf-table.log 2>&1; echo "perf table rc=$?" >> $O/summary.txt
fi
env $ENVS $A/tools/hold.sh run "verify.sh $S" $HOME/hmz-sarc/executorch/sarc/tools/verify.sh --dir $D --lock $LOCK \
  --models 1b,3b,8b --schemes 4w,8da4w --pdiff --out verify > $D/verify.out 2>&1
echo "VERIFY_DONE rc=$?" >> $D/verify.out
ART780M=$A $A/tools/trace.sh $S 1b,3b,8b 4w,8da4w "parent cand" > $D/trace.out 2>&1
echo GATE_DONE > $D/gate.done
