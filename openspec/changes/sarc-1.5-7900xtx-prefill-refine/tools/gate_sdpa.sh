#!/bin/bash
# gate_sdpa.sh <session> "<cand env>": gate.sh for a candidate that changes SDPA kernels. Same steps as gate.sh,
# preceded by the SDPA checks that sarc/tools/verify.sh does not run:
#   0a. test_llama_microbench --sdpa-correctness-only (tier all), 12 back-to-back passes with the candidate env
#       (the microbench header asks for 10+ repeats before a coopmat SDPA pass is trusted), and 3 passes without
#       it (the table kernels, as the control);
#   0b. the SDPA perf suite (--sdpa) with and without the candidate env, for the dispatched kernels and times.
A=${ART780M:-$HOME/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-03}; S=$1; ENVS=$2; D=$A/stage/$S; B=$D/test_llama_microbench
t0=$SECONDS; while (( $(cat /sys/class/hwmon/hwmon2/temp1_input) > 48000 && SECONDS - t0 < 300 )); do sleep 5; done
O=$D/sdpa-correctness; mkdir -p $O
for i in $(seq 1 12); do env $ENVS $A/tools/gl.sh $B --sdpa-correctness-only > $O/cand-r$i.log 2>&1; echo "cand r$i rc=$? see summary.txt"; done > $O/summary.txt
for i in 1 2 3; do $A/tools/gl.sh $B --sdpa-correctness-only > $O/table-r$i.log 2>&1; echo "table r$i rc=$? $(grep -c 'PASS' $O/table-r$i.log) pass-lines $(grep -ci 'FAIL\|MISMATCH' $O/table-r$i.log) fail-lines"; done >> $O/summary.txt
env $ENVS $A/tools/gl.sh $B --sdpa --json-out=$O/perf-cand.json > $O/perf-cand.log 2>&1; echo "perf cand rc=$?" >> $O/summary.txt
$A/tools/gl.sh $B --sdpa --json-out=$O/perf-table.json > $O/perf-table.log 2>&1; echo "perf table rc=$?" >> $O/summary.txt
env $ENVS $A/tools/hold.sh run "verify.sh $S" $HOME/hmz-sarc/executorch/sarc/tools/verify.sh --dir $D --lock 00000000-c400-0000-0000-000000000000 \
  --models 1b,3b,8b --schemes 4w,8da4w --pdiff --out verify > $D/verify.out 2>&1
echo "VERIFY_DONE rc=$?" >> $D/verify.out
$A/tools/session.sh $S > $D/e2e5.out 2>&1
$A/tools/trace.sh $S 1b,3b,8b 4w,8da4w "parent cand" > $D/trace.out 2>&1
python3 $A/tools/summarize.py $D/raw > $D/raw/summary.csv 2>&1
echo GATE_DONE > $D/gate.done
