#!/bin/bash
# gate_sdpa.sh <session> "<cand env>": gate.sh for a candidate that changes SDPA kernels. Same steps as gate.sh,
# preceded by the SDPA checks that sarc/tools/verify.sh does not run:
#   0a. test_llama_microbench --sdpa-correctness-only, tiers extended and full, 12 passes each with the candidate
#       env (0 mismatches and pairing=ok required), and one control pass of each tier without it;
#   0b. the SDPA perf suite (--sdpa) with and without the candidate env, for the dispatched kernels and times.
T=$(dirname "$0"); source "$T/common.sh"; S=$1; ENVS=$2; D=$A/stage/$S; B=$D/test_llama_microbench
cool_start 50 300
O=$D/sdpa-correctness; mkdir -p $O
for tier in extended full; do
  for i in $(seq 1 12); do cool_start 60 300
    env $ENVS $T/gl.sh $B --sdpa-correctness-only --sdpa-tier=$tier > $O/cand-$tier-r$i.log 2>&1
    echo "cand $tier r$i rc=$? pass-lines=$(grep -c 'PASS' $O/cand-$tier-r$i.log) fail-lines=$(grep -ci 'FAIL\|MISMATCH' $O/cand-$tier-r$i.log) pairing-ok=$(grep -c 'pairing=ok' $O/cand-$tier-r$i.log) pairing-other=$(grep 'pairing=' $O/cand-$tier-r$i.log | grep -vc 'pairing=ok')"
  done
  $T/gl.sh $B --sdpa-correctness-only --sdpa-tier=$tier > $O/table-$tier-r1.log 2>&1
  echo "table $tier r1 rc=$? pass-lines=$(grep -c 'PASS' $O/table-$tier-r1.log) fail-lines=$(grep -ci 'FAIL\|MISMATCH' $O/table-$tier-r1.log)"
done > $O/summary.txt
env $ENVS $T/gl.sh $B --sdpa --json-out=$O/perf-cand.json > $O/perf-cand.log 2>&1; echo "perf cand rc=$?" >> $O/summary.txt
$T/gl.sh $B --sdpa --json-out=$O/perf-table.json > $O/perf-table.log 2>&1; echo "perf table rc=$?" >> $O/summary.txt
env $ENVS $ET/sarc/tools/verify.sh --dir $D --lock $LOCK \
  --models 1b,3b,8b --schemes 4w,8da4w --pdiff --out verify > $D/verify.out 2>&1
echo "VERIFY_DONE rc=$?" >> $D/verify.out
$T/session.sh $S > $D/e2e5.out 2>&1
$T/trace.sh $S 1b,3b,8b 4w,8da4w "parent cand" > $D/trace.out 2>&1
python3 $T/summarize.py $D/raw > $D/raw/summary.csv 2>&1
echo GATE_DONE > $D/gate.done
