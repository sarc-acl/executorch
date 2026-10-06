#!/bin/bash
# sdpa_tiers.sh <session> "<env>" <passes> [tiers="all extended full"] [tag=cand]: test_llama_microbench
# --sdpa-correctness-only of stage/<session> per tier, <passes> passes, each one gl.sh job (lock, hold, guard).
# Output stage/<session>/sdpa-correctness/{<tag>-<tier>-r<i>.log,summary.txt}.
source "$(dirname "$(readlink -f "$0")")/env.sh"
S=$A/stage/$1; ENVS=$2; N=$3; TIERS=${4:-all extended full}; TAG=${5:-cand}; O=$S/sdpa-correctness; mkdir -p $O
for tier in $TIERS; do for i in $(seq 1 $N); do
  t0=$SECONDS; while (( $(gtemp) > 60 && SECONDS - t0 < 300 )); do sleep 5; done
  env $ENVS $T/gl.sh $S/test_llama_microbench --sdpa-correctness-only --sdpa-tier=$tier > $O/$TAG-$tier-r$i.log 2>&1
  echo "$TAG $tier r$i rc=$? passed=$(grep -c 'PASSED' $O/$TAG-$tier-r$i.log) failed=$(grep -c 'FAILED' $O/$TAG-$tier-r$i.log) mismatch_nonzero=$(grep -oE 'mismatch(es)?[=: ]+[1-9][0-9]*' $O/$TAG-$tier-r$i.log | wc -l) pairing_not_ok=$(grep 'sdpa-kernels' $O/$TAG-$tier-r$i.log | grep -vc 'pairing=ok')" >> $O/summary.txt
done; done
