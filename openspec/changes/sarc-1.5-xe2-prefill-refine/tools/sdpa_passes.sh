#!/bin/bash
# sdpa_passes.sh <session> <arm: cand|table> <passes> "<env>": <passes> runs each of
# test_llama_microbench --sdpa-correctness-only with tiers all, extended and full, cooling before each pass.
# Every log is kept as sdpa-correctness/<arm>-<tier>-r<i>.log and its return status appended to
# sdpa-correctness/rc.csv (arm,tier,pass,rc); gate_check.py reads both. Stops at a foreign GPU process (76).
. "$(dirname "$(readlink -f "$0")")/host.sh"; D=$A/stage/$1; ARM=$2; N=$3; ENVS=$4
O=$D/sdpa-correctness; mkdir -p $O || exit 2
for tier in all extended full; do for i in $(seq 1 $N); do
  cool_start
  env $ENVS $TOOLS/gl.sh $D/test_llama_microbench --sdpa-correctness-only --sdpa-tier=$tier > $O/$ARM-$tier-r$i.log 2>&1; rc=$?
  echo "$ARM,$tier,$i,$rc" >> $O/rc.csv; echo "sdpa $ARM $tier r$i rc=$rc"
  [[ $rc == 75 || $rc == 76 ]] && exit $rc
done; done
