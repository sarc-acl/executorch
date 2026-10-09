#!/bin/bash
# chain6.sh [job to wait for]: device side. Everything again on build topic3 (the last commit that changes code:
# the microbench's fused support as insert-only blocks, R3) and with the thermal throttle record in every timed run:
#   s4-aa2        A/A re-check: the parent build against topic3, both with the parent environment, under the
#                 committed clock floor (not a calibration; thresholds unchanged);
#   sdpa-error2   error against the fp32 CPU reference with topic3's test binary: stock, the campaign's parent,
#                 candidate 1 (orin-fused1), candidate 2 (orin-fused2), same seeded inputs;
#   s5-c1         candidate 1 gated again (gate_sdpa.sh): parent build with the parent environment against topic3
#                 with orin-fused1;
#   c2-pre        candidate 2 = orin-fused2 (two-pass fused kernel for head_dim 64, one-pass for 128): one
#                 correctness pass per tier; a case that is not PASSED stops the chain;
#   s6-c2         candidate 2 gated (gate_sdpa.sh) against its parent = candidate 1, both on topic3.
# (chain5, the first form of the candidate-2 chain on topic2, was ended before it started a job.)
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
BT=topic3; U=ET_VK_SARC_UNVERIFIED=1
PENV="$U ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64"; C1="$U ET_VK_SARC_DEV_PROFILE=orin-fused1"; C2="$U ET_VK_SARC_DEV_PROFILE=orin-fused2"
D=$A/stage/s4-aa2
step ./stage.sh s4-aa2 parent "$PENV" $BT "$PENV" "A/A re-check with the thermal throttle record: the parent build against topic3, both with the parent environment"
step ./session.sh s4-aa2 --clkmin-file $CHANGE/results/orin/clkmin.json > $D/e2e5.out 2>&1; tail -2 $D/e2e5.out
python3 summarize.py $D/raw > $D/raw/summary.csv 2>&1; cat $D/raw/summary.csv
python3 gate_check.py session $D/raw --clkmin $CHANGE/results/orin/clkmin.json --require-logs > $D/session-check.txt 2>&1; tail -2 $D/session-check.txt
step ./sdpa_err.sh sdpa-error2 $BT stock "parent=${PENV// /,}" "fused1=${C1// /,}" "fused2=${C2// /,}"
step ./stage.sh s5-c1 parent "$PENV" $BT "$C1" "candidate 1 again on topic3: the fused attention kernel (profile orin-fused1) against the parent build with the parent environment"
step ./gate_sdpa.sh s5-c1 "$C1"
cat $A/stage/s5-c1/gate.done 2>/dev/null
grep -q '^GATE_ACCEPTED' $A/stage/s5-c1/gate.done 2>/dev/null || { echo "CHAIN_STOPPED: s5-c1 is not accepted; candidate 2 has no accepted parent"; exit 3; }
B=$A/build/$BT/bundle/test_llama_microbench; O=$A/raw/c2-pre; mkdir -p $O
for tier in all extended full peaked fused; do
  [[ -s $O/$tier.log ]] && continue
  cool_start 120; echo "== $(date -u +%FT%TZ) c2-pre $tier"
  env $C2 ./gl.sh $B --sdpa-correctness-only --sdpa-tier=$tier > $O/$tier.log 2>&1; rc=$?
  echo "c2-pre $tier rc=$rc cases=$(grep -c '^\[sdpa-correctness\] .* S=' $O/$tier.log) passed=$(grep -c '^\[sdpa-correctness\] .* PASSED$' $O/$tier.log)" | tee -a $O/summary.txt
  [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }
done
if grep -q 'FAILED' $O/*.log || [[ $(grep -c 'rc=0 ' $O/summary.txt) != 5 ]]; then echo "CHAIN_STOPPED: c2-pre has a case that is not PASSED; candidate 2 is not gated"; grep -h 'FAILED' $O/*.log | cut -c1-300 | head -20; exit 3; fi
step ./stage.sh s6-c2 $BT "$C1" $BT "$C2" "candidate 2: profile orin-fused2 (two-pass fused kernel for head_dim 64, one-pass for 128) against candidate 1 (orin-fused1), both on topic3"
step ./gate_sdpa.sh s6-c2 "$C2"
cat $A/stage/s6-c2/gate.done 2>/dev/null
echo CHAIN_DONE
