#!/bin/bash
# chain8.sh [job to wait for]: device side. chain6 again (which was ended before it started a job), on the build
# that has the one-full-subgroup check in the fused kernel (topic4 = 0bed38090, taken from the 4070 Ti port):
#   g-pre         orin-fused1 on topic4, one correctness pass per tier (all, extended, full, peaked, fused). All
#                 cases PASSED: everything below runs on topic4. Otherwise (or topic4 not deployed) everything
#                 runs on topic3, the build without the check, and the failure is a finding (raw/g-pre/). The
#                 build used is written to BUILD.txt, which chain9 reads;
#   s4-aa2        A/A re-check: the parent build against that build, both with the parent environment, under the
#                 committed clock floor (not a calibration; thresholds unchanged), with the throttle record;
#   sdpa-error2   error against the fp32 CPU reference with that build's test binary: stock, the campaign's
#                 parent, candidate 1 (orin-fused1), candidate 2 (orin-fused2), same seeded inputs;
#   s5-c1         candidate 1 gated again (gate_sdpa.sh): parent build with the parent environment against that
#                 build with orin-fused1;
#   c2-pre        candidate 2 = orin-fused2 (two-pass fused kernel for head_dim 64, one-pass for 128): one
#                 correctness pass per tier; a case that is not PASSED stops the chain;
#   s6-c2         candidate 2 gated (gate_sdpa.sh) against its parent = candidate 1, both on that build.
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
U=ET_VK_SARC_UNVERIFIED=1
PENV="$U ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64"; C1="$U ET_VK_SARC_DEV_PROFILE=orin-fused1"; C2="$U ET_VK_SARC_DEV_PROFILE=orin-fused2"
BT=topic4; O=$A/raw/g-pre; mkdir -p $O; B=$A/build/topic4/bundle/test_llama_microbench
if [[ ! -s $A/BUILD.txt ]]; then
  if [[ -x $B && -s $A/build/topic4/bundle/logits_dump ]]; then
    for tier in all extended full peaked fused; do
      [[ -s $O/$tier.log ]] && continue
      cool_start 120; echo "== $(date -u +%FT%TZ) g-pre $tier"
      env $C1 ./gl.sh $B --sdpa-correctness-only --sdpa-tier=$tier > $O/$tier.log 2>&1; rc=$?
      echo "g-pre $tier rc=$rc cases=$(grep -c '^\[sdpa-correctness\] .* S=' $O/$tier.log) passed=$(grep -c '^\[sdpa-correctness\] .* PASSED$' $O/$tier.log) fused=$(grep -c '^\[sdpa-kernels\] .* fused=.*fused3sb' $O/$tier.log)" | tee -a $O/summary.txt
      [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }
    done
    if grep -q 'FAILED' $O/*.log || [[ $(grep -c 'rc=0 ' $O/summary.txt) != 5 ]] || [[ $(awk '{split($4, c, "="); split($5, p, "="); if (c[2] == 0 || c[2] != p[2]) n++} END {print n + 0}' $O/summary.txt) != 0 ]]; then BT=topic3; fi
  else BT=topic3; echo "topic4 is not deployed" | tee -a $O/summary.txt; fi
  echo $BT > $A/BUILD.txt
fi
BT=$(cat $A/BUILD.txt); echo "== build for everything below: $BT"; cat $O/summary.txt 2>/dev/null
D=$A/stage/s4-aa2
step ./stage.sh s4-aa2 parent "$PENV" $BT "$PENV" "A/A re-check with the thermal throttle record: the parent build against $BT, both with the parent environment"
step ./session.sh s4-aa2 --clkmin-file $CHANGE/results/orin/clkmin.json > $D/e2e5.out 2>&1; tail -2 $D/e2e5.out
python3 summarize.py $D/raw > $D/raw/summary.csv 2>&1; cat $D/raw/summary.csv
python3 gate_check.py session $D/raw --clkmin $CHANGE/results/orin/clkmin.json --require-logs > $D/session-check.txt 2>&1; tail -2 $D/session-check.txt
step ./sdpa_err.sh sdpa-error2 $BT stock "parent=${PENV// /,}" "fused1=${C1// /,}" "fused2=${C2// /,}"
step ./stage.sh s5-c1 parent "$PENV" $BT "$C1" "candidate 1 again on $BT: the fused attention kernel (profile orin-fused1) against the parent build with the parent environment"
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
step ./stage.sh s6-c2 $BT "$C1" $BT "$C2" "candidate 2: profile orin-fused2 (two-pass fused kernel for head_dim 64, one-pass for 128) against candidate 1 (orin-fused1), both on $BT"
step ./gate_sdpa.sh s6-c2 "$C2"
cat $A/stage/s6-c2/gate.done 2>/dev/null
echo CHAIN_DONE
