#!/bin/bash
# chain5.sh [job to wait for]: device side. Candidate 2 = profile orin-fused2 on build topic2: the faster form of the
# fused kernel per head_dim at kernel level (sdpa-screen1: two passes for head_dim 64, one pass for 128), against
# its parent = candidate 1 (profile orin-fused1) on the same build:
#   c2-pre        one correctness pass per tier with the candidate environment; a case that is not PASSED stops the chain;
#   sdpa-error2   error against the fp32 CPU reference on build topic2: stock, the campaign's parent, candidate 1,
#                 candidate 2 (same test binary and inputs);
#   s4-c2         the gate (gate_sdpa.sh): SDPA tiers, unmodified verify.sh against s0-parent-verify, interleaved
#                 session candidate 1 against candidate 2, traces.
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
BT=topic2; U=ET_VK_SARC_UNVERIFIED=1
PENV="$U ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64"; C1="$U ET_VK_SARC_DEV_PROFILE=orin-fused1"; C2="$U ET_VK_SARC_DEV_PROFILE=orin-fused2"
B=$A/build/$BT/bundle/test_llama_microbench; O=$A/raw/c2-pre; mkdir -p $O
for tier in all extended full peaked fused; do
  [[ -s $O/$tier.log ]] && continue
  cool_start 120; echo "== $(date -u +%FT%TZ) c2-pre $tier"
  env $C2 ./gl.sh $B --sdpa-correctness-only --sdpa-tier=$tier > $O/$tier.log 2>&1; rc=$?
  echo "c2-pre $tier rc=$rc cases=$(grep -c '^\[sdpa-correctness\] .* S=' $O/$tier.log) passed=$(grep -c '^\[sdpa-correctness\] .* PASSED$' $O/$tier.log)" | tee -a $O/summary.txt
  [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }
done
if grep -q 'FAILED' $O/*.log || [[ $(grep -c 'rc=0 ' $O/summary.txt) != 5 ]]; then echo "CHAIN_STOPPED: c2-pre has a case that is not PASSED; candidate 2 is not gated"; grep -h 'FAILED' $O/*.log | cut -c1-300 | head -20; exit 3; fi
step ./sdpa_err.sh sdpa-error2 $BT stock "parent=${PENV// /,}" "fused1=${C1// /,}" "fused2=${C2// /,}"
step ./stage.sh s4-c2 $BT "$C1" $BT "$C2" "candidate 2: profile orin-fused2 (two-pass fused kernel for head_dim 64, one-pass for 128) against candidate 1 (orin-fused1), both on topic2"
step ./gate_sdpa.sh s4-c2 "$C2"
cat $A/stage/s4-c2/gate.done 2>/dev/null
echo CHAIN_DONE
