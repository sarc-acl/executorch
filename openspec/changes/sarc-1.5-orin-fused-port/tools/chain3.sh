#!/bin/bash
# chain3.sh [job to wait for]: device side. Candidate 1 = the fused attention kernel (profile orin-fused1) on
# build topic1, against the parent (build parent, the first campaign's final stack):
#   c1-pre        one correctness pass per tier (all, extended, full, peaked, fused) with the candidate environment.
#                 Any case that is not PASSED stops the chain here: a failed kernel is diagnosed, not gated;
#   sdpa-error1   error of the attention block against the fp32 CPU reference, same test binary and inputs: the
#                 stock kernels, the parent's kernels, the candidate (reference-error rule, criterion 1);
#   sdpa-screen1  kernel-level time per layer at S = 2048, 3 rounds, interleaved: parent, candidate 1 (one pass,
#                 packed), and the unpacked / two-pass forms of the same kernel (the candidate-2 question of
#                 thresholds.txt: do the copy passes cost more than they save);
#   s3-c1         the gate (gate_sdpa.sh): SDPA tiers, unmodified verify.sh, interleaved session, traces.
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
BT=topic1; U=ET_VK_SARC_UNVERIFIED=1
PENV="$U ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64"; C1="$U ET_VK_SARC_DEV_PROFILE=orin-fused1"
B=$A/build/$BT/bundle/test_llama_microbench; O=$A/raw/c1-pre; mkdir -p $O
for tier in all extended full peaked fused; do
  [[ -s $O/$tier.log ]] && continue
  cool_start 120; echo "== $(date -u +%FT%TZ) c1-pre $tier"
  env $C1 ./gl.sh $B --sdpa-correctness-only --sdpa-tier=$tier > $O/$tier.log 2>&1; rc=$?
  echo "c1-pre $tier rc=$rc cases=$(grep -c '^\[sdpa-correctness\] .* S=' $O/$tier.log) passed=$(grep -c '^\[sdpa-correctness\] .* PASSED$' $O/$tier.log)" | tee -a $O/summary.txt
  [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }
done
if grep -q 'FAILED' $O/*.log || [[ $(grep -c 'rc=0 ' $O/summary.txt) != 5 ]]; then echo "CHAIN_STOPPED: c1-pre has a case that is not PASSED; candidate 1 is not gated"; grep -h 'FAILED' $O/*.log | cut -c1-300 | head -20; exit 3; fi
step ./sdpa_err.sh sdpa-error1 $BT stock "parent=${PENV// /,}" "fused1=${C1// /,}"
RO=ET_VK_SARC_ORIN_SDPA_FUSED=fused3sb_d64_t32x32g11s32ro,fused3sb_d128_t16x64g11s32ro
RK=ET_VK_SARC_ORIN_SDPA_FUSED=fused3sb_d64_t32x32g11s32rk,fused3sb_d128_t16x64g11s32rk
RR=ET_VK_SARC_ORIN_SDPA_FUSED=fused3sb_d64_t32x32g11s32r,fused3sb_d128_t16x64g11s32r
step ./sdpa_screen.sh sdpa-screen1 $BT 3 orin-refine5+ET_VK_SARC_SOFTMAX_VARIANT=orin_g64 orin-fused1 orin-fused1+$RO orin-fused1+$RK orin-fused1+$RR
python3 sdpa_screen_summary.py $A/raw/sdpa-screen1 > $A/raw/sdpa-screen1/summary.csv; cat $A/raw/sdpa-screen1/summary.csv
step ./stage.sh s3-c1 parent "$PENV" $BT "$C1" "candidate 1: the fused attention kernel (profile orin-fused1) on topic1 against the parent build with the parent environment"
step ./gate_sdpa.sh s3-c1 "$C1"
cat $A/stage/s3-c1/gate.done 2>/dev/null
echo CHAIN_DONE
