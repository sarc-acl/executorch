#!/bin/bash
# chain4.sh [job to wait for]: device side. Real-text evidence for candidate 1 under the reference-error rule
# (owner decision D3): full last-position logits of the 41-prompt set for the four arms, default and tiled linear:
#   parent-default, parent-tiled   build parent with the parent environment
#   c1-default, c1-tiled           build topic1 with the candidate environment (profile orin-fused1)
# then the logits at the position of the gate's unaligned item, the comparison of the arms and ref_error_rule.py
# (probe/c1-fused/). The error files are those of sdpa-error1 (chain3): arm parent and arm fused1. Before that,
# the peaked tier's error for both arms (raw/sdpa-error1-peaked/, recorded only).
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
U=ET_VK_SARC_UNVERIFIED=1; PENV="$U ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64"; C1="$U ET_VK_SARC_DEV_PROFILE=orin-fused1"
step ./probe_run.sh parent parent-default $PENV
step ./probe_run.sh parent parent-tiled $PENV ET_VK_FORCE_TILED_LINEAR=1
step ./probe_run.sh topic1 c1-default $C1
step ./probe_run.sh topic1 c1-tiled $C1 ET_VK_FORCE_TILED_LINEAR=1
# For the record (not part of the rule): the same error on the peaked tier (sharp attention rows, the rescale path
# of the one-pass kernel), parent kernels against the candidate, same binary and inputs.
E=$A/raw/sdpa-error1-peaked; mkdir -p $E; B=$A/build/topic1/bundle/test_llama_microbench
for arm in "parent:$PENV" "fused1:$C1"; do n=${arm%%:*}; [[ -s $E/$n.txt ]] && continue; cool_start 120
  env ${arm#*:} ./gl.sh $B --sdpa-correctness-only --sdpa-tier=peaked > $E/$n-peaked.log 2>&1; rc=$?; echo "peaked $n rc=$rc"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && exit $rc
  grep -h '^\[sdpa-error\]\|^\[sdpa-correctness\]\|^\[sdpa-kernels\]\|^\[sarc_dev\]' $E/$n-peaked.log > $E/$n.txt; done
P=$A/probe; O=$P/c1-fused; mkdir -p $O
python3 probe_position.py $P/broad $O parent-default parent-tiled c1-default c1-tiled > $O/differing-items.txt; echo "position rc=$? $(cat $O/differing-items.txt | tr '\n' ';')"
python3 probe_compare.py $P/broad parent-default parent-tiled c1-default c1-tiled > $O/compare.csv; echo "compare rc=$? $(tail -1 $O/compare.csv)"
cp $A/raw/sdpa-error1/parent.txt $O/sdpa-error-parent.txt; cp $A/raw/sdpa-error1/fused1.txt $O/sdpa-error-candidate.txt; cp $A/raw/sdpa-error1/stock.txt $O/sdpa-error-stock.txt
printf '%s\n' $C1 > $O/cand.env
mapfile -t ITEMS < <(grep . $O/differing-items.txt)
python3 ref_error_rule.py $O c1-fused $O/cand.env $O/sdpa-error-parent.txt $O/sdpa-error-candidate.txt $O/compare.csv "${ITEMS[@]}" > $O/reference-error-rule.txt; echo "rule rc=$? $(tail -1 $O/reference-error-rule.txt)"
echo CHAIN_DONE
