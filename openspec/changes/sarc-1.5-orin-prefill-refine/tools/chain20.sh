#!/bin/bash
# chain20.sh [job to wait for]: device side, PRIMARY. The real-text evidence for the final stack (candidates 1h +
# 2 + 3 + 4; candidate 4 changes the order of the softmax row sum): full last-position logits of the 41 prompts
# for final default and final tiled on build topic13; the parent's two arms are those of chain13 (pristine parent
# build, unchanged). Then the position logits of the gate's unaligned prompt, the comparison and the
# reference-error rule on these files (probe/final-g64/), with the reference error measured on the primary for
# orin_g64 (raw/sdpa-error5).
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
E="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64"
step ./probe_run.sh topic13 final-default $E
step ./probe_run.sh topic13 final-tiled $E ET_VK_FORCE_TILED_LINEAR=1
P=$A/probe; O=$P/final-g64; mkdir -p $O
python3 probe_position.py $P/broad $O parent-default parent-tiled final-default final-tiled > $O/differing-items.txt; echo "position rc=$? $(cat $O/differing-items.txt | tr '\n' ';')"
python3 probe_compare.py $P/broad parent-default parent-tiled final-default final-tiled > $O/compare.csv; echo "compare rc=$? $(tail -1 $O/compare.csv)"
cp $A/raw/sdpa-error5/stock.txt $O/sdpa-error-parent.txt; cp $A/raw/sdpa-error5/orin_g64.txt $O/sdpa-error-candidate.txt
printf '%s\n' $E > $O/cand.env
mapfile -t ITEMS < <(grep . $O/differing-items.txt)
python3 ref_error_rule.py $O final-g64 $O/cand.env $O/sdpa-error-parent.txt $O/sdpa-error-candidate.txt $O/compare.csv "${ITEMS[@]}" > $O/reference-error-rule.txt; echo "rule rc=$? $(tail -1 $O/reference-error-rule.txt)"
echo CHAIN_DONE
