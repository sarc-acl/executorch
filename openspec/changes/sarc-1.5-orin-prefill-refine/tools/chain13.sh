#!/bin/bash
# chain13.sh [job to wait for]: device side. The evidence of the reference-error rule (second owner decision of
# 2026-10-04) for candidate 1h, which the owner's decision of 2026-10-05 requires "as written": the full
# last-position logits of the 41 real-text prompts for the four arms (parent default, parent tiled, candidate
# default, candidate tiled), all six cells; then, on the device (numpy), the position logits of the gate's
# unaligned prompt, the real-text comparison and the rule itself (probe/refine1-nzf/).
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
E="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine1 ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf"
step ./probe_run.sh parent parent-default
step ./probe_run.sh topic6 nzf-default $E
step ./probe_run.sh parent parent-tiled ET_VK_FORCE_TILED_LINEAR=1
step ./probe_run.sh topic6 nzf-tiled $E ET_VK_FORCE_TILED_LINEAR=1
P=$A/probe; O=$P/refine1-nzf; mkdir -p $O
python3 probe_position.py $P/broad $O parent-default parent-tiled nzf-default nzf-tiled > $O/differing-items.txt; echo "position rc=$? $(cat $O/differing-items.txt | tr '\n' ';')"
python3 probe_compare.py $P/broad parent-default parent-tiled nzf-default nzf-tiled > $O/compare.csv; echo "compare rc=$? $(tail -1 $O/compare.csv)"
cp $A/raw/sdpa-error2/stock.txt $O/sdpa-error-parent.txt; cp $A/raw/sdpa-error2/refine1-nzf.txt $O/sdpa-error-candidate.txt
printf '%s\n' $E > $O/cand.env
mapfile -t ITEMS < <(grep . $O/differing-items.txt)   # no empty item when nothing differs
python3 ref_error_rule.py $O refine1-nzf $O/cand.env $O/sdpa-error-parent.txt $O/sdpa-error-candidate.txt $O/compare.csv "${ITEMS[@]}" > $O/reference-error-rule.txt; echo "rule rc=$? $(tail -1 $O/reference-error-rule.txt)"
echo CHAIN_DONE
