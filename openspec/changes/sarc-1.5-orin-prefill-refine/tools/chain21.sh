#!/bin/bash
# chain21.sh [job to wait for]: device side, PRIMARY. Build topic14 = the corrected orin_g softmax (review finding
# 2026-10-06: in topic13 every lane of a subgroup stored the reduced value into its subgroup's slot, a data race
# by the Vulkan memory model; now the elected lane alone stores it). Candidate 4 is gated and measured again:
#   sdpa-error6  error of the attention block against the fp32 reference for stock, 4070ti_nzf and orin_g64, the
#                raw outputs of the last two, and the direct difference of the corrected orin_g64 from 4070ti_nzf
#                and from topic13's orin_g64 (raw/sdpa-error5/dump-orin_g64);
#   s9-c4r       candidate 4 over its parent (candidates 1h + 2 + 3), both arms on topic14: gate_sdpa.sh (12
#                extended + 12 full SDPA passes, unmodified verify.sh, interleaved session, traces);
#   s10-final    the corrected final stack against the pristine parent: timed session and traces;
#   s11-noenv    verify.sh on topic14 with nothing selected against the parent control;
#   probe        41-prompt real-text logits of the corrected final stack, default and tiled, then the comparison
#                and the reference-error rule (probe/final-g64r/; the parent's two arms are those of chain13).
# The coordinator uses the device under the same gpu-lab lock for a separate measurement: the chain first waits,
# without a time limit, until the lock is free and no process of ~/llamacpp-compare is left, and every tool takes the lock for each GPU job as before.
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
echo "== $(date -u +%FT%TZ) waiting for the coordinator's session and the gpu-lab lock"
while :; do flock "$HOME/.cache/gpu-lab/lock-$LOCK" true; pgrep -f 'llamacpp-compare/' > /dev/null || break; sleep 30; done; echo "== $(date -u +%FT%TZ) lock free"
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
ok() { grep -q '^GATE_ACCEPTED' $A/stage/$1/gate.done 2>/dev/null; }
U="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE"; NZF=ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf; G=ET_VK_SARC_SOFTMAX_VARIANT=orin_g64
BT=topic14; BASE="$U=orin-refine5 $NZF"; C4="$U=orin-refine5 $G"
E0=ET_VK_SARC_UNVERIFIED=1,ET_VK_SARC_DEV_PROFILE=orin-refine1
R=("stock"); for v in 4070ti_nzf orin_g64; do mkdir -p $A/raw/sdpa-error6/dump-$v; R+=("$v=$E0,ET_VK_SARC_SOFTMAX_VARIANT=$v,ET_VK_SDPA_DUMP_DIR=$A/raw/sdpa-error6/dump-$v"); done
step ./sdpa_err.sh sdpa-error6 $BT "${R[@]}"
python3 sdpa_diff.py $A/raw/sdpa-error6/dump-4070ti_nzf $A/raw/sdpa-error6/dump-orin_g64 > $A/raw/sdpa-error6/diff-orin_g64-vs-4070ti_nzf.csv; echo "diff vs nzf rc=$?"
python3 sdpa_diff.py $A/raw/sdpa-error5/dump-orin_g64 $A/raw/sdpa-error6/dump-orin_g64 > $A/raw/sdpa-error6/diff-orin_g64-vs-topic13-orin_g64.csv; echo "diff vs topic13 rc=$?"
step ./stage.sh s9-c4r $BT "$BASE" $BT "$C4" "candidate 4 again on the corrected build: softmax orin_g64 (elected-lane store) on top of orin-refine5 + 4070ti_nzf (the parent arm)"
step ./gate_sdpa.sh s9-c4r "$C4"
grep -q '^GATE_ABORTED' $A/stage/s9-c4r/gate.done 2>/dev/null && { echo "CHAIN_STOPPED: s9-c4r aborted"; exit 3; }
F=$BASE; ok s9-c4r && F=$C4
step ./stage.sh s10-final parent "" $BT "$F" "final on the corrected build: everything accepted ($F) against the pristine parent"
step ./timed.sh s10-final "$F"
step ./noenv_verify.sh $BT s11-noenv
ok s9-c4r || { echo "CHAIN_DONE (candidate 4 not accepted: no probe)"; exit 0; }
step ./probe_run.sh $BT finalr-default $C4
step ./probe_run.sh $BT finalr-tiled $C4 ET_VK_FORCE_TILED_LINEAR=1
P=$A/probe; O=$P/final-g64r; mkdir -p $O
python3 probe_position.py $P/broad $O parent-default parent-tiled finalr-default finalr-tiled > $O/differing-items.txt; echo "position rc=$? $(cat $O/differing-items.txt | tr '\n' ';')"
python3 probe_compare.py $P/broad parent-default parent-tiled finalr-default finalr-tiled > $O/compare.csv; echo "compare rc=$? $(tail -1 $O/compare.csv)"
cp $A/raw/sdpa-error6/stock.txt $O/sdpa-error-parent.txt; cp $A/raw/sdpa-error6/orin_g64.txt $O/sdpa-error-candidate.txt
printf '%s\n' $C4 > $O/cand.env
mapfile -t ITEMS < <(grep . $O/differing-items.txt)
python3 ref_error_rule.py $O final-g64r $O/cand.env $O/sdpa-error-parent.txt $O/sdpa-error-candidate.txt $O/compare.csv "${ITEMS[@]}" > $O/reference-error-rule.txt; echo "rule rc=$? $(tail -1 $O/reference-error-rule.txt)"
echo CHAIN_DONE
