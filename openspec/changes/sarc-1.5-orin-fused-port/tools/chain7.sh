#!/bin/bash
# chain7.sh [job to wait for]: device side. The closing chain, on build topic3 (the last commit that changes code).
# The final stack is fixed by a rule written before s6-c2 has a number (STATUS.md, 2026-10-09 01:22 UTC; owner
# decision 2026-10-09 01:00 UTC): orin-fused2 only if s6-c2 is GATE_ACCEPTED and its geomean gain over candidate 1
# is at least 2 % (outside the noise band); in every other case orin-fused1, whose gate on topic3 against the
# parent is s5-c1. Without an accepted s5-c1 nothing here runs.
#   s7n-noenv     hook control (D4): unmodified verify.sh on topic3 with nothing selected against s0n-noenv;
#   s7-final      only when the final stack is orin-fused2: its full gate against the tuned parent (gate_sdpa.sh);
#   s8-pristine   the final stack against the pristine state (the parent build, no environment): timed session,
#                 traces;
#   probe         41-prompt real-text logits of the final stack on topic3, default and tiled, the comparison with
#                 the parent's two arms (chain4) and the reference-error rule on sdpa-error2 (probe/final-fused/);
#   mem1          memory probe of the K / V copies: parent environment against the final stack, 2 rounds;
#   roof          fresh roofs, igpu-roofline `fast` (raw/roof-final/), clocks as found.
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
ok() { grep -q '^GATE_ACCEPTED' $A/stage/$1/gate.done 2>/dev/null; }
BT=topic3; U=ET_VK_SARC_UNVERIFIED=1
PENV="$U ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64"; C1="$U ET_VK_SARC_DEV_PROFILE=orin-fused1"; C2="$U ET_VK_SARC_DEV_PROFILE=orin-fused2"
ok s5-c1 || { echo "CHAIN_STOPPED: s5-c1 is not accepted; there is no final stack"; exit 3; }
F=$C1; ARM=fused1; G=$(sed -n 's/^geomean over 6 cells: \([0-9.]*\) .*/\1/p' $A/stage/s6-c2/raw/summary.csv 2>/dev/null)
if ok s6-c2 && [[ -n $G ]] && awk -v g="$G" 'BEGIN {exit !(g >= 1.02)}'; then F=$C2; ARM=fused2; fi
echo "final stack: [$F] (s6-c2: $(cut -c1-60 $A/stage/s6-c2/gate.done 2>/dev/null), geomean ratio ${G:-none})" | tee $A/FINAL.txt
step env PARENT_CTL_NAME=s0n-noenv ./noenv_verify.sh $BT s7n-noenv
if [[ $ARM == fused2 ]]; then
  step ./stage.sh s7-final parent "$PENV" $BT "$F" "final stack (orin-fused2) against the parent build with the parent environment: full gate"
  step ./gate_sdpa.sh s7-final "$F"
  ok s7-final || { echo "CHAIN_STOPPED: s7-final is not accepted"; exit 3; }
fi
step ./stage.sh s8-pristine parent "" $BT "$F" "final stack ($F) against the pristine state: the parent build with no environment"
step ./timed.sh s8-pristine "$F"
step ./probe_run.sh $BT final-default $F
step ./probe_run.sh $BT final-tiled $F ET_VK_FORCE_TILED_LINEAR=1
P=$A/probe; O=$P/final-fused; mkdir -p $O
python3 probe_position.py $P/broad $O parent-default parent-tiled final-default final-tiled > $O/differing-items.txt; echo "position rc=$? $(cat $O/differing-items.txt | tr '\n' ';')"
python3 probe_compare.py $P/broad parent-default parent-tiled final-default final-tiled > $O/compare.csv; echo "compare rc=$? $(tail -1 $O/compare.csv)"
cp $A/raw/sdpa-error2/parent.txt $O/sdpa-error-parent.txt; cp $A/raw/sdpa-error2/$ARM.txt $O/sdpa-error-candidate.txt; cp $A/raw/sdpa-error2/stock.txt $O/sdpa-error-stock.txt
printf '%s\n' $F > $O/cand.env
mapfile -t ITEMS < <(grep . $O/differing-items.txt)
python3 ref_error_rule.py $O final-fused $O/cand.env $O/sdpa-error-parent.txt $O/sdpa-error-candidate.txt $O/compare.csv "${ITEMS[@]}" > $O/reference-error-rule.txt; echo "rule rc=$? $(tail -1 $O/reference-error-rule.txt)"
step ./memprobe.sh mem1 $BT 2 "parent=${PENV// /,}" "$ARM=${F// /,}"
step ./roof.sh final
echo CHAIN_DONE
