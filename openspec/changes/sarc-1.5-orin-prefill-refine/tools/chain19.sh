#!/bin/bash
# chain19.sh [job to wait for]: device side, PRIMARY. Build topic13 for every arm but the pristine parent.
#   s5-c3     candidate 3 = the 4w tiles of orin-refine5 on top of candidates 1h + 2. Parent arm: orin-refine3 +
#             4070ti_nzf (the accepted stack), so the session measures the gain over its parent directly. gate.sh.
#   s6-c4     candidate 4 = softmax orin_g128 (subgroup reductions; an arithmetic change in the order of a row's
#             sum) on top of what is accepted by then. gate_sdpa.sh.
#   s7-final  everything accepted, against the pristine parent: timed session and traces (its full gate is the
#             last accepted gate above: same binaries, same environment).
#   s8-noenv  the control of the hook decision: verify.sh on topic13 with nothing selected against the parent control.
# A rejected candidate is left out of the later arms; an aborted gate (device lost, lock, foreign process) ends the chain.
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
ok() { grep -q '^GATE_ACCEPTED' $A/stage/$1/gate.done 2>/dev/null; }
aborted() { grep -q '^GATE_ABORTED' $A/stage/$1/gate.done 2>/dev/null && { echo "CHAIN_STOPPED: $1 aborted"; exit 3; }; }
U="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE"; NZF=ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf; G=ET_VK_SARC_SOFTMAX_VARIANT=orin_g128
BASE="$U=orin-refine3 $NZF"
C3="$U=orin-refine5 $NZF"
step ./stage.sh s5-c3 topic13 "$BASE" topic13 "$C3" "candidate 3: the 4w tiles of orin-refine5 on top of candidates 1h + 2 (parent arm = orin-refine3 + 4070ti_nzf)"
step ./gate.sh s5-c3 "$C3"; aborted s5-c3
P=orin-refine3; ok s5-c3 && P=orin-refine5
BASE="$U=$P $NZF"; C4="$U=$P $G"
step ./stage.sh s6-c4 topic13 "$BASE" topic13 "$C4" "candidate 4: softmax orin_g128 on top of $P + 4070ti_nzf (the parent arm)"
step ./gate_sdpa.sh s6-c4 "$C4"; aborted s6-c4
F=$BASE; ok s6-c4 && F=$C4
step ./stage.sh s7-final parent "" topic13 "$F" "final: everything accepted ($F) against the pristine parent"
step ./timed.sh s7-final "$F"
step ./noenv_verify.sh topic13 s8-noenv
echo CHAIN_DONE
