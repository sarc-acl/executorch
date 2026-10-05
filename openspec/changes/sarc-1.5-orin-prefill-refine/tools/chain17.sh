#!/bin/bash
# chain17.sh [job to wait for]: device side (duck-stable: a screen; the primary confirms). Build topic12, softmax
# third batch (gen_orin_softmax.py): more workers per row, subgroup reductions, and the measurement-only pass
# twins xp0 / xp1 / xp2 (wrong output by construction: timed only). SDPA screen 6 against 4070ti_nzf and f64, 2
# rounds; then, as a sanity check, the error of the real variants against the fp32 reference.
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
V=(4070ti_nzf orin_f64 orin_f128 orin_f256 orin_g32 orin_g64 orin_g128 orin_g256 orin_g512)
E0=ET_VK_SARC_UNVERIFIED=1,ET_VK_SARC_DEV_PROFILE=orin-refine1
P=(); R=(); for v in "${V[@]}"; do P+=("orin-refine1+ET_VK_SARC_SOFTMAX_VARIANT=$v"); R+=("$v=$E0,ET_VK_SARC_SOFTMAX_VARIANT=$v"); done
for v in orin_xp0 orin_xp1 orin_xp2; do P+=("orin-refine1+ET_VK_SARC_SOFTMAX_VARIANT=$v"); done
step ./sdpa_screen.sh sdpa-screen6 topic12 2 "${P[@]}"
step ./sdpa_err.sh sdpa-error4 topic12 "${R[@]}"
echo CHAIN_DONE
