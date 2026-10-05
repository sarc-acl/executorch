#!/bin/bash
# chain16.sh [job to wait for]: device side (run on duck-stable as a screen; the primary confirms what it ranks
# first). Build topic11, the softmax variants with fewer barriers per row (gen_orin_softmax.py, second family),
# all with the QK^T and attn*V kernels of orin-refine1: SDPA screen 5 against 4070ti_nzf, 2 rounds; then, as a
# sanity check only (the reported error is measured on the primary), the error of each against the fp32 reference.
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
V=(4070ti_nzf orin_z64 orin_f64 orin_t32 orin_z32 orin_f32 orin_t16 orin_z16 orin_f16 orin_t8 orin_z8 orin_f8 orin_f4 orin_f2 orin_f1)
E0=ET_VK_SARC_UNVERIFIED=1,ET_VK_SARC_DEV_PROFILE=orin-refine1
P=(); R=(); for v in "${V[@]}"; do P+=("orin-refine1+ET_VK_SARC_SOFTMAX_VARIANT=$v"); R+=("$v=$E0,ET_VK_SARC_SOFTMAX_VARIANT=$v"); done
step ./sdpa_screen.sh sdpa-screen5 topic11 2 "${P[@]}"
step ./sdpa_err.sh sdpa-error3 topic11 "${R[@]}"
echo CHAIN_DONE
