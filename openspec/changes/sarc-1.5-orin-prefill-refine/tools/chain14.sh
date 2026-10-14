#!/bin/bash
# chain14.sh [job to wait for]: device side (either Orin; on duck-stable a screen whose ranking the primary
# confirms). Build topic10, the softmax variants that read a row once (gen_orin_softmax.py), all with the QK^T
# and attn*V kernels of orin-refine1: SDPA screen 4 against 4070ti_nzf, 2 rounds; then the bit comparison of the
# attention output with 4070ti_nzf's (the N = 2 variants exercise the reload path on every case).
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
P=(); for v in 4070ti_nzf orin_l8 orin_l8e orin_l16 orin_l16e orin_l32 orin_l32e orin_s8 orin_s8e orin_s16 orin_s16e orin_s32 orin_s32e; do P+=("orin-refine1+ET_VK_SARC_SOFTMAX_VARIANT=$v"); done
step ./sdpa_screen.sh sdpa-screen4 topic10 2 "${P[@]}"
step ./sdpa_bitcmp.sh sdpa-bit1 topic10 orin-refine1 4070ti_nzf orin_l2 orin_l2e orin_s2 orin_s2e orin_l8 orin_l8e orin_l16 orin_l16e orin_l32 orin_l32e orin_s8 orin_s8e orin_s16 orin_s16e orin_s32 orin_s32e
echo CHAIN_DONE
