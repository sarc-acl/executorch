#!/bin/bash
# chain18.sh [job to wait for]: device side, PRIMARY. Build topic13. Confirmation of what the second device's
# screens ranked first, and the pre-checks of the next two candidates:
#   1. SDPA screen 7: softmax 4070ti_nzf against orin_f64, orin_g64, orin_g128 (2 rounds, the kernels of orin-refine1);
#   2. error of the attention block against the fp32 reference for stock, 4070ti_nzf, orin_g64, orin_g128, and the
#      raw outputs of the last three (the direct difference of g64 / g128 from 4070ti_nzf is computed from them);
#   3. the 4w tile of orin-lin-refine5: bit comparison with the shipped kernels and production-diff on all models.
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
E0=ET_VK_SARC_UNVERIFIED=1,ET_VK_SARC_DEV_PROFILE=orin-refine1
P=(); R=("stock"); for v in 4070ti_nzf orin_f64 orin_g64 orin_g128; do P+=("orin-refine1+ET_VK_SARC_SOFTMAX_VARIANT=$v"); done
for v in 4070ti_nzf orin_g64 orin_g128; do mkdir -p $A/raw/sdpa-error5/dump-$v; R+=("$v=$E0,ET_VK_SARC_SOFTMAX_VARIANT=$v,ET_VK_SDPA_DUMP_DIR=$A/raw/sdpa-error5/dump-$v"); done
step ./sdpa_screen.sh sdpa-screen7 topic13 2 "${P[@]}"
step ./sdpa_err.sh sdpa-error5 topic13 "${R[@]}"
for v in orin_g64 orin_g128; do python3 sdpa_diff.py $A/raw/sdpa-error5/dump-4070ti_nzf $A/raw/sdpa-error5/dump-$v > $A/raw/sdpa-error5/diff-$v-vs-4070ti_nzf.csv; echo "diff $v rc=$?"; done
step ./bitcmp.sh bit-q4b topic13 4w orin_t256x128k16g42s32bt bx_t128x128k32g42s32f32c
step ./pdiff.sh pdiff-q4b topic13 4w 1b,3b,8b base orin_t256x128k16g42s32bt
echo CHAIN_DONE
