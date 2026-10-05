#!/bin/bash
# chain11.sh [job to wait for]: device side. 4w screen 1 (build topic5): every existing subgroup-32 dev tile (the
# 780M sweep tiles incl. texel-wise staging, the 4070 Ti `ga` tiles) against the Orin rows, 1 round. Then the
# softmax variants through the local hook (build hook4, NOT committed): reference error of the attention block for
# stock, orin-refine1 and orin-refine1 with each softmax variant (same test binary), and their kernel times.
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
step ./screen.sh screen1-4w topic5 4w 1 base t128x128k16g22s32 t128x128k32g42s32 t128x128k32g24s32 t128x128k32g42s32f32 t128x128k32g42s32f32c t128x128k32g24s32f32c \
  t256x128k32g24s32f32c t256x128k32g44s32f32c t128x256k32g42s32f32c t128x128k32g42s32f32cbt t128x128k32g22s32f32c t256x128k32g42s32f32c t64x256k32g41s32f32c \
  bx_t128x256k32g42s32f32c bx_t128x128k32g42s32f32c 4070ti_t256x128k16g42s32gac 4070ti_t256x128k16g24s32gac 4070ti_t256x256k16g44s32gac 4070ti_t128x256k16g42s32gac \
  4070ti_t128x128k32g42s32gac 4070ti_t128x128k32g44s32gac 4070ti_t128x128k32g24s32gac 4070ti_t128x128k16g42s32gac 4070ti_t256x128k16g42s32gacbt 4070ti_t128x128k32g42s32gacbt
E1=ET_VK_SARC_UNVERIFIED=1,ET_VK_SARC_DEV_PROFILE=orin-refine1
step ./sdpa_err.sh sdpa-error2 hook4 stock "refine1=$E1" "refine1-f32=$E1,ET_VK_SARC_SOFTMAX_VARIANT=4070ti_f32" "refine1-nz=$E1,ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nz" "refine1-nzf=$E1,ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf"
step ./sdpa_screen.sh sdpa-screen3 hook4 2 orin-refine1 orin-refine1+ET_VK_SARC_SOFTMAX_VARIANT=4070ti_f32 orin-refine1+ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nz orin-refine1+ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf
echo CHAIN_DONE
