#!/bin/bash
# chain7.sh [job to wait for]: device side, after the gates of chain6. Screens and evidence, one GPU job at a time:
#   1. 8da4w screen 3 (build topic5): the second batch of whole-texel tiles against the shipped kernel and
#      candidate 2's tile, production-diff on the 1B shapes first, 2 rounds; phase timing of candidate 2's tile;
#   2. 4w screen 1 (build topic5): every existing subgroup-32 dev tile (the 780M sweep tiles incl. texel-wise
#      staging `bx`, the 4070 Ti `ga` tiles) against the Orin rows, 1 round;
#   3. softmax variants through the local hook (build hook4, NOT committed): reference error of the attention
#      block for stock, orin-refine1 and orin-refine1 with each softmax variant (same test binary), and their
#      kernel times.
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
B2="orin_bf_t64x128k64g42s32mk32ra orin_bf_t64x128k64g22s32mk32ra orin_bf_t64x64k128g24s32mk32ra orin_bf_t64x64k128g42s32mk32ra orin_bf_t64x64k64g22s32mk32ra orin_bf_t128x128k64g22s32mk32ra"
step ./pdiff.sh pdiff-bf3 topic5 8da4w 1b $B2
step ./screen.sh screen3-8da4w topic5 8da4w 2 base orin_bf_t128x128k64g24s32mk32ra orin_bf_t128x128k64g42s32mk32ra $B2
step ./prof.sh prof3 topic5 8da4w:ET_VK_SARC_DQ8CA_VARIANT:orin_bf_t128x128k64g24s32mk32rap:128:128:64 8da4w:ET_VK_SARC_DQ8CA_VARIANT:orin_bf_t64x128k64g42s32mk32rap:64:128:64
step ./screen.sh screen1-4w topic5 4w 1 base t128x128k16g22s32 t128x128k32g42s32 t128x128k32g24s32 t128x128k32g42s32f32 t128x128k32g42s32f32c t128x128k32g24s32f32c \
  t256x128k32g24s32f32c t256x128k32g44s32f32c t128x256k32g42s32f32c t128x128k32g42s32f32cbt t128x128k32g22s32f32c t256x128k32g42s32f32c t64x256k32g41s32f32c \
  bx_t128x256k32g42s32f32c bx_t128x128k32g42s32f32c 4070ti_t256x128k16g42s32gac 4070ti_t256x128k16g24s32gac 4070ti_t256x256k16g44s32gac 4070ti_t128x256k16g42s32gac \
  4070ti_t128x128k32g42s32gac 4070ti_t128x128k32g44s32gac 4070ti_t128x128k32g24s32gac 4070ti_t128x128k16g42s32gac 4070ti_t256x128k16g42s32gacbt 4070ti_t128x128k32g42s32gacbt
if [[ -x $A/build/hook4/bundle/test_llama_microbench ]]; then
  E1=ET_VK_SARC_UNVERIFIED=1,ET_VK_SARC_DEV_PROFILE=orin-refine1
  step ./sdpa_err.sh sdpa-error2 hook4 stock "refine1=$E1" "refine1-f32=$E1,ET_VK_SARC_SOFTMAX_VARIANT=4070ti_f32" "refine1-nz=$E1,ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nz" "refine1-nzf=$E1,ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf"
  step ./sdpa_screen.sh sdpa-screen3 hook4 2 orin-refine1 orin-refine1+ET_VK_SARC_SOFTMAX_VARIANT=4070ti_f32 orin-refine1+ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nz orin-refine1+ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf
fi
echo CHAIN_DONE
