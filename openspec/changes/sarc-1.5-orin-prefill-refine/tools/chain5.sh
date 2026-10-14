#!/bin/bash
# chain5.sh: device side, after chain4. Build topic3: SDPA screen 2 (the best QK^T and attn*V kernels of screen 1
# and the 12 Orin packed-staging QK^T tiles, 2 rounds, no stock arm), then phase timing of the Orin 4w tiles and
# of two whole-texel 8da4w twins.
cd "$(dirname "$0")"
while ! grep -q "^DONE\|^KILLED" ~/hmz-sarc-orin/jobs/${1:-chain4}.status; do sleep 20; done   # $1 = the job to wait for
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
O=orin-qk-orin_pk
step ./sdpa_screen.sh sdpa-screen2 topic3 2 orin-qk-pk_t128x64k64g42s32nf orin-qk-pk_t128x64k32g42s32nf orin-qk-pk_t64x64k32g22s32nf orin-qk-pk_t64x64k32g21s32nf \
  ${O}_t64x64k64g22s32nf ${O}_t64x64k64g21s32nf ${O}_t64x64k64g42s32nf ${O}_t128x64k64g24s32nf ${O}_t128x64k64g44s32nf ${O}_t128x64k64g22s32nf \
  ${O}_t64x128k64g42s32nf ${O}_t32x64k64g42s32nf ${O}_t64x64k128g22s32nf ${O}_t64x64k128g42s32nf ${O}_t64x64k128g21s32nf ${O}_t64x64k128g44s32nf \
  orin-av-t64x64k32g42s32 orin-av-t64x64k32g24s32 orin-av-ml_t64x128k32g42s32 orin-av-ml_t128x128k32g44s32 orin-av-ml_t64x128k32g44s32
step ./prof.sh prof2 topic3 4w:ET_VK_SARC_Q4GSW_VARIANT:t256x128k16g22s32p:256:128:16 4w:ET_VK_SARC_Q4GSW_VARIANT:t128x128k32g42s32f32p:128:128:32 \
  8da4w:ET_VK_SARC_DQ8CA_VARIANT:orin_bf1_t128x128k128g44s32mk32rap:128:128:128 8da4w:ET_VK_SARC_DQ8CA_VARIANT:orin_bf_t128x128k64g44s32mk32rap:128:128:64
echo CHAIN_DONE
