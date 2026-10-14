#!/bin/bash
# chain9.sh [job to wait for]: device side. Decode A/B of the parent and the two candidates (3 runs each, 1B);
# 8da4w screen 3 (build topic5): the second batch of whole-texel tiles against the shipped kernel and the best
# tiles of screen 2, production-diff on the 1B shapes first, 2 rounds; phase timing of the two best forms.
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
E1=ET_VK_SARC_UNVERIFIED=1,ET_VK_SARC_DEV_PROFILE=orin-refine1
step ./decode_ab.sh decode-ab1 3 1b parent=parent "refine1=topic4:$E1" lin2=topic4:ET_VK_SARC_DEV_PROFILE=orin-lin-refine2
step ./pdiff.sh pdiff-bf3 topic5 8da4w 1b orin_bf_t64x128k64g42s32mk32ra orin_bf_t64x128k64g22s32mk32ra orin_bf_t64x64k128g24s32mk32ra orin_bf_t64x64k128g42s32mk32ra orin_bf_t64x64k64g22s32mk32ra orin_bf_t128x128k64g22s32mk32ra
step ./screen.sh screen3-8da4w topic5 8da4w 2 base orin_bf_t128x128k64g24s32mk32ra orin_bf_t128x128k64g42s32mk32ra orin_bf_t64x128k64g42s32mk32ra orin_bf_t64x128k64g22s32mk32ra orin_bf_t64x64k128g24s32mk32ra orin_bf_t64x64k128g42s32mk32ra orin_bf_t64x64k64g22s32mk32ra orin_bf_t128x128k64g22s32mk32ra
step ./prof.sh prof3 topic5 8da4w:ET_VK_SARC_DQ8CA_VARIANT:orin_bf_t128x128k64g24s32mk32rap:128:128:64 8da4w:ET_VK_SARC_DQ8CA_VARIANT:orin_bf_t64x128k64g42s32mk32rap:64:128:64
echo CHAIN_DONE
