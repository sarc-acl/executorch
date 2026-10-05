#!/bin/bash
# chain10.sh [job to wait for]: device side. Candidate 2 = orin-lin-refine2 (8da4w: whole-texel weight staging,
# orin_bf_t128x128k64g24s32mk32ra), build topic6, against the pristine parent. The kernel claims not to change
# the arithmetic, so first: bit comparison of its output with the shipped kernel's on the 12 model shapes
# (raw/bit-bf) and the sampled production-diff on all three models; then gate.sh s3-c2.
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
E2="ET_VK_SARC_DEV_PROFILE=orin-lin-refine2"
step ./bitcmp.sh bit-bf topic6 8da4w orin_bf_t128x128k64g24s32mk32ra orin_bf_t128x128k64g42s32mk32ra
step ./pdiff.sh pdiff-bf2 topic6 8da4w 1b,3b,8b orin_bf_t128x128k64g24s32mk32ra
step ./stage.sh s3-c2 parent "" topic6 "$E2" "candidate 2: orin-lin-refine2 (8da4w whole-texel weight staging) against the pristine parent"
step ./gate.sh s3-c2 "$E2"
echo CHAIN_DONE
