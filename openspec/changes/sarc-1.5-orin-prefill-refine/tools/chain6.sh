#!/bin/bash
# chain6.sh [job to wait for]: device side. Candidate 1 (orin-refine1, SDPA prefill kernels, build topic4) and
# candidate 2 (orin-lin-refine2, 8da4w whole-texel weight staging, the same build), each against the pristine parent.
#   1. reference error of the attention block for the stock arm and orin-refine1 (same test binary);
#   2. gate_sdpa.sh s2-c1; when it ends GATE_REJECTED at verify-check, the remaining steps run for evidence
#      (gate_rest.sh: session, traces), the verdict stays;
#   3. candidate 2: bit comparison of the kernel's output with the shipped kernel's and production-diff on the 3B
#      and 8B shapes, then gate.sh s3-c2.
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
E1="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine1"
step ./sdpa_err.sh sdpa-error1 topic4 stock "refine1=${E1// /,}"
step ./stage.sh s2-c1 parent "" topic4 "$E1" "candidate 1: orin-refine1 (SDPA prefill kernels) against the pristine parent"
step ./gate_sdpa.sh s2-c1 "$E1"
grep -q '^GATE_REJECTED .* step verify-check failed' $A/stage/s2-c1/gate.done && step ./gate_rest.sh s2-c1
E2="ET_VK_SARC_DEV_PROFILE=orin-lin-refine2"
step ./bitcmp.sh bit-bf topic4 8da4w orin_bf_t128x128k64g24s32mk32ra orin_bf_t128x128k64g42s32mk32ra
step ./pdiff.sh pdiff-bf2 topic4 8da4w 1b,3b,8b orin_bf_t128x128k64g24s32mk32ra
step ./stage.sh s3-c2 parent "" topic4 "$E2" "candidate 2: orin-lin-refine2 (8da4w whole-texel weight staging) against the pristine parent"
step ./gate.sh s3-c2 "$E2"
echo CHAIN_DONE
