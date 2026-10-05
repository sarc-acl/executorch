#!/bin/bash
# chain12.sh [job to wait for]: device side. 4w screen 2 (build topic8): the staging variants of the shipped Orin
# tiles (gen_orin_q4.py) against the Orin rows, 2 rounds. Then the pre-checks of the 4w tile of orin-lin-refine3
# (texel-wise weight staging on the fp32 shape): bit comparison with the shipped kernels and production-diff on 8B.
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
step ./screen.sh screen2-4w topic8 4w 2 base orin_t256x128k16g22s32 orin_t256x128k16g22s32bt orin_t256x128k16g22s32c orin_t256x128k16g22s32cbt orin_t128x128k16g22s32bt orin_t256x128k16g42s32 orin_t256x128k16g42s32bt orin_t256x128k16g24s32 orin_t256x128k16g24s32bt orin_t128x128k32g42s32f32bt orin_t256x128k16g44s32bt orin_t128x128k32g22s32bt
step ./bitcmp.sh bit-q4 topic8 4w bx_t128x128k32g42s32f32c t128x128k32g42s32f32cbt orin_t256x128k16g22s32 orin_t256x128k16g22s32bt orin_t256x128k16g22s32cbt orin_t128x128k32g42s32f32bt
step ./pdiff.sh pdiff-q4 topic8 4w 8b base bx_t128x128k32g42s32f32c
echo CHAIN_DONE
