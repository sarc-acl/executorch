#!/bin/bash
# chain4.sh: device side, after chain3. The Orin whole-texel 8da4w twins (build topic2): production-diff on the
# 1B shapes for every tile (a tile that fails is not screened further), then a 2-round kernel-level screen against
# the shipped kernel and the half-texel twin.
cd "$(dirname "$0")"
while ! grep -q "^DONE\|^KILLED" ~/hmz-sarc-orin/jobs/${1:-chain3}.status; do sleep 20; done   # $1 = the job to wait for
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
step ./pdiff.sh pdiff-bf topic2 8da4w 1b base orin_bf_t128x128k64g44s32mk32ra orin_bf_t128x128k64g42s32mk32ra orin_bf_t128x128k64g24s32mk32ra orin_bf_t256x128k32g44s32mk32ra orin_bf1_t128x128k128g44s32mk32ra orin_bf1_t128x128k128g42s32mk32ra orin_bf1_t128x128k128g24s32mk32ra orin_bf1_t256x128k64g44s32mk32ra orin_bf1_t128x128k64g42s32mk32ra
step ./screen.sh screen2-8da4w topic2 8da4w 2 base orin_bf_t128x128k64g44s32mk32ra orin_bf_t128x128k64g42s32mk32ra orin_bf_t128x128k64g24s32mk32ra orin_bf_t256x128k32g44s32mk32ra orin_bf1_t128x128k128g44s32mk32ra orin_bf1_t128x128k128g42s32mk32ra orin_bf1_t128x128k128g24s32mk32ra orin_bf1_t256x128k64g44s32mk32ra orin_bf1_t128x128k64g42s32mk32ra 4070ti_bh_t128x128k64g44s32mk32ra 4070ti_bh_t128x128k64g42s32mk32ra
echo CHAIN_DONE
