#!/bin/bash
# chain3.sh: device side, after chain2b. Kernel-level screen of the existing 8da4w dev tiles on the Orin (the
# 4070 Ti campaign's zpgtr sweep tiles and its half-texel weight staging `bh`), 1 round, then the stock arm of
# the SDPA reference error (the parent's arithmetic with the topic test binary).
cd "$(dirname "$0")"
while ! grep -q "^DONE\|^KILLED" ~/hmz-sarc-orin/jobs/chain2b.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
step ./screen.sh screen1-8da4w topic1 8da4w 1 base 4070ti_bh_t128x128k64g44s32mk32ra 4070ti_bh_t128x128k64g42s32mk32ra 4070ti_bh_t128x128k64g24s32mk32ra \
  4070ti_bh_t256x128k32g42s32mk32ra 4070ti_t128x128k64g42s32mk32ra 4070ti_t128x128k64g24s32mk32ra 4070ti_t256x128k32g44s32mk32ra \
  4070ti_t256x128k32g42s32mk32ra 4070ti_t128x256k32g81s32mk32ra 4070ti_t128x128k32g42s32mk32ra 4070ti_t128x64k64g44s32mk32ra \
  4070ti_t128x64k64g42s32mk32ra 4070ti_t64x128k64g42s32mk32ra 4070ti_t256x64k64g44s32mk32ra 4070ti_t256x64k64g42s32mk32ra \
  4070ti_t256x128k32g24s32mk32ra 4070ti_t128x128k64g22s32mk32ra
step ./sdpa_err.sh sdpa-error topic1 stock
echo CHAIN_DONE
