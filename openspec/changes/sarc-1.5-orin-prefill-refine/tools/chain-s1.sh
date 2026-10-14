#!/bin/bash
# chain-s1.sh: SECOND Orin (duck-stable), screening only (owner offer 2026-10-05). Nothing measured here is
# reported as a result: the primary device confirms whatever this ranks first.
#   1. agreement batch: the 26 configurations of 4w screen 1 (build topic5, 1 round), the batch duck-naughty
#      measured at 10:10 to 10:40 UTC. Threshold, fixed before this ran: Spearman rank correlation of the
#      per-configuration geomean kernel time >= 0.95 and every per-configuration time ratio within 0.95 to 1.05
#      (tools/agree.py). Below it the second device is not used.
#   2. 4w screen 2 (build topic8): the staging variants of the shipped Orin 4w tiles, 2 rounds.
# Before each step: the LLM service of that device must still be inactive and 4 GB of memory available.
cd "$(dirname "$0")"; source ./common.sh
[[ $(hostname) == duck-stable ]] || { echo "duck-stable only" >&2; exit 2; }
guard() { [[ $(systemctl is-active llama-gemma.service 2>/dev/null) != active && $(mem_avail_mb) -ge 4000 ]] || { echo "CHAIN_STOPPED: llama-gemma.service is back or memory is short ($(mem_avail_mb) MB): not using this device"; exit 3; }; }
step() { guard; echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
step ./screen.sh screen1-4w topic5 4w 1 base t128x128k16g22s32 t128x128k32g42s32 t128x128k32g24s32 t128x128k32g42s32f32 t128x128k32g42s32f32c t128x128k32g24s32f32c    t256x128k32g24s32f32c t256x128k32g44s32f32c t128x256k32g42s32f32c t128x128k32g42s32f32cbt t128x128k32g22s32f32c t256x128k32g42s32f32c t64x256k32g41s32f32c    bx_t128x256k32g42s32f32c bx_t128x128k32g42s32f32c 4070ti_t256x128k16g42s32gac 4070ti_t256x128k16g24s32gac 4070ti_t256x256k16g44s32gac 4070ti_t128x256k16g42s32gac    4070ti_t128x128k32g42s32gac 4070ti_t128x128k32g44s32gac 4070ti_t128x128k32g24s32gac 4070ti_t128x128k16g42s32gac 4070ti_t256x128k16g42s32gacbt 4070ti_t128x128k32g42s32gacbt 
step ./screen.sh screen2-4w topic8 4w 2 base orin_t256x128k16g22s32 orin_t256x128k16g22s32bt orin_t256x128k16g22s32c orin_t256x128k16g22s32cbt orin_t128x128k16g22s32bt orin_t256x128k16g42s32 orin_t256x128k16g42s32bt orin_t256x128k16g24s32 orin_t256x128k16g24s32bt orin_t128x128k32g42s32f32bt orin_t256x128k16g44s32bt orin_t128x128k32g22s32bt
echo CHAIN_DONE
