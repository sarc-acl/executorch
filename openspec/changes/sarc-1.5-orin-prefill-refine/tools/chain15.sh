#!/bin/bash
# chain15.sh [job to wait for]: device side, both Orins. Owner decision 2026-10-05 19:10 UTC: retest of the one
# configuration of the agreement batch that was outside the threshold (4070ti_t256x256k16g44s32gac, ratio 0.80,
# one round per device), 3 rounds on each device, with `base` beside it for scale. Same build (topic5) and
# invocation as the batch. The first batch and its DISAGREE verdict stay as recorded.
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
step ./screen.sh agree-retest topic5 4w 3 base 4070ti_t256x256k16g44s32gac
echo CHAIN_DONE
