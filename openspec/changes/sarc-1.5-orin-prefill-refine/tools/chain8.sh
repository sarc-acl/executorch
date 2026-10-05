#!/bin/bash
# chain8.sh [job to wait for]: device side. Candidate 1h = orin-refine1 with the fp32 softmax without the zero
# tail (ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf), through build hook4 (topic4 + the softmax-name hook, a local patch
# that is NOT committed), against the pristine parent: the full SDPA gate. Measured for the owner: the branch
# cannot reach this softmax. When the gate ends GATE_REJECTED at verify-check, the remaining steps run for
# evidence (gate_rest.sh).
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
E="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine1 ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf"
step ./stage.sh s4-c1h parent "" hook4 "$E" "candidate 1h: orin-refine1 + fp32 softmax without the zero tail, build hook4 (local softmax-name hook, NOT committed), against the pristine parent"
step ./gate_sdpa.sh s4-c1h "$E"
grep -q '^GATE_REJECTED .* step verify-check failed' $A/stage/s4-c1h/gate.done && step ./gate_rest.sh s4-c1h
echo CHAIN_DONE
