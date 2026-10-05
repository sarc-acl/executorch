#!/bin/bash
# chain8.sh [job to wait for]: device side. Candidate 1h = orin-refine1 with the fp32 softmax without the zero
# tail (ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf), build topic6 (the committed branch: the softmax-name hook is the
# one release-zone edit the owner accepted on 2026-10-05), against the pristine parent.
#   1. reference error of the attention block: stock, orin-refine1 and orin-refine1 with each softmax variant,
#      all with topic6's test binary (raw/sdpa-error2);
#   2. the full SDPA gate (gate_sdpa.sh s4-c1h). When it ends GATE_REJECTED at verify-check, the remaining steps
#      run for evidence (gate_rest.sh); the verdict then depends on the reference-error rule (logits probe).
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
E="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine1 ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf"
E1=ET_VK_SARC_UNVERIFIED=1,ET_VK_SARC_DEV_PROFILE=orin-refine1
step ./sdpa_err.sh sdpa-error2 topic6 stock "refine1=$E1" "refine1-f32=$E1,ET_VK_SARC_SOFTMAX_VARIANT=4070ti_f32" "refine1-nz=$E1,ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nz" "refine1-nzf=$E1,ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf"
step ./stage.sh s4-c1h parent "" topic6 "$E" "candidate 1h: orin-refine1 + fp32 softmax without the zero tail (softmax-name hook, owner decision 2026-10-05), against the pristine parent"
step ./gate_sdpa.sh s4-c1h "$E"
grep -q '^GATE_REJECTED .* step verify-check failed' $A/stage/s4-c1h/gate.done && step ./gate_rest.sh s4-c1h
echo CHAIN_DONE
