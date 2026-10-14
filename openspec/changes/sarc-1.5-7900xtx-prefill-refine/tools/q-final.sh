#!/bin/bash
# q-final.sh (GPU host, via rstart): the final verification (R11) of the final stack on the build `final` of the committed head (no local patch), against the
# pristine parent (build `parent`, ET_VK_SARC_UNVERIFIED=1 only): gate (one timed session, SDPA tiers 12 passes each, verify.sh against the snapshot
# s0-parent-verify, ETDump traces), SDPA output / reference-error evidence (D3.1), real-text logits probe (D3.2 / D3.3). The final environment is
# read from <gpu-root>/final-env.txt (written by the actor after candidate 6 is decided).
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/q-final.status; }
st "waiting for the final environment"; until [[ -s $A/final-env.txt ]]; do sleep 30; done
F=$(cat $A/final-env.txt)
st "final gate, env [$F]"; $T/gate.sh final "$F" sdpa; st "final gate done"
$T/sdpa_evidence.sh final "all extended peaked full" > $A/stage/final/sdpa-evidence.out 2>&1; st "final sdpa evidence done: $(head -1 $A/stage/final/sdpa-evidence.out)"
$T/probe_run.sh final > $A/stage/final/probe.out 2>&1; st "final probe done: $(tail -1 $A/stage/final/probe.out)"
st Q_FINAL_DONE
