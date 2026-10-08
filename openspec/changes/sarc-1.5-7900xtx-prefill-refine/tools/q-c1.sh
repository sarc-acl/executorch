#!/bin/bash
# q-c1.sh (GPU host, via rstart): candidate 1 (softmax r3 of the 780M through the softmax-variant hook, ET_VK_SARC_780M_PROFILE=c7, on the
# parent binary) after q-aa: gate (timed session against the parent, SDPA tiers, verify.sh against s0, traces), then the SDPA output evidence.
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/q-c1.status; }
st "waiting for q-aa"; until grep -q Q_AA_DONE $A/logs/q-aa.status 2>/dev/null; do sleep 30; done
C1="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7"
st "c1 gate"; $T/gate.sh c1-softmax "$C1" sdpa; st "c1 gate done"
$T/sdpa_evidence.sh c1-softmax > $A/stage/c1-softmax/sdpa-evidence.out 2>&1; st "c1 sdpa evidence done: $(head -1 $A/stage/c1-softmax/sdpa-evidence.out)"
st Q_C1_DONE
