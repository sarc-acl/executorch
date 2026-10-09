#!/bin/bash
# q-c6.sh (GPU host, via rstart): candidate 6 (profile 7900xtx-refine5: candidate 5 plus attn*V sweep_t32x32k32g22s32 for head dimension 64) against
# candidate 5 (profile 7900xtx-refine4), the same build `final`: gate (timed session, SDPA tiers 12 passes each, verify.sh against the snapshot,
# traces), SDPA output / reference-error evidence, real-text logits probe. Adoption rule: proposal.md "Adoption rule for candidate 6".
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/q-c6.status; }
C6="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_7900XTX_PROFILE=7900xtx-refine5"
st "c6 gate"; $T/gate.sh c6-attn "$C6" sdpa; st "c6 gate done"
$T/sdpa_evidence.sh c6-attn "all extended peaked full" > $A/stage/c6-attn/sdpa-evidence.out 2>&1; st "c6 sdpa evidence done: $(head -1 $A/stage/c6-attn/sdpa-evidence.out)"
st Q_C6_DONE
