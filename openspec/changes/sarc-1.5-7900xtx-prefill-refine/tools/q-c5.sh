#!/bin/bash
# q-c5.sh (GPU host, via rstart): candidate 5 (unfused attention kernels of the screen: QK^T pk_t128x128k32g42s32nf, attn*V sweep t64x64k32g42s32; profile
# 7900xtx-refine4, build c5) against candidate 3 (profile 7900xtx-refine2, the same build): gate (timed session, SDPA tiers 12 passes each,
# verify.sh against the snapshot, traces), SDPA output / reference-error evidence (D3.1), real-text logits probe (D3.2 / D3.3).
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/q-c5.status; }
C5="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_7900XTX_PROFILE=7900xtx-refine4"
st "c5 gate"; $T/gate.sh c5-sdpa "$C5" sdpa; st "c5 gate done"
$T/sdpa_evidence.sh c5-sdpa "all extended peaked full" > $A/stage/c5-sdpa/sdpa-evidence.out 2>&1; st "c5 sdpa evidence done: $(head -1 $A/stage/c5-sdpa/sdpa-evidence.out)"
$T/probe_run.sh c5-sdpa > $A/stage/c5-sdpa/probe.out 2>&1; st "c5 probe done: $(tail -1 $A/stage/c5-sdpa/probe.out)"
st Q_C5_DONE
