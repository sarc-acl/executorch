#!/bin/bash
# q-c2.sh (GPU host, via rstart): candidate 2 (fused attention kernel fused3sb, the rk variants picked by the fused-variant screen; build c2)
# against candidate 1 (parent binary, ET_VK_SARC_780M_PROFILE=c7), after q-screens: gate (timed session, SDPA tiers 12 passes each, verify.sh
# against the snapshot, traces), SDPA output / reference-error evidence (D3.1), real-text logits probe (D3.2 / D3.3), and a kernel-level
# comparison of fused3sb against fused3 for the same variants (M2a is a barrier change, not meant to cost time).
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/q-c2.status; }
st "waiting for q-screens"; until grep -q Q_SCREENS_DONE $A/logs/q-screens.status 2>/dev/null; do sleep 30; done
C1="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7"
FU="ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rk,fused3sb_d128_t16x64g11s32rk"
st "c2 gate"; $T/gate.sh c2-fused "$C1 $FU" sdpa; st "c2 gate done"
$T/sdpa_evidence.sh c2-fused "all extended peaked full" > $A/stage/c2-fused/sdpa-evidence.out 2>&1; st "c2 sdpa evidence done: $(head -1 $A/stage/c2-fused/sdpa-evidence.out)"
$T/probe_run.sh c2-fused > $A/stage/c2-fused/probe.out 2>&1; st "c2 probe done: $(tail -1 $A/stage/c2-fused/probe.out)"
$T/fused_screen.sh c2-fused 3 $A/stage/c2-fused/fused-sb-screen.csv "$C1" \
  fused3_d64_t32x32g11s32rk,fused3_d128_t16x64g11s32rk fused3sb_d64_t32x32g11s32rk,fused3sb_d128_t16x64g11s32rk > $A/logs/fused-sb-screen.out 2>&1
st "fused3 vs fused3sb screen done"; st Q_C2_DONE
