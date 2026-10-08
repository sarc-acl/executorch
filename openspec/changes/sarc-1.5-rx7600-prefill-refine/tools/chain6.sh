#!/bin/bash
# chain6.sh: candidate 4 (whole-texel 8da4w staging everywhere, profile rx7600-refine3) and M2a (fused3sb: subgroupBarrier()
# after every memoryBarrierShared()), both from build c4 (commit 129cea7ac) against candidate 3 (build c3, profile
# rx7600-refine2), one gate each (timed session, SDPA tiers for M2a, verify.sh, traces) and the SDPA output evidence of M2a.
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/chain6.status; }
B="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7"
F=fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko; FSB=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko
C3="$B ET_VK_SARC_780M_SDPA_FUSED=$F ET_VK_SARC_RX7600_PROFILE=rx7600-refine2"
C4="$B ET_VK_SARC_780M_SDPA_FUSED=$F ET_VK_SARC_RX7600_PROFILE=rx7600-refine3"
M2="$B ET_VK_SARC_780M_SDPA_FUSED=$FSB ET_VK_SARC_RX7600_PROFILE=rx7600-refine2"
st "stage c4-texel"
$T/stage.sh c4-texel c3 "$C3" c4 "$C4" "candidate 4: whole-texel 8da4w staging everywhere (rx7600-refine3, build c4 = commit 129cea7ac) against candidate 3 (build c3, rx7600-refine2)" > $A/logs/stage-c4.out 2>&1
st "c4 gate"; $T/gate.sh c4-texel "$C4"; st "c4 gate done"
st "stage m2a-sgbarrier"
$T/stage.sh m2a-sgbarrier c3 "$C3" c4 "$M2" "M2a: fused3sb (subgroupBarrier after every memoryBarrierShared) on candidate 3 (rx7600-refine2), build c4 against build c3" > $A/logs/stage-m2a.out 2>&1
st "m2a gate"; $T/gate.sh m2a-sgbarrier "$M2" sdpa; st "m2a gate done"
$T/sdpa_evidence.sh m2a-sgbarrier "all extended peaked full" > $A/stage/m2a-sgbarrier/sdpa-evidence.out 2>&1; st "m2a sdpa evidence done: $(head -1 $A/stage/m2a-sgbarrier/sdpa-evidence.out)"
st CHAIN6_DONE
