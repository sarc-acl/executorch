#!/bin/bash
# chain5.sh: candidate 3 (linear kernel per layer shape, profile rx7600-refine2, build c3 of commit 6ebf39484) against
# candidate 2 (parent binary, env switches): timed session, verify.sh, traces (gate.sh; no SDPA tiers: no attention
# kernel changes). The stage c3-linear is made by stage.sh beforehand.
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/chain5.status; }
C3="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko ET_VK_SARC_RX7600_PROFILE=rx7600-refine2"
st "c3 gate"; $T/gate.sh c3-linear "$C3"; st "c3 gate done"; st CHAIN5_DONE
