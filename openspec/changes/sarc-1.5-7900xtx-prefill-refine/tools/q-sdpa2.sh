#!/bin/bash
# q-sdpa2.sh (GPU host, via rstart): the second attention screen (3 rounds, exact names) of the variants added for it, against the incumbents of
# candidate 5 (QK^T pk_t128x128k32g42s32nf, attn*V sweep_t64x64k32g42s32), on the microbench of stage c6-screen (build c6b); after q-roof.
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/q-sdpa2.status; }
st "waiting for q-roof"; until grep -q Q_ROOF_DONE $A/logs/q-roof.status 2>/dev/null; do sleep 30; done
D=$A/stage/c6-screen; echo "ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7" > $D/sdpa-screen2.env
QI=sarc_sdpa_qk_coopmat_pk_t128x128k32g42s32nf; AI=sarc_sdpa_av_coopmat_sweep_t64x64k32g42s32; Q=sarc_sdpa_qk_coopmat_pk_; V=sarc_sdpa_av_coopmat_sweep_
PS=(table qk-${Q}t128x128k32g24s32nf qk-${Q}t128x128k32g22s64nf qk-${Q}t128x128k32g44s32nf qk-${Q}t128x128k32g22s32nf qk-${Q}t64x128k32g42s32nf
  av-${V}t128x128k32g44s32 av-${V}t128x128k32g82s32 av-${V}t128x128k32g42s64 av-${V}t32x32k32g22s32)
st "sdpa screen 2 (${#PS[@]} items)"; $T/sdpa_screen2.sh c6-screen 3 $D/sdpa-screen2.csv $D/sdpa-screen2.env $QI $AI "${PS[@]}" > $A/logs/sdpa-screen2.out 2>&1
st "sdpa screen 2 done"; st Q_SDPA2_DONE
