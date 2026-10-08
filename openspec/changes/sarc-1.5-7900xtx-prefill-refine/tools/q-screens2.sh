#!/bin/bash
# q-screens2.sh (GPU host, via rstart): completes the 8da4w linear screen after q-c2 (the first pass of q-screens.sh lost every job after the 13th of
# round 1 to a foreign monitor: gl.sh refused to start while `amdgpu_top` of another session held the card, and the screen script skipped the job; gl.sh now waits).
# Resumable: finished (round, kernel) rows of screen-8da4w.csv are kept.
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/q-screens2.status; }
st "waiting for q-c2"; until grep -q Q_C2_DONE $A/logs/q-c2.status 2>/dev/null; do sleep 30; done
S=c1-softmax; D=$A/stage/$S
R=sarc_dev_780m_x_linear_dq8ca_coopmat_; Z=sarc_dev_linear_dq8ca_coopmat_zpg_bt_; W=sarc_linear_dq8ca_coopmat_zpg_sweep_
K8=(table ${R}zpg_bt_t128x64k32g22s32afmb2 ${R}zpg_t256x64k64g48s32afmb1 ${Z}t128x128k32g42s32 ${Z}t128x64k32g22s32 ${Z}t128x64k64g22s32 ${Z}t128x64k64g42s32
  ${W}t128x128k32g22s32 ${W}t128x128k32g24s32 ${W}t128x128k32g42s32 ${W}t128x32k32g22s32 ${W}t128x64k32g12s32 ${W}t128x64k32g21s32 ${W}t128x64k32g22s32
  ${W}t128x64k32g24s32 ${W}t128x64k64g22s32 ${W}t128x64k64g42s32 ${W}t256x32k32g14s32 ${W}t256x64k32g24s32 ${W}t256x64k32g42s32 ${W}t64x128k32g22s32
  ${W}t64x128k32g41s32 ${W}t64x64k32g21s32 ${W}t64x64k32g22s32)
st "linear screen 8da4w (${#K8[@]} kernels), rest"; $T/linear_screen.sh $S 8da4w 3 $D/screen-8da4w.csv $D/screen.env "${K8[@]}" > $A/logs/linear-screen-8da4w-2.out 2>&1; st "linear screen 8da4w done"
st Q_SCREENS2_DONE
