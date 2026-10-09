#!/bin/bash
# q-set2.sh (GPU host, via rstart): second set, items 2 and 3 (8da4w and 4w linear, per-K-step synchronisation / staging row pitch), on the microbench of
# stage s2-screen (build c7): (1) phase timing of the twins (the release 4w table kernel's twin and the 7900xtx 8da4w twins: control pitch and 24-byte A pitch per tile),
# (2) the kernel-level screens, 3 rounds, of the variants against the incumbents of the 7900xtx-refine5 profile (analysis: tools/screen_pick2.py). Resumable.
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/q-set2.status; }
S=s2-screen; D=$A/stage/$S; echo "ET_VK_SARC_UNVERIFIED=1" > $D/screen.env
st "phase timing"
$T/phases.sh $S q4-table 4w sarc_dev_prof_q4gsw_t256x128k32g24s32f32cbtp 256 128
$T/phases.sh $S dq-t256x64k32g24s32pa4pb4p 8da4w sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k32g24s32pa4pb4p 256 64
$T/phases.sh $S dq-t256x64k32g24s32pa6pb4p 8da4w sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k32g24s32pa6pb4p 256 64
$T/phases.sh $S dq-t128x64k32g22s32pa4pb4p 8da4w sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t128x64k32g22s32pa4pb4p 128 64
$T/phases.sh $S dq-t128x64k32g22s32pa6pb4p 8da4w sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t128x64k32g22s32pa6pb4p 128 64
$T/phases.sh $S dq-t128x64k32g42s32pa4pb4p 8da4w sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t128x64k32g42s32pa4pb4p 128 64
$T/phases.sh $S dq-t128x64k32g42s32pa6pb4p 8da4w sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t128x64k32g42s32pa6pb4p 128 64
$T/phases.sh $S dq-t256x64k64g48s32pa4pb4cshap 8da4w sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32pa4pb4cshap 256 64
$T/phases.sh $S dq-t256x64k64g48s32pa6pb4cshap 8da4w sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32pa6pb4cshap 256 64
st "phase timing done"
KDQ=(
  table
  sarc_linear_dq8ca_coopmat_zpg_sweep_t256x64k32g24s32
  sarc_linear_dq8ca_coopmat_zpg_sweep_t128x64k32g22s32
  sarc_linear_dq8ca_coopmat_zpg_sweep_t64x64k32g22s32
  sarc_dev_780m_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32afmb1
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k32g24s32pa4pb4
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k32g24s32pa5pb4
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k32g24s32pa6pb4
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k32g24s32pa8pb4
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k32g24s32pa6pb6
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t128x64k32g22s32pa4pb4
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t128x64k32g22s32pa5pb4
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t128x64k32g22s32pa6pb4
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t128x64k32g22s32pa8pb4
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t128x64k32g22s32pa6pb6
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t128x64k32g42s32pa4pb4
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t128x64k32g42s32pa5pb4
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t128x64k32g42s32pa6pb4
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t128x64k32g42s32pa8pb4
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t128x64k32g42s32pa6pb6
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32pa4pb4csha
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32pa5pb4csha
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32pa6pb4csha
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32pa6pb6csha
)
KQ4=(
  table
  sarc_linear_q4gsw_coopmat_sweep_t128x128k32g42s32f32cbt
  sarc_dev_7900xtx_x_linear_q4gsw_coopmat_t256x128k32g24s32f32cbtbp8
  sarc_dev_7900xtx_x_linear_q4gsw_coopmat_t256x128k32g24s32f32cbtbp4
  sarc_dev_7900xtx_x_linear_q4gsw_coopmat_t256x128k32g24s32f32cbtbp12
  sarc_dev_7900xtx_x_linear_q4gsw_coopmat_t256x128k32g24s32f32cbtap8bp8
  sarc_dev_7900xtx_x_linear_q4gsw_coopmat_t256x128k32g24s32f32cbtap4bp8
  sarc_dev_7900xtx_x_linear_q4gsw_coopmat_t256x128k32g24s32f32cbtap4bp4
  sarc_dev_7900xtx_x_linear_q4gsw_coopmat_t256x128k32g24s32f32cbtap4bp12
  sarc_dev_7900xtx_x_linear_q4gsw_coopmat_t256x128k32g24s32f32cbtap12bp4
  sarc_dev_7900xtx_x_linear_q4gsw_coopmat_t256x128k32g24s32f32cbtap8bp12
  sarc_dev_7900xtx_x_linear_q4gsw_coopmat_t256x128k32g28s32f32cbtbp8
  sarc_dev_7900xtx_x_linear_q4gsw_coopmat_t256x128k32g28s32f32cbtbp4
  sarc_dev_7900xtx_x_linear_q4gsw_coopmat_t256x128k32g28s32f32cbtbp12
  sarc_dev_7900xtx_x_linear_q4gsw_coopmat_t256x128k32g28s32f32cbtap8bp8
  sarc_dev_7900xtx_x_linear_q4gsw_coopmat_t256x128k32g28s32f32cbtap4bp8
  sarc_dev_7900xtx_x_linear_q4gsw_coopmat_t256x128k32g28s32f32cbtap4bp4
  sarc_dev_7900xtx_x_linear_q4gsw_coopmat_t256x128k32g28s32f32cbtap4bp12
  sarc_dev_7900xtx_x_linear_q4gsw_coopmat_t256x128k32g28s32f32cbtap12bp4
  sarc_dev_7900xtx_x_linear_q4gsw_coopmat_t256x128k32g28s32f32cbtap8bp12
)
st "linear screen 8da4w (${#KDQ[@]} kernels)"; $T/linear_screen.sh $S 8da4w 3 $D/screen2-8da4w.csv $D/screen.env "${KDQ[@]}" > $A/logs/linear-screen2-8da4w.out 2>&1; st "8da4w screen done"
st "linear screen 4w (${#KQ4[@]} kernels)"; $T/linear_screen.sh $S 4w 3 $D/screen2-4w.csv $D/screen.env "${KQ4[@]}" > $A/logs/linear-screen2-4w.out 2>&1; st "4w screen done"
st Q_SET2_DONE
