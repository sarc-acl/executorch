#!/bin/bash
# q-set2b.sh (GPU host, via rstart): second set, item 2, the synchronisation variants of the 8da4w body (branch-free chunk loop, stores interleaved with the MMAs,
# ablation twins that remove work: measurement only) on the two incumbent K32 tiles, 3 rounds, on the microbench of stage s2b-screen (build c8). Analysis: tools/screen_pick2.py
# against the incumbents of the 7900xtx-refine5 profile (variant filter "bf"); the ablation ratios are read from the ratios file.
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/q-set2b.status; }
S=s2b-screen; D=$A/stage/$S; echo "ET_VK_SARC_UNVERIFIED=1" > $D/screen.env
KDQ=(
  table
  sarc_linear_dq8ca_coopmat_zpg_sweep_t256x64k32g24s32
  sarc_linear_dq8ca_coopmat_zpg_sweep_t128x64k32g22s32
  sarc_linear_dq8ca_coopmat_zpg_sweep_t64x64k32g22s32
  sarc_dev_780m_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32afmb1
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k32g24s32pa4pb4
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t128x64k32g42s32pa4pb4
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k32g24s32bf
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k32g24s32bfa0b0
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k32g24s32bfa0b1
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k32g24s32bfa1b1
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t128x64k32g42s32bf
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t128x64k32g42s32bfa0b0
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t128x64k32g42s32bfa0b1
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t128x64k32g42s32bfa1b1
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k32g24s32abl1
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k32g24s32abl2
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k32g24s32abl4
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k32g24s32abl7
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k32g24s32abl8
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k32g24s32abl16
  sarc_dev_7900xtx_x_linear_dq8ca_coopmat_zpg_t256x64k32g24s32abl23
)
st "linear screen 8da4w, synchronisation variants (${#KDQ[@]} kernels)"; $T/linear_screen.sh $S 8da4w 3 $D/screen3-8da4w.csv $D/screen.env "${KDQ[@]}" > $A/logs/linear-screen3-8da4w.out 2>&1
st "screen done"; st Q_SET2B_DONE
