#!/bin/bash
# q-set2c.sh (GPU host, via rstart): second set, item 3: the ablation twins of the 4w table kernel (no barrier, no next-chunk stores, neither; measurement only) against the
# table kernel and the pitch control, 3 rounds, 4w, on the microbench of stage s2c-screen (build c9). They bound what any cut of the 4w per-K-step synchronisation could gain.
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/q-set2c.status; }
S=s2c-screen; D=$A/stage/$S; echo "ET_VK_SARC_UNVERIFIED=1" > $D/screen.env; P=sarc_dev_7900xtx_x_linear_q4gsw_coopmat_t256x128k32g24s32f32cbtbp8
KQ4=(table sarc_linear_q4gsw_coopmat_sweep_t128x128k32g42s32f32cbt ${P} ${P}abl4 ${P}abl16 ${P}abl20)
st "linear screen 4w, ablation twins (${#KQ4[@]} kernels)"; $T/linear_screen.sh $S 4w 3 $D/screen3-4w.csv $D/screen.env "${KQ4[@]}" > $A/logs/linear-screen3-4w.out 2>&1
st "screen done"; st Q_SET2C_DONE
