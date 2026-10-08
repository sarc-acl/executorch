#!/bin/bash
# q-screens.sh (GPU host, via rstart): after q-c1: (1) a guard, one correctness pass (tier all) of the incumbent fused attention pair on
# this driver (the first time those kernels run here; a failure stops the queue); (2) the kernel-level screen of the fused variants (3 rounds);
# (3) the complete linear screens, 4w and 8da4w (3 rounds, the table kernel and the same candidate lists the RX 7600 screened).
# All on the binaries of stage c1-softmax (parent build): kernel timings, not timed sessions. Resumable (every screen skips finished rows).
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/q-screens.status; }
st "waiting for q-c1"; until grep -q Q_C1_DONE $A/logs/q-c1.status 2>/dev/null; do sleep 30; done
S=c1-softmax; D=$A/stage/$S; BASE="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7"
F0=fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko
if [[ ! -f $D/fused-guard.log ]]; then
  env $BASE ET_VK_SARC_780M_SDPA_FUSED=$F0 $T/gl.sh $D/test_llama_microbench --sdpa-correctness-only --sdpa-tier=all > $D/fused-guard.log 2>&1; rc=$?
  p=$(grep -c PASSED $D/fused-guard.log); f=$(grep -c FAILED $D/fused-guard.log); k=$(grep 'sdpa-kernels' $D/fused-guard.log | grep -c 'fused=sarc_dev_780m_sdpa_fused3')
  st "fused guard rc=$rc passed=$p failed=$f fused_kernel_lines=$k"
  { (( rc == 0 && f == 0 && p > 0 && k > 0 )); } || { st "STOP: fused guard did not pass (see fused-guard.log)"; exit 1; }
fi
echo "ET_VK_SARC_UNVERIFIED=1" > $D/screen.env
st "fused screen"
$T/fused_screen.sh $S 3 $D/fused-screen.csv "$BASE" \
  $F0 fused3_d64_t16x32g11s32rko,fused3_d128_t16x32g11s32rko fused3_d64_t32x64g11s32rko,fused3_d128_t32x32g11s32rko \
  fused3_d64_t32x32g11s32rk,fused3_d128_t16x16g11s32rko fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rk > $A/logs/fused-screen.out 2>&1
st "fused screen done"
P=sarc_dev_780m_x_linear_q4gsw_coopmat_; Q=sarc_linear_q4gsw_coopmat_sweep_; X=sarc_dev_linear_q4gsw_coopmat_bx_
K4=(table ${P}t128x128k32g24s32f32cbt ${P}t128x256k32g42s32f32c ${P}t128x256k32g42s32f32cbt ${P}t256x128k32g18s32f32bbt ${P}t256x128k32g18s32f32cbt
  ${P}t256x128k32g24s32f32bbt ${P}t256x128k32g24s32f32cbt ${P}t256x128k32g28s32f32bbt ${P}t256x128k32g28s32f32cbt ${X}t128x128k32g42s32f32c ${X}t128x256k32g42s32f32c
  ${Q}t128x128k32g22s32f32c ${Q}t128x128k32g24s32f32c ${Q}t128x128k32g42s32f32c ${Q}t128x128k32g42s32f32cbt ${Q}t128x256k32g42s32f32c ${Q}t256x128k32g24s32f32c
  ${Q}t256x128k32g42s32f32c ${Q}t256x128k32g44s32f32c ${Q}t64x256k32g41s32f32c)
st "linear screen 4w (${#K4[@]} kernels)"; $T/linear_screen.sh $S 4w 3 $D/screen-4w.csv $D/screen.env "${K4[@]}" > $A/logs/linear-screen-4w.out 2>&1; st "linear screen 4w done"
R=sarc_dev_780m_x_linear_dq8ca_coopmat_; Z=sarc_dev_linear_dq8ca_coopmat_zpg_bt_; W=sarc_linear_dq8ca_coopmat_zpg_sweep_
K8=(table ${R}zpg_bt_t128x64k32g22s32afmb2 ${R}zpg_t256x64k64g48s32afmb1 ${Z}t128x128k32g42s32 ${Z}t128x64k32g22s32 ${Z}t128x64k64g22s32 ${Z}t128x64k64g42s32
  ${W}t128x128k32g22s32 ${W}t128x128k32g24s32 ${W}t128x128k32g42s32 ${W}t128x32k32g22s32 ${W}t128x64k32g12s32 ${W}t128x64k32g21s32 ${W}t128x64k32g22s32
  ${W}t128x64k32g24s32 ${W}t128x64k64g22s32 ${W}t128x64k64g42s32 ${W}t256x32k32g14s32 ${W}t256x64k32g24s32 ${W}t256x64k32g42s32 ${W}t64x128k32g22s32
  ${W}t64x128k32g41s32 ${W}t64x64k32g21s32 ${W}t64x64k32g22s32)
st "linear screen 8da4w (${#K8[@]} kernels)"; $T/linear_screen.sh $S 8da4w 3 $D/screen-8da4w.csv $D/screen.env "${K8[@]}" > $A/logs/linear-screen-8da4w.out 2>&1; st "linear screen 8da4w done"
st Q_SCREENS_DONE
