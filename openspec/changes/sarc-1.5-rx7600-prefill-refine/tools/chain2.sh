#!/bin/bash
# chain2.sh: after chain1: kernel-level screens of the 4w and 8da4w linear kernels on the twelve real shapes (port
# items 3 and 4: the kernel per layer shape, the whole-texel 8da4w staging), 3 rounds, parent microbench binary.
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/chain2.status; }
st "waiting for chain1"; until grep -q CHAIN1_DONE $A/logs/chain1.status 2>/dev/null; do sleep 60; done
S=$A/stage/screens; mkdir -p $S; cp -f $A/build/rx7600/parent/tests/test_llama_microbench $S/
{ echo "kernel screens, parent build test_llama_microbench, env ET_VK_SARC_UNVERIFIED=1"; sha256sum $S/test_llama_microbench; } > $S/STAGE.md
Q=sarc_linear_q4gsw_coopmat_sweep_; X=sarc_dev_780m_x_linear_q4gsw_coopmat_
st "4w screen"
$T/linear_screen.sh screens 4w 3 $S/screen-4w.csv "ET_VK_SARC_UNVERIFIED=1" table \
  ${Q}t128x128k32g22s32f32c ${Q}t128x128k32g24s32f32c ${Q}t128x128k32g42s32f32c ${Q}t128x128k32g42s32f32cbt ${Q}t128x256k32g42s32f32c \
  ${Q}t256x128k32g24s32f32c ${Q}t256x128k32g42s32f32c ${Q}t256x128k32g44s32f32c ${Q}t64x256k32g41s32f32c \
  sarc_dev_linear_q4gsw_coopmat_bx_t128x128k32g42s32f32c sarc_dev_linear_q4gsw_coopmat_bx_t128x256k32g42s32f32c \
  ${X}t128x128k32g24s32f32cbt ${X}t128x256k32g42s32f32c ${X}t128x256k32g42s32f32cbt ${X}t256x128k32g18s32f32bbt ${X}t256x128k32g18s32f32cbt \
  ${X}t256x128k32g24s32f32bbt ${X}t256x128k32g24s32f32cbt ${X}t256x128k32g28s32f32bbt ${X}t256x128k32g28s32f32cbt > $A/logs/screen-4w.out 2>&1
D=sarc_linear_dq8ca_coopmat_zpg_sweep_; B=sarc_dev_linear_dq8ca_coopmat_zpg_bt_
st "8da4w screen"
$T/linear_screen.sh screens 8da4w 3 $S/screen-8da4w.csv "ET_VK_SARC_UNVERIFIED=1" table \
  ${D}t128x128k32g22s32 ${D}t128x128k32g24s32 ${D}t128x128k32g42s32 ${D}t128x32k32g22s32 ${D}t128x64k32g12s32 ${D}t128x64k32g21s32 \
  ${D}t128x64k32g22s32 ${D}t128x64k32g24s32 ${D}t128x64k64g22s32 ${D}t128x64k64g42s32 ${D}t256x32k32g14s32 ${D}t256x64k32g24s32 \
  ${D}t256x64k32g42s32 ${D}t64x128k32g22s32 ${D}t64x128k32g41s32 ${D}t64x64k32g21s32 ${D}t64x64k32g22s32 \
  ${B}t128x128k32g42s32 ${B}t128x64k32g22s32 ${B}t128x64k64g22s32 ${B}t128x64k64g42s32 \
  sarc_dev_780m_x_linear_dq8ca_coopmat_zpg_bt_t128x64k32g22s32afmb2 sarc_dev_780m_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32afmb1 > $A/logs/screen-8da4w.out 2>&1
st CHAIN2_DONE
