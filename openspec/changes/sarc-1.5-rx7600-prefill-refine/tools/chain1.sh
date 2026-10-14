#!/bin/bash
# chain1.sh: after chain0: logits_probe build for the parent build; candidate 1 (softmax r3, ET_VK_SARC_780M_PROFILE=c7, on
# the parent binary) gate + bit-identity evidence; then the kernel-level screen of the fused attention variants.
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/chain1.status; }
st "waiting for chain0"; until grep -q CHAIN0_DONE $A/logs/chain0.status 2>/dev/null; do sleep 60; done
st "waiting for the A/A calibration in thresholds.txt"; until grep -q "^calibrated=" $T/thresholds.txt; do sleep 60; done
st "probe build"; touch $A/.building; $T/hold.sh run "build probe parent" $T/build_probe.sh parent > $A/build/rx7600/parent-probe.log 2>&1; st "probe build rc=$?"; rm -f $A/.building
P="ET_VK_SARC_UNVERIFIED=1"; C1="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7"
$T/stage.sh c1-softmax parent "$P" parent "$C1" "candidate 1: softmax r3 of the 780M (ET_VK_SARC_780M_PROFILE=c7) on the parent binary" > $A/logs/stage-c1.out 2>&1
cp -f $A/build/rx7600/parent/probe/logits_probe $A/stage/c1-softmax/lp
st "c1 gate"; $T/gate.sh c1-softmax "$C1" sdpa; st "c1 gate done"
$T/sdpa_evidence.sh c1-softmax > $A/stage/c1-softmax/sdpa-evidence.out 2>&1; st "c1 sdpa evidence done: $(head -1 $A/stage/c1-softmax/sdpa-evidence.out)"
F="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7"
$T/fused_screen.sh c1-softmax 3 $A/stage/c1-softmax/fused-screen.csv "$F" \
  fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko fused3_d64_t16x32g11s32rko,fused3_d128_t16x32g11s32rko \
  fused3_d64_t32x64g11s32rko,fused3_d128_t32x32g11s32rko fused3_d64_t32x32g11s32rk,fused3_d128_t16x16g11s32rko \
  fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rk > $A/logs/fused-screen.out 2>&1
st "fused screen done"; st CHAIN1_DONE
