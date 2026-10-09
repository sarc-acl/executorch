#!/bin/bash
# chain8.sh <session> <build tag> <profile> <note>: round 2 gate of one candidate whose only change against round 1's final stack is the
# linear kernel selection of <profile> (no attention change): build <tag> (exported commit, full build) against the round-1 final build
# `final` (profile rx7600-refine2): golden checks, stage, gate (timed session 5 valid runs per cell, next token on three prompts, unmodified
# verify.sh against s0-parent-verify, warm traces), byte comparison of the outputs of every prefill linear shape (arithmetic unchanged).
# For a candidate stacked on an earlier round-2 candidate set PARENT_TAG / PARENT_PROFILE in the environment.
source "$(dirname "$(readlink -f "$0")")/env.sh"
S=$1; TAG=$2; PROF=$3; NOTE=$4; PT=${PARENT_TAG:-final}; PP=${PARENT_PROFILE:-rx7600-refine2}
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/chain8-$S.status; }
st "waiting for the build $TAG"; until grep -q BUILD_BOTH_DONE $A/build/rx7600/$TAG.src.txt 2>/dev/null; do sleep 30; done
grep -q 'rc=0 main' $A/build/rx7600/$TAG.src.txt && grep -q 'rc=0 traced' $A/build/rx7600/$TAG.src.txt || { st "STOP: build $TAG failed"; exit 1; }
while pgrep -f 'tools/linear_screen.sh' > /dev/null; do sleep 30; done   # kernel screens are GPU jobs: not alongside a timed session
st "golden"; $T/spirv_same.sh $TAG > $A/logs/golden-$TAG.out 2>&1; st "golden: $(tail -2 $A/logs/golden-$TAG.out | tr '\n' ' ')"
B="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko"
PE="$B ET_VK_SARC_RX7600_PROFILE=$PP"; CE="$B ET_VK_SARC_RX7600_PROFILE=$PROF"
st "stage $S"; $T/stage.sh $S $PT "$PE" $TAG "$CE" "$NOTE" > $A/logs/stage-$S.out 2>&1 || { st "STOP: stage failed"; exit 1; }
st "gate"; $T/gate.sh $S "$CE"; st "gate done"
st "bitwise"; $T/linear_bitwise.sh $S > $A/stage/$S/linear-bitwise.out 2>&1; st "bitwise: $(tail -1 $A/stage/$S/linear-bitwise.out)"
st CHAIN8_DONE
