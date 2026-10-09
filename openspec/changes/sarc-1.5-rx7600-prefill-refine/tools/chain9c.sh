#!/bin/bash
# chain9c.sh <final build tag>: the steps of chain9.sh after the gate of r2-final, rerun after the root filesystem filled up during them (ENOSPC,
# 2026-10-09 09:25-09:28 UTC; the failed outputs are in <artifacts>/superseded/r2-final-enospc). Same text as chain9.sh from `bitwise` on.
# Original description: the final verification of round 2 (R11) on the build of the committed head (no local patch):
#   session r2-final      = pristine parent (build `parent`, ET_VK_SARC_UNVERIFIED=1) against the final stack (profile rx7600-refine5 + fused3sb + softmax r3):
#                           gate (timed session, SDPA tiers 12 passes each, unmodified verify.sh against s0, warm traces), byte comparison of the outputs of
#                           every prefill linear shape, SDPA output / reference-error evidence;
#   session r2-final-r1   = round 1's final stack (build `final`, rx7600-refine2) against the final stack: timed session and next token only.
source "$(dirname "$(readlink -f "$0")")/env.sh"
TAG=$1
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/chain9.status; }
B="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko"
P="ET_VK_SARC_UNVERIFIED=1"; R1="$B ET_VK_SARC_RX7600_PROFILE=rx7600-refine2"; F="$B ET_VK_SARC_RX7600_PROFILE=rx7600-refine5"
st "resume after ENOSPC: chain9c (the bitwise comparison was started by chain9b, whose shell was stopped because it lacked the environment variables of the second session)"
while pgrep -f "tools/linear_bitwise.sh" > /dev/null; do sleep 20; done; st "bitwise: $(tail -1 $A/stage/r2-final/linear-bitwise.out)"
$T/sdpa_evidence.sh r2-final "all extended peaked full" > $A/stage/r2-final/sdpa-evidence.out 2>&1; st "sdpa evidence done: $(head -1 $A/stage/r2-final/sdpa-evidence.out)"
st "stage r2-final-r1"; $T/stage.sh r2-final-r1 final "$R1" $TAG "$F" "round 2 final stack (build $TAG) against round 1's final stack (build final, rx7600-refine2)" > $A/logs/stage-r2-final-r1.out 2>&1 || { st "STOP: stage 2 failed"; exit 1; }
D=$A/stage/r2-final-r1; eval "$($T/hold.sh vars)"
while :; do $T/hold.sh wait "chain9: timed session r2-final-r1"; t0=$SECONDS; while (( $(gtemp) > 48 && SECONDS - t0 < 1800 )); do sleep 10; done; [[ -e $HOLD ]] || break; done
echo "prestart_temp=$(gtemp) waited=$((SECONDS - t0))s $(date -u +%FT%TZ)" > $D/prestart.txt
$T/e2e5.sh --stage $D --out raw --reps 5 > $D/e2e5.out 2>&1; /usr/bin/python3 $T/summarize.py $D/raw 5 > $D/raw/summary.csv 2>&1
st "r2-final-r1 session done: $(tail -1 $D/raw/summary.csv)"
st CHAIN9_DONE
