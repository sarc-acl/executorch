#!/bin/bash
# chain7.sh: the final verification (R11) of the final stack on the build `final` of the committed head (18cc0d53a),
# against the pristine parent (build `parent`, ET_VK_SARC_UNVERIFIED=1 only): logits_probe build, stage `final`, gate
# (timed session, SDPA tiers 12 passes each, verify.sh against s0, traces), SDPA output / reference-error evidence,
# real-text logits probe (D3.2 / D3.3). The golden check is run by hand on build/rx7600/final (not a GPU job).
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/chain7.status; }
st "waiting for the build"; until grep -q BUILD_BOTH_DONE $A/build/rx7600/final.src.txt 2>/dev/null; do sleep 30; done
grep -q 'rc=0 main' $A/build/rx7600/final.src.txt || { st "STOP: build failed"; exit 1; }
st "probe build"; touch $A/.building; $T/hold.sh run "build probe final" $T/build_probe.sh final > $A/build/rx7600/final-probe.log 2>&1; st "probe build rc=$?"; rm -f $A/.building
P="ET_VK_SARC_UNVERIFIED=1"
F="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko ET_VK_SARC_RX7600_PROFILE=rx7600-refine2"
$T/stage.sh final parent "$P" final "$F" "final stack (rx7600-refine2 + fused3sb, build final = commit $(cat $A/src/rx7600/final/COMMIT)) against the pristine parent f5f1bf10c" > $A/logs/stage-final.out 2>&1
cp -f $A/build/rx7600/final/probe/logits_probe $A/stage/final/lp
st "final gate"; $T/gate.sh final "$F" sdpa; st "final gate done"
$T/sdpa_evidence.sh final "all extended peaked full" > $A/stage/final/sdpa-evidence.out 2>&1; st "final sdpa evidence done: $(head -1 $A/stage/final/sdpa-evidence.out)"
$T/probe_run.sh final > $A/stage/final/probe.out 2>&1
<toolchain-share>/Python-3.12.9-1/bin/python3 $T/probe_compare.py $A/stage/final/probe $A/stage/final/probe/real-text-compare.csv > $A/stage/final/probe/compare.out 2>&1
st "final probe done: $(tail -1 $A/stage/final/probe/compare.out)"
st CHAIN7_DONE
