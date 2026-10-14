#!/bin/bash
# chain1.sh <topic commit>: the first detached chain of the campaign, one unit at a time.
#   1. builds: parent (PARENT_COMMIT) and topic1 (<topic commit>: hooks + fused kernel sources), each from an
#      export of one commit, with their logits probes; pristine = a copy of the first campaign's build `parent`
#      (the pristine parent of its s12-final5, PRISTINE_COMMIT), not rebuilt
#   2. SPIR-V identity (spv_identity.sh) and the test_sarc_select part of hook condition D4 (select_check.sh)
#   3. parent snapshots: s0-parent-verify (parent, parent environment), s0-parent-noenv and s0-topic1-noenv
#      (no environment; compared line by line for hook condition D4)
#   4. s1-aa: baseline and A/A, parent against topic1, both with the parent environment, --calibrate
#   5. the one kernel screen of task section 5 (thresholds.txt kernel_screen), 3 rounds
# Every unit obeys the coordinator hold through the tool it calls. Ends CHAIN1_DONE.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"; REV=${1:?topic commit}; mkdir -p $A/logs $A/build
ST=$A/logs/chain1.status; say() { echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
say "chain1 start $REV (kernel $(uname -r), $(vulkaninfo --summary 2>/dev/null | grep -m1 driverInfo | xargs))"
for x in parent:$PARENT_COMMIT topic1:$REV; do t=${x%%:*}; r=${x##*:}
  if ! grep -qsx BUILD_BOTH_OK $A/build/$t.src.txt; then
    $TOOLS/build-both.sh $t $r > $A/logs/build-$t.out 2>&1 || { say "CHAIN1_STOPPED build $t failed"; exit 1; }
    $TOOLS/build-probe.sh $t > $A/logs/build-probe-$t.out 2>&1 || { say "CHAIN1_STOPPED probe build $t failed"; exit 1; }
  fi; say "$t built ($(sed -n 's/^commit=//p' $A/build/$t.src.txt)), $(grep golden_rc $A/build/$t.src.txt)"
done
if ! grep -qsx BUILD_BOTH_OK $A/build/pristine.src.txt; then
  cp -a $XE2_FIRST/build/parent $A/build/pristine && cp -a $XE2_FIRST/build/parent-traced $A/build/pristine-traced \
    && cp -a $XE2_FIRST/build/parent.golden.txt $A/build/pristine.golden.txt && mkdir -p $A/src && cp -a $XE2_FIRST/src/parent.export-manifest $A/src/pristine.export-manifest \
    && { echo "copied_from=$XE2_FIRST/build/parent (the first campaign's pristine parent build, timed in its s12-final5) at $(date -u +%FT%TZ)"; cat $XE2_FIRST/build/parent.src.txt; } > $A/build/pristine.src.txt \
    || { say "CHAIN1_STOPPED copy of the pristine build failed"; exit 1; }
  ( cd $A/build && sha256sum pristine/llama/examples/models/llama/llama_main pristine/tests/test_llama_microbench pristine-traced/llama/examples/models/llama/llama_main ) > $A/build/pristine.copy.txt
fi; say "pristine copied: $(head -1 $A/build/pristine.copy.txt | cut -c1-16) (first campaign: $(grep -m1 llama_main $XE2_FIRST/build/parent.src.txt | cut -c1-16))"
$TOOLS/spv_identity.sh topic1 parent /mnt/linux-share/hmz-campaigns/b580-fused/.artifacts/build/topic6 > $A/logs/spv-identity-topic1.txt 2>&1; say "spirv identity rc=$? $(tail -1 $A/logs/spv-identity-topic1.txt)"
$TOOLS/select_check.sh d4 parent topic1 > $A/logs/select-d4.txt 2>&1; say "select: $(tail -1 $A/logs/select-d4.txt)"
for x in "s0-parent-verify parent $PARENT_ENV" "s0-parent-noenv parent" "s0-topic1-noenv topic1"; do set -- $x; n=$1; b=$2; shift 2
  [[ -e $A/stage/$n/gate.done ]] || $TOOLS/parent_verify.sh $n $b "$*" > $A/logs/$n.out 2>&1; say "$n rc=$? $(cat $A/stage/$n/gate.done 2>/dev/null)"
done
python3 $TOOLS/verify_diff.py $A/stage/s0-parent-noenv/verify.out $A/stage/s0-topic1-noenv/verify.out > $A/stage/s0-topic1-noenv/verify_diff.txt 2>&1; say "hook condition, verify with no environment: $(tail -1 $A/stage/s0-topic1-noenv/verify_diff.txt)"
if [[ ! -e $A/stage/s1-aa ]]; then
  $TOOLS/stage.sh s1-aa parent "$PARENT_ENV" topic1 "$PARENT_ENV" "baseline and A/A: parent against the topic build, both with the parent environment (xe2-refine5)" > $A/logs/s1-aa.stage.out 2>&1 || { say "CHAIN1_STOPPED staging s1-aa failed"; exit 1; }
  $TOOLS/session.sh s1-aa --calibrate > $A/stage/s1-aa/e2e5.out 2>&1; say "session s1-aa rc=$? $(tail -1 $A/stage/s1-aa/e2e5.out | cut -c1-160)"
  python3 $TOOLS/summarize.py $A/stage/s1-aa/raw > $A/stage/s1-aa/raw/summary.csv 2>&1
fi
[[ -s $CLKMIN_FILE && -s $IDLE_FILE ]] || { say "CHAIN1_STOPPED no calibration from s1-aa"; exit 1; }
$TOOLS/screen_sdpa.sh screen1-select topic1 3 xe2-refine5 b70-fused-d64_t32x32s32m8ro b70-fused-d64_t16x64s16m8g4roj b70-fused-d64_t16x64s16m8g4oj \
  b70-fused-d128_t16x64s32m8ro b70-fused-d128_t16x128s16m8g8oj b70-fused-d128_t16x64s16m8g4oj > $A/logs/screen1-select.out 2>&1; say "screen1-select rc=$? $(tail -1 $A/raw/screen1-select/env.txt)"
python3 $TOOLS/screen_sdpa_summary.py $A/raw/screen1-select/screen.csv xe2-refine5 > $A/raw/screen1-select/summary.csv 2>&1
$TOOLS/collect.sh > $A/logs/collect-chain1.out 2>&1; say "collect rc=$?"
say "CHAIN1_DONE"
