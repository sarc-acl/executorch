#!/bin/bash
# chain8.sh <commit> <build tag> <session> <profile> <ref name>: build a topic commit and gate one candidate
# profile on it against the parent (build parent2, parent environment), with 7 repeats (calibration of s1-aa2).
#   1. build <tag> and its logits probe (skipped when the tag is already built)
#   2. stage <session>; gate_sdpa.sh (12 passes x 3 SDPA tiers, verify.sh, timed session, traces, gate_check)
#   3. 12 passes each of the tiers peaked and fused with the candidate environment (reported, not gate items)
#   4. sdpa_ref.sh: error against the fp32 reference, parent kernels and candidate on the staged test binary
#   5. probe.sh (logits, four arms) and decide.py --arithmetic; decode_ab.sh; collect.sh
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"; REV=${1:?commit}; TAG=${2:?tag}; S=${3:?session}; PROF=${4:?profile}; REF=${5:?ref name}
ST=$A/logs/chain8-$S.status; say() { echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
say "chain8 start $REV $TAG $S $PROF"
if ! grep -qsx BUILD_BOTH_OK $A/build/$TAG.src.txt; then
  $TOOLS/build-both.sh $TAG $REV > $A/logs/build-$TAG.out 2>&1 || { say "CHAIN8_STOPPED $TAG build failed"; exit 1; }
fi
[[ -x $A/build/$TAG/probe/logits_probe ]] || $TOOLS/build-probe.sh $TAG > $A/logs/build-probe-$TAG.out 2>&1 || { say "CHAIN8_STOPPED probe build failed"; exit 1; }
say "$TAG built ($(sed -n 's/^commit=//p' $A/build/$TAG.src.txt)), $(grep golden_rc $A/build/$TAG.src.txt)"
CE="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=$PROF"
$TOOLS/stage.sh $S parent2 "$PARENT_ENV" $TAG "$CE" "candidate $PROF against the parent (b580-refine3)" > $A/logs/$S.stage.out 2>&1 || { say "CHAIN8_STOPPED staging failed"; exit 1; }
B580_REPS=7 $TOOLS/gate_sdpa.sh $S > $A/logs/gate-$S.out 2>&1; say "gate $S rc=$? $(cat $A/stage/$S/gate.done 2>/dev/null)"
D=$A/stage/$S; O=$D/sdpa-correctness-extra; mkdir -p $O
for tier in peaked fused; do for i in $(seq 1 12); do
  env $CE $TOOLS/gl.sh $D/test_llama_microbench --sdpa-correctness-only --sdpa-tier=$tier > $O/cand-$tier-r$i.log 2>&1; echo "cand,$tier,$i,$?" >> $O/rc.csv
done; say "extra tier $tier: 12 passes, $(cat $O/cand-$tier-r*.log | grep -c 'fused=sarc_dev') fused case runs, $(cat $O/cand-$tier-r*.log | grep 'sdpa-correctness\] ' | grep -c 'mismatches=0/') case runs with 0 mismatches of $(cat $O/cand-$tier-r*.log | grep -c 'mismatches=')"; done
$TOOLS/sdpa_ref.sh $REF $TAG $PROF > $A/logs/$REF.out 2>&1; say "$REF rc=$? full: $(tail -1 $A/raw/$REF/full.csv) extended: $(tail -1 $A/raw/$REF/extended.csv) peaked: $(tail -1 $A/raw/$REF/peaked.csv)"
$TOOLS/probe.sh $S > $A/logs/probe-$S.out 2>&1; say "probe $S rc=$? $(tail -1 $A/stage/$S/probe/analysis.txt 2>/dev/null)"
python3 $TOOLS/decide.py $D --arithmetic $A/raw/$REF/full.csv > $A/logs/decide-$S.out 2>&1; say "decide $S rc=$? $(head -1 $D/decision.txt 2>/dev/null)"
$TOOLS/decode_ab.sh $S > $A/logs/decode-$S.out 2>&1; say "decode $S rc=$?"
$TOOLS/collect.sh > $A/logs/collect-$S.out 2>&1; say "collect rc=$?"
say "CHAIN8_DONE $S"
