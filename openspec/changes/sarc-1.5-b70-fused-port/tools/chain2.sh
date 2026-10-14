#!/bin/bash
# chain2.sh <build tag> <session> <profile> <ref name> [commit]: gate one candidate profile on a topic build
# against the parent (build parent, parent environment). After the B580 campaign's chain8.sh.
#   1. (only with <commit>) build <tag> from an export of the commit, and its logits probe
#   2. stage <session>; gate_sdpa.sh: 12 passes x SDPA tiers all / extended / full, unmodified verify.sh compared
#      line by line with s0-parent-verify, the timed session (repeats from <artifacts>/reps, set from the A/A by
#      the rule of thresholds.txt), warm traces, gate_check.py
#   3. 12 passes each of tiers peaked and fused with the candidate environment (reported, not gate items)
#   4. sdpa_ref.sh: error against the fp32 reference, parent kernels and candidate on the staged build
#   5. probe.sh (logits, four arms) and decide.py --arithmetic; attention table of the traces; decode; collect
# Every unit obeys the coordinator hold through the tool it calls. Ends CHAIN2_DONE <session>.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"; TAG=${1:?tag}; S=${2:?session}; PROF=${3:?profile}; REF=${4:?ref name}; REV=${5:-}
ST=$A/logs/chain2-$S.status; say() { echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
REPS=$(cat $A/reps 2>/dev/null); [[ $REPS == 5 || $REPS == 7 ]] || { say "CHAIN2_STOPPED no repeat count in $A/reps"; exit 1; }
say "chain2 start $TAG $S $PROF reps=$REPS (kernel $(uname -r))"
if ! grep -qsx BUILD_BOTH_OK $A/build/$TAG.src.txt; then
  [[ -n $REV ]] || { say "CHAIN2_STOPPED build $TAG missing and no commit given"; exit 1; }
  $TOOLS/build-both.sh $TAG $REV > $A/logs/build-$TAG.out 2>&1 || { say "CHAIN2_STOPPED $TAG build failed"; exit 1; }
  $TOOLS/build-probe.sh $TAG > $A/logs/build-probe-$TAG.out 2>&1 || { say "CHAIN2_STOPPED probe build failed"; exit 1; }
  $TOOLS/spv_identity.sh $TAG parent /mnt/linux-share/hmz-campaigns/b580-fused/.artifacts/build/topic6 > $A/logs/spv-identity-$TAG.txt 2>&1; say "spirv identity $TAG rc=$? $(tail -1 $A/logs/spv-identity-$TAG.txt)"
fi
say "$TAG = $(sed -n 's/^commit=//p' $A/build/$TAG.src.txt), $(grep golden_rc $A/build/$TAG.src.txt)"
CE="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=$PROF"
$TOOLS/stage.sh $S parent "$PARENT_ENV" $TAG "$CE" "candidate $PROF against the parent (xe2-refine5)" > $A/logs/$S.stage.out 2>&1 || { say "CHAIN2_STOPPED staging failed"; exit 1; }
XE2_REPS=$REPS $TOOLS/gate_sdpa.sh $S > $A/logs/gate-$S.out 2>&1; say "gate $S rc=$? $(cat $A/stage/$S/gate.done 2>/dev/null); $(tail -1 $A/stage/$S/verify_diff.txt 2>/dev/null)"
D=$A/stage/$S; O=$D/sdpa-correctness-extra; mkdir -p $O
for tier in peaked fused; do for i in $(seq 1 12); do
  env $CE $TOOLS/gl.sh $D/test_llama_microbench --sdpa-correctness-only --sdpa-tier=$tier > $O/cand-$tier-r$i.log 2>&1; echo "cand,$tier,$i,$?" >> $O/rc.csv
done; say "extra tier $tier: 12 passes, $(cat $O/cand-$tier-r*.log | grep -c 'fused=sarc_dev') fused case runs, $(cat $O/cand-$tier-r*.log | grep 'sdpa-correctness\] ' | grep -c 'mismatches=0/') case runs with 0 mismatches of $(cat $O/cand-$tier-r*.log | grep -c 'mismatches=')"; done
say "gated tiers: $(cat $D/sdpa-correctness/cand-*.log | grep -c 'fused=sarc_dev') case runs served by the fused kernel of $(cat $D/sdpa-correctness/cand-*.log | grep -c '^\[sdpa-kernels\]')"
$TOOLS/sdpa_ref.sh $REF $TAG $PROF > $A/logs/$REF.out 2>&1; say "$REF rc=$? full: $(tail -1 $A/raw/$REF/full.csv) extended: $(tail -1 $A/raw/$REF/extended.csv) peaked: $(tail -1 $A/raw/$REF/peaked.csv)"
$TOOLS/probe.sh $S > $A/logs/probe-$S.out 2>&1; say "probe $S rc=$? $(tail -1 $D/probe/analysis.txt 2>/dev/null)"
python3 $TOOLS/decide.py $D --arithmetic $A/raw/$REF/full.csv > $A/logs/decide-$S.out 2>&1; say "decide $S rc=$? $(head -1 $D/decision.txt 2>/dev/null)"
$XE2_PYTHON $TOOLS/trace_attention.py $D > $A/logs/trace-attention-$S.out 2>&1; say "attention table rc=$?"
$TOOLS/decode_ab.sh $S > $A/logs/decode-$S.out 2>&1; say "decode $S rc=$?"
mkdir -p $C/results/b70/sdpa-error/$REF && cp -f $A/raw/$REF/*.csv $A/raw/$REF/env.txt $C/results/b70/sdpa-error/$REF/ 2>/dev/null
$TOOLS/collect.sh > $A/logs/collect-$S.out 2>&1; say "collect rc=$?"
say "CHAIN2_DONE $S"
