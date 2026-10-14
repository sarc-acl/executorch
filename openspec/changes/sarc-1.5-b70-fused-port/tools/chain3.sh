#!/bin/bash
# chain3.sh <final profile> <commit>: the closing (task section 6.5, rule R11) on the build of the committed
# branch head, no local patch.
#   1. chain2.sh topic2 s3-final <profile> final-ref <commit>: build topic2 from an export of the commit, SPIR-V
#      identity, the full gate of the final stack against the parent (SDPA tiers, verify.sh, timed session,
#      traces), the extra tiers, reference error, logits probe, decision, attention table, decode
#   2. hook condition on the final build: test_sarc_select (executables kept in raw/final-select/) and
#      verify.sh with no environment (s0-topic2-noenv) against s0-parent-noenv, line by line
#   3. s4-pristine: the first campaign's pristine parent build (no profile, no ET_VK_SARC_UNVERIFIED) against
#      topic2 with the final profile: timed session, then warm traces of 1B 4w and 8B 4w with the attention table
#   4. sarc/tools/check.sh --no-build on the working copy; collection
# Every unit obeys the coordinator hold through the tool it calls. Ends CHAIN3_DONE.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"; PROF=${1:?final profile}; REV=$(git -C $ET rev-parse --verify "${2:?commit}^{commit}") || exit 2
ST=$A/logs/chain3.status; say() { echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }; REPS=$(cat $A/reps)
say "chain3 start $PROF $REV"
$TOOLS/chain2.sh topic2 s3-final $PROF final-ref $REV > $A/logs/chain2-s3-final.out 2>&1; say "chain2 s3-final rc=$? $(cat $A/stage/s3-final/gate.done 2>/dev/null); $(head -1 $A/stage/s3-final/decision.txt 2>/dev/null)"
[[ $(sed -n 's/^commit=//p' $A/build/topic2.src.txt) == "$REV" ]] || { say "CHAIN3_STOPPED topic2 is not the build of $REV"; exit 1; }
$TOOLS/select_check.sh final-select parent topic2 > $A/logs/select-final.txt 2>&1; say "select: $(tail -1 $A/logs/select-final.txt)"
[[ -e $A/stage/s0-topic2-noenv/gate.done ]] || $TOOLS/parent_verify.sh s0-topic2-noenv topic2 "" > $A/logs/s0-topic2-noenv.out 2>&1
python3 $TOOLS/verify_diff.py $A/stage/s0-parent-noenv/verify.out $A/stage/s0-topic2-noenv/verify.out > $A/stage/s0-topic2-noenv/verify_diff.txt 2>&1; say "hook condition on topic2, verify with no environment: $(tail -1 $A/stage/s0-topic2-noenv/verify_diff.txt)"
CE="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=$PROF"; SP=s4-pristine
$TOOLS/stage.sh $SP pristine "" topic2 "$CE" "final stack $PROF on the committed head against the pristine parent of the first campaign (build parent of its s12-final5; no profile, no ET_VK_SARC_UNVERIFIED)" > $A/logs/$SP.stage.out 2>&1 || { say "CHAIN3_STOPPED staging $SP failed"; exit 1; }
$TOOLS/session.sh $SP --reps $REPS > $A/stage/$SP/e2e5.out 2>&1; say "session $SP rc=$? $(tail -1 $A/stage/$SP/e2e5.out | cut -c1-160)"
python3 $TOOLS/summarize.py $A/stage/$SP/raw > $A/stage/$SP/raw/summary.csv 2>&1
$TOOLS/trace.sh $SP 1b,8b 4w "parent cand" > $A/stage/$SP/trace.out 2>&1; say "trace $SP rc=$?"
$XE2_PYTHON $TOOLS/trace_attention.py $A/stage/$SP > $A/logs/trace-attention-$SP.out 2>&1; say "attention table $SP rc=$?"
( cd $ET && bash sarc/tools/check.sh --no-build ) > $A/logs/check-no-build.txt 2>&1; say "check.sh --no-build rc=$? $(tail -1 $A/logs/check-no-build.txt)"
$TOOLS/collect.sh > $A/logs/collect-chain3.out 2>&1; say "collect rc=$?"
say "CHAIN3_DONE"
