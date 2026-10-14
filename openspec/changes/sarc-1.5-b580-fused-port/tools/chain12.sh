#!/bin/bash
# chain12.sh <commit>: after chain 9. The gate of candidate 2 in session s3-c2 failed on its timing items only:
# a desktop client held 33 to 45 % of the card for the whole session and no timed run was valid. This chain
# builds the committed head and repeats that gate on it. As run on 2026-10-09 it called busy_wait before the
# gate and inside session.sh (removed since: thresholds-history.md).
#   1. build `pristine` (PRISTINE_COMMIT, for the closing session) and `topic7` (the commit) with its probe
#   2. hook condition on topic7: unmodified verify.sh with no environment against s0-parent-noenv
#   3. stage s3b-c2 (topic7 b580-fused1 against topic7 b580-fused2); gate_sdpa.sh, 7 repeats
#   4. reference error (b580-fused2, b580-refine3-nzf), logits probe, decide.py, decode comparison, collection
# Ends CHAIN12_DONE; chain11.sh <final profile> follows by hand.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"; REV=$(git -C $ET rev-parse --verify "${1:?commit}^{commit}") || exit 2; TAG=topic7; S=s3b-c2
ST=$A/logs/chain12.status; say() { echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
say "chain12 waiting for chain9; commit $REV"
until grep -q 'CHAIN9_DONE\|CHAIN9_STOPPED' $A/logs/chain9.status 2>/dev/null; do sleep 30; done
for t in "pristine $PRISTINE_COMMIT" "$TAG $REV"; do set -- $t
  if ! grep -qsx BUILD_BOTH_OK $A/build/$1.src.txt; then
    $TOOLS/build-both.sh $1 $2 > $A/logs/build-$1.out 2>&1 || { say "CHAIN12_STOPPED $1 build failed"; exit 1; }
  fi
  say "$1 built ($(sed -n 's/^commit=//p' $A/build/$1.src.txt)), $(grep golden_rc $A/build/$1.src.txt)"
done
[[ -x $A/build/$TAG/probe/logits_probe ]] || $TOOLS/build-probe.sh $TAG > $A/logs/build-probe-$TAG.out 2>&1 || { say "CHAIN12_STOPPED probe build failed"; exit 1; }
$TOOLS/parent_verify.sh s0-$TAG-noenv $TAG "" > $A/logs/s0-$TAG-noenv.out 2>&1; say "s0-$TAG-noenv rc=$? $(cat $A/stage/s0-$TAG-noenv/gate.done 2>/dev/null)"
python3 $TOOLS/verify_diff.py $A/stage/s0-parent-noenv/verify.out $A/stage/s0-$TAG-noenv/verify.out > $A/stage/s0-$TAG-noenv/verify_diff.txt 2>&1
say "hook condition, no environment, parent2 vs $TAG: $(tail -1 $A/stage/s0-$TAG-noenv/verify_diff.txt)"
U="ET_VK_SARC_UNVERIFIED=1"; C1="$U ET_VK_SARC_DEV_PROFILE=b580-fused1"; C2="$U ET_VK_SARC_DEV_PROFILE=b580-fused2"
$TOOLS/stage.sh $S $TAG "$C1" $TAG "$C2" "candidate 2 b580-fused2 against its parent, candidate 1 (b580-fused1); same build, the committed head; repeats the gate of s3-c2" > $A/logs/$S.stage.out 2>&1 || { say "CHAIN12_STOPPED staging failed"; exit 1; }
B580_REPS=7 $TOOLS/gate_sdpa.sh $S > $A/logs/gate-$S.out 2>&1; say "gate $S rc=$? $(cat $A/stage/$S/gate.done 2>/dev/null)"
for p in b580-fused2 b580-refine3-nzf; do
  $TOOLS/sdpa_ref.sh c2-ref7-$p $TAG $p > $A/logs/c2-ref7-$p.out 2>&1; say "c2-ref7-$p rc=$? full: $(tail -1 $A/raw/c2-ref7-$p/full.csv) extended: $(tail -1 $A/raw/c2-ref7-$p/extended.csv) peaked: $(tail -1 $A/raw/c2-ref7-$p/peaked.csv)"
done
$TOOLS/probe.sh $S > $A/logs/probe-$S.out 2>&1; say "probe $S rc=$? $(tail -1 $A/stage/$S/probe/analysis.txt 2>/dev/null)"
python3 $TOOLS/decide.py $A/stage/$S --arithmetic $A/raw/c2-ref7-b580-refine3-nzf/full.csv > $A/logs/decide-$S.out 2>&1; say "decide $S rc=$? $(head -1 $A/stage/$S/decision.txt 2>/dev/null)"
$TOOLS/decode_ab.sh $S > $A/logs/decode-$S.out 2>&1; say "decode $S rc=$?"
$TOOLS/collect.sh > $A/logs/collect-$S.out 2>&1; say "collect rc=$?"
say "CHAIN12_DONE"
