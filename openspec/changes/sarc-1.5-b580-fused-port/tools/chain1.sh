#!/bin/bash
# chain1.sh <topic commit>: the first detached chain of the campaign, one unit after another, each under its own
# locks and the coordinator hold. Status lines go to logs/chain1.status; a unit that fails stops the chain.
#   1. wait for the parent build (build/parent2, started separately)
#   2. build the topic commit as build/topic1
#   3. smoke run of the fused kernel: one pass of every SDPA correctness tier with b580-fused1 (not a gate)
#   4. parent snapshots: s0-parent-verify (parent environment), s0-parent-noenv (no environment)
#   5. hook condition (D4): topic1 with no environment against s0-parent-noenv, line by line
#   6. baseline + A/A with calibration: s1-aa, parent2 against topic1, both with the parent environment
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"; REV=${1:?topic commit}; ST=$A/logs/chain1.status; mkdir -p $A/logs $A/raw
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
die() { say "CHAIN1_STOPPED $*"; exit 1; }
say "chain1 start, topic $REV"
while [[ ! -s $A/logs/build-parent2.status ]]; do sleep 20; done
grep -qx BUILD_BOTH_OK $A/build/parent2.src.txt || die "parent2 build failed"
say "parent2 built"
$TOOLS/build-both.sh topic1 $REV > $A/logs/build-topic1.out 2>&1 || die "topic1 build failed"
say "topic1 built"
O=$A/raw/c1-smoke; mkdir -p $O; T=$A/build/topic1/tests/test_llama_microbench
for tier in all extended full peaked fused; do
  env $PARENT_ENV ET_VK_SARC_DEV_PROFILE=b580-fused1 $TOOLS/gl.sh $T --sdpa-correctness-only --sdpa-tier=$tier > $O/fused1-$tier.log 2>&1
  say "smoke b580-fused1 $tier rc=$? $(grep -c 'mismatches=0/' $O/fused1-$tier.log) case(s) with 0 mismatches, $(grep -c 'fused=sarc_dev' $O/fused1-$tier.log) fused, $(grep -c FAILED $O/fused1-$tier.log) FAILED line(s)"
done
$TOOLS/parent_verify.sh s0-parent-verify parent2 "$PARENT_ENV" > $A/logs/s0-parent-verify.out 2>&1; say "s0-parent-verify rc=$? $(cat $A/stage/s0-parent-verify/gate.done 2>/dev/null)"
grep -q 'VERIFY_DONE rc=0' $A/stage/s0-parent-verify/verify.out || die "no parent snapshot"
$TOOLS/parent_verify.sh s0-parent-noenv parent2 "" > $A/logs/s0-parent-noenv.out 2>&1; say "s0-parent-noenv rc=$? $(cat $A/stage/s0-parent-noenv/gate.done 2>/dev/null)"
$TOOLS/parent_verify.sh s0-topic1-noenv topic1 "" > $A/logs/s0-topic1-noenv.out 2>&1; say "s0-topic1-noenv rc=$? $(cat $A/stage/s0-topic1-noenv/gate.done 2>/dev/null)"
python3 $TOOLS/verify_diff.py $A/stage/s0-parent-noenv/verify.out $A/stage/s0-topic1-noenv/verify.out > $A/stage/s0-topic1-noenv/verify_diff.txt 2>&1
say "hook condition, no environment, parent2 vs topic1: $(tail -1 $A/stage/s0-topic1-noenv/verify_diff.txt)"
$TOOLS/stage.sh s1-aa parent2 "$PARENT_ENV" topic1 "$PARENT_ENV" "baseline + A/A: parent 51d9d757f against the topic build, both with the parent environment" > $A/logs/s1-aa.stage.out 2>&1 || die "staging s1-aa failed"
$TOOLS/session.sh s1-aa --calibrate > $A/logs/s1-aa.out 2>&1; say "s1-aa rc=$? $(tail -1 $A/stage/s1-aa/raw/done.txt 2>/dev/null)"
say "CHAIN1_DONE"
