#!/bin/bash
# chain9.sh: gate of candidate 2 (b580-fused2 = b580-fused1 + the fp32 no-tail softmax 4070ti_nzf for the calls
# the fused kernel does not take) against its parent, candidate 1; both arms are build topic6. Waits for chain 8.
#   1. softmax look, not timed: one pass of tiers all / extended / full / peaked with b580-refine3-nzf (the softmax
#      variant between the parent's QK^T and attn*V kernels, where its shortened zero tail is active), and the
#      test's control run with the cooperative-matrix kernels disabled under b580-fused2 (--sdpa-force-fallback)
#   2. stage s3-c2 and gate_sdpa.sh, 7 repeats
#   3. reference error: b580-fused2 and b580-refine3-nzf against the parent's kernels
#   4. logits probe, decide.py, decode comparison, collection
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"; S=s3-c2; TAG=topic6
ST=$A/logs/chain9.status; say() { echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
say "chain9 waiting for chain8"
until grep -q 'CHAIN8_DONE\|CHAIN8_STOPPED' $A/logs/chain8-s2-c1.status 2>/dev/null; do sleep 30; done
U="ET_VK_SARC_UNVERIFIED=1"; C1="$U ET_VK_SARC_DEV_PROFILE=b580-fused1"; C2="$U ET_VK_SARC_DEV_PROFILE=b580-fused2"
O=$A/raw/c2-smoke; mkdir -p $O; T=$A/build/$TAG/tests/test_llama_microbench
for tier in all extended full peaked; do
  env $U ET_VK_SARC_DEV_PROFILE=b580-refine3-nzf $TOOLS/gl.sh $T --sdpa-correctness-only --sdpa-tier=$tier > $O/nzf-$tier.log 2>&1
  say "smoke b580-refine3-nzf $tier rc=$? $(grep -c 'mismatches=0/.*PASSED' $O/nzf-$tier.log)/$(grep -c 'mismatches=' $O/nzf-$tier.log) cases with 0 mismatches, softmax $(grep -o 'softmax=[a-z0-9_]*' $O/nzf-$tier.log | sort -u | tr '\n' ' ')"
done
env $C2 $TOOLS/gl.sh $T --sdpa-correctness-only --sdpa-tier=extended --sdpa-force-fallback > $O/fused2-fallback-extended.log 2>&1
say "control b580-fused2 with the cooperative-matrix kernels disabled, tier extended: rc=$? $(grep -c 'mismatches=0/' $O/fused2-fallback-extended.log)/$(grep -c 'mismatches=' $O/fused2-fallback-extended.log) cases with 0 mismatches, softmax $(grep -o 'softmax=[a-z0-9_]*' $O/fused2-fallback-extended.log | sort -u | tr '\n' ' ')"
$TOOLS/stage.sh $S $TAG "$C1" $TAG "$C2" "candidate 2 b580-fused2 against its parent, candidate 1 (b580-fused1); same build" > $A/logs/$S.stage.out 2>&1 || { say "CHAIN9_STOPPED staging failed"; exit 1; }
B580_REPS=7 $TOOLS/gate_sdpa.sh $S > $A/logs/gate-$S.out 2>&1; say "gate $S rc=$? $(cat $A/stage/$S/gate.done 2>/dev/null)"
for p in b580-fused2 b580-refine3-nzf; do
  $TOOLS/sdpa_ref.sh c2-ref6-$p $TAG $p > $A/logs/c2-ref6-$p.out 2>&1; say "c2-ref6-$p rc=$? full: $(tail -1 $A/raw/c2-ref6-$p/full.csv) extended: $(tail -1 $A/raw/c2-ref6-$p/extended.csv) peaked: $(tail -1 $A/raw/c2-ref6-$p/peaked.csv)"
done
$TOOLS/probe.sh $S > $A/logs/probe-$S.out 2>&1; say "probe $S rc=$? $(tail -1 $A/stage/$S/probe/analysis.txt 2>/dev/null)"
python3 $TOOLS/decide.py $A/stage/$S --arithmetic $A/raw/c2-ref6-b580-refine3-nzf/full.csv > $A/logs/decide-$S.out 2>&1; say "decide $S rc=$? $(head -1 $A/stage/$S/decision.txt 2>/dev/null)"
$TOOLS/decode_ab.sh $S > $A/logs/decode-$S.out 2>&1; say "decode $S rc=$?"
$TOOLS/collect.sh > $A/logs/collect-$S.out 2>&1; say "collect rc=$?"
say "CHAIN9_DONE"
