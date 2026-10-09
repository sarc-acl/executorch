#!/bin/bash
# chain11.sh <final profile> [commit=HEAD]: the closing of the campaign (task section 6.6, rule R11) on the
# build of the committed branch head, no local patch.
#   1. the builds `pristine` (the first campaign's pristine parent, PRISTINE_COMMIT) and `topic7` come from
#      chain12.sh; topic7 must be the build of the commit, or of a commit that differs from it only under openspec/
#   2. (the hook condition on topic7 with no environment is a unit of chain12.sh)
#   3. stage s4-final (parent2 with the parent environment against topic7 with the final profile) and
#      gate_sdpa.sh, 7 repeats: the full gate of the final stack, the timed session, the traces
#   4. 12 passes each of tiers peaked and fused; reference error on the final build (sdpa_ref.sh)
#   5. stage s5-pristine (pristine, no environment, against topic7 with the final profile): timed session only
#   6. roofs (igpu-roofline, plan fast), decode comparison of s4-final, collection
# Every unit waits for the coordinator hold through the tools it calls. Ends CHAIN11_DONE.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"; PROF=${1:?final profile}; REV=$(git -C $ET rev-parse --verify "${2:-HEAD}^{commit}") || exit 2
TAG=topic7; S=s4-final; SP=s5-pristine; CE="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=$PROF"
ST=$A/logs/chain11.status; say() { echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
say "chain11 start $REV $TAG $PROF (kernel $(uname -r))"
for t in pristine $TAG; do grep -qsx BUILD_BOTH_OK $A/build/$t.src.txt || { say "CHAIN11_STOPPED build $t missing (chain12.sh builds it)"; exit 1; }; done
[[ $(sed -n 's/^commit=//p' $A/build/$TAG.src.txt) == "$REV" ]] || git -C $ET diff --quiet $(sed -n 's/^commit=//p' $A/build/$TAG.src.txt) $REV -- . ':!openspec' || { say "CHAIN11_STOPPED $TAG is not the build of $REV (sources differ outside openspec/)"; exit 1; }
busy_wait "gate $S"
$TOOLS/stage.sh $S parent2 "$PARENT_ENV" $TAG "$CE" "final stack $PROF on the committed head against the parent (b580-refine3)" > $A/logs/$S.stage.out 2>&1 || { say "CHAIN11_STOPPED staging $S failed"; exit 1; }
B580_REPS=7 $TOOLS/gate_sdpa.sh $S > $A/logs/gate-$S.out 2>&1; say "gate $S rc=$? $(cat $A/stage/$S/gate.done 2>/dev/null)"
D=$A/stage/$S; O=$D/sdpa-correctness-extra; mkdir -p $O
for tier in peaked fused; do for i in $(seq 1 12); do
  env $CE $TOOLS/gl.sh $D/test_llama_microbench --sdpa-correctness-only --sdpa-tier=$tier > $O/cand-$tier-r$i.log 2>&1; echo "cand,$tier,$i,$?" >> $O/rc.csv
done; say "extra tier $tier: 12 passes, $(cat $O/cand-$tier-r*.log | grep -c 'fused=sarc_dev') fused case runs, $(cat $O/cand-$tier-r*.log | grep 'sdpa-correctness\] ' | grep -c 'mismatches=0/') case runs with 0 mismatches of $(cat $O/cand-$tier-r*.log | grep -c 'mismatches=')"; done
$TOOLS/sdpa_ref.sh final-ref $TAG $PROF > $A/logs/final-ref.out 2>&1; say "final-ref rc=$? full: $(tail -1 $A/raw/final-ref/full.csv) extended: $(tail -1 $A/raw/final-ref/extended.csv) peaked: $(tail -1 $A/raw/final-ref/peaked.csv)"
$TOOLS/stage.sh $SP pristine "" $TAG "$CE" "final stack $PROF on the committed head against the pristine parent of the first campaign (no profile)" > $A/logs/$SP.stage.out 2>&1 || { say "CHAIN11_STOPPED staging $SP failed"; exit 1; }
hold_wait "session $SP"; gpu_shared || { say "CHAIN11_STOPPED build lock"; exit 75; }
$TOOLS/session.sh $SP --reps 7 > $A/stage/$SP/e2e5.out 2>&1; say "session $SP rc=$? $(tail -1 $A/stage/$SP/e2e5.out | cut -c1-120)"
python3 $TOOLS/summarize.py $A/stage/$SP/raw > $A/stage/$SP/raw/summary.csv 2>&1
$TOOLS/roof.sh final > $A/logs/roof-final.out 2>&1; say "roof rc=$? $(tail -1 $A/logs/roof-final.out)"
$TOOLS/decode_ab.sh $S > $A/logs/decode-$S.out 2>&1; say "decode $S rc=$?"
$TOOLS/collect.sh > $A/logs/collect-$S.out 2>&1; say "collect rc=$?"
say "CHAIN11_DONE"
