#!/bin/bash
# chain10.sh [job to wait for]: device side. Everything again on the builds whose shaders are compiled with the
# golden's glslc (review of 2026-10-09: the cross image's glslc fails sarc/golden/spirv.json on 14 shipped variants
# of other devices, and compiles this campaign's attention kernels to other bytes). Builds parentg (8973ced76) and
# topic4p (0bed38090): the same commits as parent and topic4, the same recipe, the pinned shader compiler.
# Thresholds, clock floor and tools unchanged; the controls are new because the binaries are.
#   s0g-parent-verify, s0gn-noenv   unmodified verify.sh on parentg with the parent environment / nothing selected;
#   s9-aa2g       A/A: parentg against topic4p, both with the parent environment;
#   sdpa-error3   error against the fp32 reference with topic4p's test binary: stock, parent, orin-fused1, orin-fused2;
#   s10-c1g       candidate 1 gated (gate_sdpa.sh): parentg + parent environment against topic4p + orin-fused1;
#   c2g-pre, s11-c2g   candidate 2 (orin-fused2) pre-check and gate against candidate 1, both on topic4p;
#   final stack   the rule of STATUS.md (01:22 UTC) applied to s11-c2g: orin-fused2 only if accepted with at least
#                 2 % geomean, else orin-fused1; then s12-finalg only if it is orin-fused2;
#   s12n-noenv    hook control: verify.sh on topic4p with nothing selected against s0gn-noenv;
#   s13-pristineg the final stack against parentg with no environment: timed session, traces;
#   probe         41-prompt real-text logits: parentg default / tiled, final stack on topic4p default / tiled; the
#                 comparison and the reference-error rule on sdpa-error3 (probe/finalg-fused/).
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
ok() { grep -q '^GATE_ACCEPTED' $A/stage/$1/gate.done 2>/dev/null; }
PB=parentg; BT=topic4p; U=ET_VK_SARC_UNVERIFIED=1
PENV="$U ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64"; C1="$U ET_VK_SARC_DEV_PROFILE=orin-fused1"; C2="$U ET_VK_SARC_DEV_PROFILE=orin-fused2"
need $A/build/$PB/bundle/llama_main $A/build/$BT/bundle/llama_main $A/build/$PB/bundle/logits_dump $A/build/$BT/bundle/logits_dump
step env PARENT_CTL_NAME=s0g-parent-verify ./parent_verify.sh $PB "$PENV"
step env PARENT_CTL_NAME=s0gn-noenv ./parent_verify.sh $PB ""
ok s0g-parent-verify && ok s0gn-noenv || { echo "CHAIN_STOPPED: a control on parentg is not accepted"; exit 3; }
export PARENT_CTL_NAME=s0g-parent-verify
D=$A/stage/s9-aa2g
step ./stage.sh s9-aa2g $PB "$PENV" $BT "$PENV" "A/A on the builds with the golden's glslc: parentg against topic4p, both with the parent environment"
step ./session.sh s9-aa2g --clkmin-file $CHANGE/results/orin/clkmin.json > $D/e2e5.out 2>&1; tail -2 $D/e2e5.out
python3 summarize.py $D/raw > $D/raw/summary.csv 2>&1; cat $D/raw/summary.csv
python3 gate_check.py session $D/raw --clkmin $CHANGE/results/orin/clkmin.json --require-logs > $D/session-check.txt 2>&1; tail -2 $D/session-check.txt
step ./sdpa_err.sh sdpa-error3 $BT stock "parent=${PENV// /,}" "fused1=${C1// /,}" "fused2=${C2// /,}"
step ./stage.sh s10-c1g $PB "$PENV" $BT "$C1" "candidate 1 on the builds with the golden's glslc: orin-fused1 on topic4p against parentg with the parent environment"
step ./gate_sdpa.sh s10-c1g "$C1"
cat $A/stage/s10-c1g/gate.done 2>/dev/null
ok s10-c1g || { echo "CHAIN_STOPPED: s10-c1g is not accepted"; exit 3; }
B=$A/build/$BT/bundle/test_llama_microbench; O=$A/raw/c2g-pre; mkdir -p $O
for tier in all extended full peaked fused; do
  [[ -s $O/$tier.log ]] && continue
  cool_start 120; echo "== $(date -u +%FT%TZ) c2g-pre $tier"
  env $C2 ./gl.sh $B --sdpa-correctness-only --sdpa-tier=$tier > $O/$tier.log 2>&1; rc=$?
  echo "c2g-pre $tier rc=$rc cases=$(grep -c '^\[sdpa-correctness\] .* S=' $O/$tier.log) passed=$(grep -c '^\[sdpa-correctness\] .* PASSED$' $O/$tier.log)" | tee -a $O/summary.txt
  [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }
done
F=$C1; ARM=fused1; G=""
if grep -q 'FAILED' $O/*.log || [[ $(grep -c 'rc=0 ' $O/summary.txt) != 5 ]]; then echo "c2g-pre has a case that is not PASSED; candidate 2 is not gated"; grep -h 'FAILED' $O/*.log | cut -c1-300 | head -20
else
  step ./stage.sh s11-c2g $BT "$C1" $BT "$C2" "candidate 2 on topic4p: orin-fused2 against candidate 1 (orin-fused1)"
  step ./gate_sdpa.sh s11-c2g "$C2"
  cat $A/stage/s11-c2g/gate.done 2>/dev/null
  G=$(sed -n 's/^geomean over 6 cells: \([0-9.]*\) .*/\1/p' $A/stage/s11-c2g/raw/summary.csv 2>/dev/null)
  if ok s11-c2g && [[ -n $G ]] && awk -v g="$G" 'BEGIN {exit !(g >= 1.02)}'; then F=$C2; ARM=fused2; fi
fi
echo "final stack: [$F] (s11-c2g: $(cut -c1-60 $A/stage/s11-c2g/gate.done 2>/dev/null), geomean ratio ${G:-none})" | tee $A/FINALG.txt
step env PARENT_CTL_NAME=s0gn-noenv ./noenv_verify.sh $BT s12n-noenv
if [[ $ARM == fused2 ]]; then
  step ./stage.sh s12-finalg $PB "$PENV" $BT "$F" "final stack (orin-fused2) on topic4p against parentg with the parent environment: full gate"
  step ./gate_sdpa.sh s12-finalg "$F"
  ok s12-finalg || { echo "CHAIN_STOPPED: s12-finalg is not accepted"; exit 3; }
fi
step ./stage.sh s13-pristineg $PB "" $BT "$F" "final stack ($F) on topic4p against the pristine state: parentg with no environment"
step ./timed.sh s13-pristineg "$F"
step ./probe_run.sh $PB parentg-default $PENV
step ./probe_run.sh $BT finalg-default $F
step ./probe_run.sh $PB parentg-tiled $PENV ET_VK_FORCE_TILED_LINEAR=1
step ./probe_run.sh $BT finalg-tiled $F ET_VK_FORCE_TILED_LINEAR=1
P=$A/probe; O=$P/finalg-fused; mkdir -p $O
python3 probe_position.py $P/broad $O parentg-default parentg-tiled finalg-default finalg-tiled > $O/differing-items.txt; echo "position rc=$? $(cat $O/differing-items.txt | tr '\n' ';')"
python3 probe_compare.py $P/broad parentg-default parentg-tiled finalg-default finalg-tiled > $O/compare.csv; echo "compare rc=$? $(tail -1 $O/compare.csv)"
cp $A/raw/sdpa-error3/parent.txt $O/sdpa-error-parent.txt; cp $A/raw/sdpa-error3/$ARM.txt $O/sdpa-error-candidate.txt; cp $A/raw/sdpa-error3/stock.txt $O/sdpa-error-stock.txt
printf '%s\n' $F > $O/cand.env
mapfile -t ITEMS < <(grep . $O/differing-items.txt)
python3 ref_error_rule.py $O finalg-fused $O/cand.env $O/sdpa-error-parent.txt $O/sdpa-error-candidate.txt $O/compare.csv "${ITEMS[@]}" > $O/reference-error-rule.txt; echo "rule rc=$? $(tail -1 $O/reference-error-rule.txt)"
echo CHAIN_DONE
