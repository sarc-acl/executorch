#!/bin/bash
# parent_verify.sh <parent build tag> ["<env>"]: the control every candidate gate is compared with. Stages the
# parent binaries alone in the control's stage directory (s0-parent-verify, or PARENT_CTL_NAME) and runs the
# unmodified sarc/tools/verify.sh on them with the given environment (this campaign's parent is the tuned stack;
# empty = nothing selected), plus one SDPA correctness pass per tier with the same environment (recorded only).
source "$(dirname "$0")/common.sh"; source $TOOLS/gatelib.sh; PB=$1; PENV=${2:-}; D=$PARENT_CTL
[[ -e $D/verify.out ]] && { echo "parent control exists" >&2; exit 2; }
L=$A/build/$PB/bundle/llama_main; SO=$A/build/$PB/bundle/libllama_runner.so
need $L $SO $A/build/$PB/bundle/test_llama_microbench $TOOLS/r1304.txt
mkdir -p $D/sdpa-correctness; stage_verify_runner $D $L; cp -f $SO $A/build/$PB/bundle/test_llama_microbench $KIT/prompts/prompt_*.txt $TOOLS/r1304.txt $D/
{ echo "control: parent build/$PB, environment [$PENV]"; grep '^commit \|^tree-sha256' $A/build/$PB.src.txt; sha256sum $D/verify-bin/llama_main $D/llama_main $D/libllama_runner.so $D/test_llama_microbench $D/*.txt; } > $D/STAGE.md
cool_start 300
for tier in all extended full; do
  env $PENV $TOOLS/gl.sh $D/test_llama_microbench --sdpa-correctness-only --sdpa-tier=$tier > $D/sdpa-correctness/table-$tier-r1.log 2>&1
  rc=$?; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && finish GATE_ABORTED "sdpa control $tier rc=$rc" $rc
  echo "table $tier r1 rc=$rc cases=$(grep -c '^\[sdpa-correctness\]' $D/sdpa-correctness/table-$tier-r1.log) nonzero-mismatch=$(grep '^\[sdpa-correctness\]' $D/sdpa-correctness/table-$tier-r1.log | grep -vc ' mismatches=0/')" >> $D/sdpa-correctness/summary.txt
done
step verify run_verify "$PENV"
step verify-check python3 $TOOLS/gate_check.py verify $D $D > $D/verify-check.txt 2>&1
finish GATE_ACCEPTED "parent control complete" 0
