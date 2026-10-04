#!/bin/bash
# parent_verify.sh <parent build tag>: the control every candidate gate is compared with. Stages the parent
# binaries alone in stage/s0-parent-verify and runs the unmodified sarc/tools/verify.sh on them with no
# environment, plus one SDPA correctness pass per tier (recorded only: this device has no SDPA rows, so the
# parent dispatches the upstream kernels and the test's "coopmat fired" condition cannot hold).
source "$(dirname "$0")/common.sh"; source $TOOLS/gatelib.sh; PB=$1; D=$PARENT_CTL
[[ -e $D/verify.out ]] && { echo "parent control exists" >&2; exit 2; }
L=$A/build/$PB/llama/examples/models/llama/llama_main; SO=$(find $A/build/$PB/llama -name libllama_runner.so 2>/dev/null | head -1)
need $L "${SO:-$A/build/$PB/libllama_runner.so}" $A/build/$PB/tests/test_llama_microbench $TOOLS/r1304.txt
mkdir -p $D/sdpa-correctness; cp -f $L $SO $A/build/$PB/tests/test_llama_microbench $KIT/prompts/prompt_*.txt $TOOLS/r1304.txt $D/
{ echo "control: parent build/$PB, no environment"; grep '^commit \|^tree-sha256' $A/build/$PB.src.txt; sha256sum $D/llama_main $D/libllama_runner.so $D/test_llama_microbench $D/*.txt; } > $D/STAGE.md
cool_start 50 300
for tier in extended full; do
  $TOOLS/gl.sh $D/test_llama_microbench --sdpa-correctness-only --sdpa-tier=$tier > $D/sdpa-correctness/table-$tier-r1.log 2>&1
  rc=$?; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && finish GATE_ABORTED "sdpa control $tier rc=$rc" $rc
  echo "table $tier r1 rc=$rc cases=$(grep -c '^\[sdpa-correctness\]' $D/sdpa-correctness/table-$tier-r1.log) nonzero-mismatch=$(grep '^\[sdpa-correctness\]' $D/sdpa-correctness/table-$tier-r1.log | grep -vc ' mismatches=0/')" >> $D/sdpa-correctness/summary.txt
done
step verify run_verify ""
step verify-check python3 $TOOLS/gate_check.py verify $D/verify.out $D/verify.out > $D/verify-check.txt 2>&1
finish GATE_ACCEPTED "parent control complete" 0
