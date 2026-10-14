#!/bin/bash
# parent_verify.sh <parent build tag> [<session>=s0-parent-verify ["<env>"=PARENT_ENV]]: the snapshot every
# candidate gate is compared with. Stages the parent binaries alone in stage/<session> and runs the unmodified
# sarc/tools/verify.sh on them with the given environment, plus one SDPA correctness pass per tier (recorded).
# s0-parent-verify = the parent with its profile (4070ti-refine1); s0-pristine-verify = the same build with an
# empty environment, the reference of the release-zone hook control (hook_control.sh).
source "$(dirname "$0")/common.sh"; source $TOOLS/gatelib.sh; PB=$1; D=$A/stage/${2:-s0-parent-verify}; PE=${3-$PARENT_ENV}
[[ -e $D/verify.out ]] && { echo "parent control exists" >&2; exit 2; }
L=$A/build/$PB/llama/examples/models/llama/llama_main; SO=$(find $A/build/$PB/llama -name libllama_runner.so 2>/dev/null | head -1)
need $L "${SO:-$A/build/$PB/libllama_runner.so}" $A/build/$PB/tests/test_llama_microbench $TOOLS/r1304.txt
mkdir -p $D/sdpa-correctness $D/parent; : > $D/parent/env; for kv in $PE; do echo "$kv" >> $D/parent/env; done; stage_verify_runner $D $L; cp -f $SO $A/build/$PB/tests/test_llama_microbench $KIT/prompts/prompt_*.txt $TOOLS/r1304.txt $D/
{ echo "control: parent build/$PB, environment [$PE]"; grep '^commit \|^tree-sha256' $A/build/$PB.src.txt; sha256sum $D/verify-bin/llama_main $D/llama_main $D/libllama_runner.so $D/test_llama_microbench $D/*.txt; } > $D/STAGE.md
cool_start 50 300
for tier in extended full; do
  env $PE $TOOLS/gl.sh $D/test_llama_microbench --sdpa-correctness-only --sdpa-tier=$tier > $D/sdpa-correctness/table-$tier-r1.log 2>&1
  rc=$?; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && finish GATE_ABORTED "sdpa control $tier rc=$rc" $rc
  echo "table $tier r1 rc=$rc cases=$(grep -c '^\[sdpa-correctness\]' $D/sdpa-correctness/table-$tier-r1.log) nonzero-mismatch=$(grep '^\[sdpa-correctness\]' $D/sdpa-correctness/table-$tier-r1.log | grep -vc ' mismatches=0/')" >> $D/sdpa-correctness/summary.txt
done
step verify run_verify "$PE"
step verify-check python3 $TOOLS/gate_check.py verify $D $D > $D/verify-check.txt 2>&1
finish GATE_ACCEPTED "parent control complete" 0
