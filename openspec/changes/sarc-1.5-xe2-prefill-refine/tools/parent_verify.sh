#!/bin/bash
# parent_verify.sh: the parent control, stage/s0-parent-verify. Unmodified sarc/tools/verify.sh on the pristine
# parent build (build/parent, no environment) with the same options as every candidate gate, and one pass of
# each SDPA correctness tier with the table kernels, so the parent's gate status is on record for comparison.
. "$(dirname "$(readlink -f "$0")")/host.sh"; D=$A/stage/s0-parent-verify
UNALIGNED=${XE2_UNALIGNED:-$HOME/.cache/et-e2e/sarc15-r4/r1304.txt}
grep -qx BUILD_BOTH_OK $A/build/parent.src.txt || { echo "build/parent is not a successful build" >&2; exit 2; }
[[ -e $D ]] && { echo "$D already exists" >&2; exit 2; }
mkdir -p $D && cp -f $A/build/parent/llama/examples/models/llama/llama_main $(find $A/build/parent/llama -name libllama_runner.so | head -1) \
  $A/build/parent/tests/test_llama_microbench $ET/openspec/changes/sarc-1.5-e2e-benchmark/kit/prompts/prompt_*.txt $UNALIGNED $D/ || exit 2
{ echo "control: pristine parent build (build/parent, $(sed -n 's/^commit=//p' $A/build/parent.src.txt)), no environment"; sha256sum $D/llama_main $D/test_llama_microbench $D/*.txt; } > $D/STAGE.md
$TOOLS/sdpa_passes.sh s0-parent-verify table 1 "" > $D/sdpa.out 2>&1; [[ $? == 76 ]] && { echo "ABORTED foreign GPU process" | tee $D/gate.done; exit 76; }
cool_start
guarded $D/verify.others $ET/sarc/tools/verify.sh --dir $D --lock $LOCK --models 1b,3b,8b --schemes 4w,8da4w --pdiff --out verify > $D/verify.out 2>&1
rc=$?; echo "VERIFY_DONE rc=$rc" >> $D/verify.out
python3 $TOOLS/gate_check.py $D --control > $D/gate.txt 2>&1; rc=$?
tail -1 $D/gate.txt > $D/gate.done; cat $D/gate.txt; exit $rc
