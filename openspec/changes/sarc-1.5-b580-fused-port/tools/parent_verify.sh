#!/bin/bash
# parent_verify.sh <name> <build tag> "<env>": a control snapshot, stage/<name>. Unmodified sarc/tools/verify.sh
# on build/<build tag> with the given environment (space-separated KEY=VALUE, may be empty) and the same options
# as every candidate gate, and one pass of each SDPA correctness tier, so that the parent's gate status is on
# record for the line-by-line comparison.
#   parent_verify.sh s0-parent-verify parent "$PARENT_ENV"   the snapshot every candidate is compared with
#   parent_verify.sh s0-parent-noenv parent ""               the no-environment snapshot of the hook conditions (D4)
# A foreign GPU process ends it at once: gate.done = GATE_ABORTED, exit 76, no gate_check.py verdict. The
# aborted directory stays as evidence; move it to superseded/ before running the control again.
. "$(dirname "$(readlink -f "$0")")/host.sh"; N=${1:?name}; B=${2:?build tag}; ENVS=${3:-}; D=$A/stage/$N
hold_wait "control $N"; gpu_shared || exit 75
UNALIGNED=${B580_UNALIGNED:-$HOME/.cache/et-e2e/sarc15-r4/r1304.txt}
grep -qx BUILD_BOTH_OK $A/build/$B.src.txt || { echo "build/$B is not a successful build" >&2; exit 2; }
[[ -e $D ]] && { echo "$D already exists" >&2; exit 2; }
mkdir -p $D && cp -f $A/build/$B/llama/examples/models/llama/llama_main $(find $A/build/$B/llama -name libllama_runner.so | head -1) \
  $A/build/$B/tests/test_llama_microbench $ET/openspec/changes/sarc-1.5-e2e-benchmark/kit/prompts/prompt_*.txt $UNALIGNED $D/ || exit 2
{ echo "control: build/$B ($(sed -n 's/^commit=//p' $A/build/$B.src.txt)), environment [$ENVS]"; sha256sum $D/llama_main $D/test_llama_microbench $D/*.txt; } > $D/STAGE.md
$TOOLS/sdpa_passes.sh $N table 1 "$ENVS" > $D/sdpa.out 2>&1; [[ $? == 76 ]] && { echo "GATE_ABORTED foreign GPU process during the SDPA passes" | tee $D/gate.done; exit 76; }
cool_start
guarded $D/verify.others env $ENVS $ET/sarc/tools/verify.sh --dir $D --lock $LOCK --models 1b,3b,8b --schemes 4w,8da4w --pdiff --out verify > $D/verify.out 2>&1
rc=$?; echo "VERIFY_DONE rc=$rc" >> $D/verify.out
[[ $rc == 76 ]] && { echo "GATE_ABORTED foreign GPU process during verify.sh" | tee $D/gate.done; exit 76; }
python3 $TOOLS/gate_check.py $D --control > $D/gate.txt 2>&1; rc=$?
tail -1 $D/gate.txt > $D/gate.done; cat $D/gate.txt; exit $rc
