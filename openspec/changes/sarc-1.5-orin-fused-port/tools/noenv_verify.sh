#!/bin/bash
# noenv_verify.sh <build tag> <session name>: device side. The control the owner's hook decision of 2026-10-05 asks
# for: the unmodified sarc/tools/verify.sh on a build of the branch (which contains the release-zone hook and the
# whole dev zone) with NOTHING selected (no environment), compared item by item and case by case with the parent
# control (gate_check.py verify: 0 findings = every device-visible item equals the pristine parent's). verify.out
# of both is also compared line by line with the rates removed (verify-lines.diff, expected empty).
source "$(dirname "$0")/common.sh"; source $TOOLS/gatelib.sh; BT=$1; D=$A/stage/$2
[[ $SIDE == device ]] || { echo "device only" >&2; exit 2; }
[[ -e $D/verify.out ]] && { echo "session $2 exists" >&2; exit 2; }
L=$A/build/$BT/bundle/llama_main; SO=$A/build/$BT/bundle/libllama_runner.so
need $L $SO $A/build/$BT/bundle/test_llama_microbench $TOOLS/r1304.txt $PARENT_CTL/verify.out
mkdir -p $D; stage_verify_runner $D $L; cp -f $SO $A/build/$BT/bundle/test_llama_microbench $KIT/prompts/prompt_*.txt $TOOLS/r1304.txt $D/
{ echo "control: build/$BT with no environment (nothing selected), against the parent control"; grep '^commit \|^tree-sha256\|^local-patch' $A/build/$BT.src.txt; sha256sum $D/verify-bin/llama_main $D/llama_main $D/libllama_runner.so $D/test_llama_microbench $D/*.txt; } > $D/STAGE.md
cool_start 300
step verify run_verify ""
strip() { sed -E 's/prefill_tok_s=[0-9.]+/prefill_tok_s=X/; s/decode_tok_s=[0-9.]+/decode_tok_s=X/' "$1"; }
diff <(strip $PARENT_CTL/verify.out) <(strip $D/verify.out) > $D/verify-lines.diff; echo "verify.out against the parent control, rates removed: $(wc -l < $D/verify-lines.diff) differing lines" | tee $D/verify-lines.txt
step verify-check python3 $TOOLS/gate_check.py verify $D $PARENT_CTL > $D/verify-check.txt 2>&1
[[ -s $D/verify-lines.diff ]] && finish GATE_REJECTED "verify.out differs from the parent control" 1
finish GATE_ACCEPTED "nothing selected: verify.sh equals the parent control (0 findings, 0 differing lines)" 0
