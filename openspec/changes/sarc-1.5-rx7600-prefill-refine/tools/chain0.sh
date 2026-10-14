#!/bin/bash
# chain0.sh: A/A session aa2 (parent build in both arms, 7 repeats; its parent arm is the baseline check), then the
# snapshot s0-parent-verify (unmodified verify.sh on the parent build) and the SDPA tiers of the parent, 1 pass each.
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/chain0.status; }
st start
$T/stage.sh aa2 parent "ET_VK_SARC_UNVERIFIED=1" parent "ET_VK_SARC_UNVERIFIED=1" "A/A: the parent build f5f1bf10c in both arms" > $A/logs/stage-aa2.out 2>&1
st "aa2 staged; timed session starts"
$T/e2e5.sh --stage $A/stage/aa2 --out raw --reps 7 --extra 3 > $A/stage/aa2/e2e5.out 2>&1
/usr/bin/python3 $T/summarize.py $A/stage/aa2/raw 5 > $A/stage/aa2/raw/summary.csv 2>&1
/usr/bin/python3 $T/summarize.py $A/stage/aa2/raw 7 > $A/stage/aa2/raw/summary7.csv 2>&1
st "aa2 done"
S0=$A/stage/s0-parent-verify; mkdir -p $S0
L=$A/build/rx7600/parent; cp -f $L/llama/examples/models/llama/llama_main $(find $L/llama -name libllama_runner.so | head -1) $L/tests/test_llama_microbench $S0/
cp -f $ET/openspec/changes/sarc-1.5-e2e-benchmark/kit/prompts/prompt_*.txt $S0/
{ echo "s0-parent-verify: parent build (tag parent, $(cat $A/src/rx7600/parent/COMMIT)), env ET_VK_SARC_UNVERIFIED=1, no profile"; sha256sum $S0/llama_main $S0/libllama_runner.so $S0/test_llama_microbench; } > $S0/STAGE.md
st "s0 verify starts"
$T/verify_stage.sh s0-parent-verify "ET_VK_SARC_UNVERIFIED=1"
st "s0 verify done"
$T/sdpa_tiers.sh s0-parent-verify "ET_VK_SARC_UNVERIFIED=1" 1 "all extended full" table
st "s0 sdpa tiers done"; st CHAIN0_DONE
