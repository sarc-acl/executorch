#!/bin/bash
# final.sh <profile>: closing measurements, one GPU job at a time.
#   1. control: unmodified sarc/tools/verify.sh on the pristine dev/1.5 build (no environment), same options as
#      the candidate gates, and 3 passes of --sdpa-correctness-only, so the gate status of the parent is on record
#      next to the candidates' (stage/s0-parent-verify);
#   2. direct comparison: pristine dev/1.5 build (parent) vs the topic build with the final profile (cand), the
#      same 5-run session as every gate (stage/s7-final), then warm traces of both arms.
A=$HOME/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-03; P=$1; ET=$HOME/hmz-sarc/executorch
S0=$A/stage/s0-parent-verify; mkdir -p $S0
cp -f $A/build/parent/llama/examples/models/llama/llama_main $(find $A/build/parent/llama -name libllama_runner.so | head -1) $A/build/parent/tests/test_llama_microbench $S0/
cp -f $ET/openspec/changes/sarc-1.5-e2e-benchmark/kit/prompts/prompt_*.txt $A/tools/r1304.txt $S0/
{ echo "control: pristine dev/1.5 build (build/parent, ef079ac41), no environment"; sha256sum $S0/llama_main $S0/test_llama_microbench; } > $S0/STAGE.md
t0=$SECONDS; while (( $(cat /sys/class/hwmon/hwmon2/temp1_input) > 48000 && SECONDS - t0 < 300 )); do sleep 5; done
mkdir -p $S0/sdpa-correctness
for i in 1 2 3; do $A/tools/gl.sh $S0/test_llama_microbench --sdpa-correctness-only > $S0/sdpa-correctness/table-r$i.log 2>&1; done
$ET/sarc/tools/verify.sh --dir $S0 --lock 00000000-c400-0000-0000-000000000000 --models 1b,3b,8b --schemes 4w,8da4w --pdiff --out verify > $S0/verify.out 2>&1
echo "VERIFY_DONE rc=$?" >> $S0/verify.out
$A/tools/stage.sh s7-final parent "" topic "ET_VK_SARC_DEV_PROFILE=$P" "final: pristine dev/1.5 build vs the topic build $(cat $A/build/topic.commit) with profile $P" > $A/logs/stage-s7.out 2>&1
$A/tools/session.sh s7-final > $A/stage/s7-final/e2e5.out 2>&1
$A/tools/trace.sh s7-final 1b,3b,8b 4w,8da4w "parent cand" > $A/stage/s7-final/trace.out 2>&1
python3 $A/tools/summarize.py $A/stage/s7-final/raw > $A/stage/s7-final/raw/summary.csv 2>&1
echo FINAL_DONE > $A/stage/s7-final/final.done
