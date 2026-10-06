#!/bin/bash
# r1_full.sh: passes 2..12 of --sdpa-tier=full with profile 780m-refine3, plus one control pass with the table kernels.
A=$HOME/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-03; B=$A/build/topic-r1/tests/test_llama_microbench; O=$A/stage/r1-sdpa-ext; cd $O
for i in $(seq 2 12); do
  t0=$SECONDS; while (( $(cat /sys/class/hwmon/hwmon2/temp1_input) > 55000 && SECONDS - t0 < 300 )); do sleep 5; done
  env ET_VK_SARC_DEV_PROFILE=780m-refine3 $A/tools/gl.sh $B --sdpa-correctness-only --sdpa-tier=full > cand-full-r$i.log 2>&1; echo "full r$i rc=$?" >> full.progress
done
$A/tools/gl.sh $B --sdpa-correctness-only --sdpa-tier=full > table-full-r1.log 2>&1; echo "table full r1 rc=$?" >> full.progress
echo FULL_DONE >> full.progress
