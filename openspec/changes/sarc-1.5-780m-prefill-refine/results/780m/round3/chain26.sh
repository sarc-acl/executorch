#!/bin/bash
# chain26.sh (round 3, 2026-10-08): the measurements, on build head3 (chain25.sh), one GPU job at a time.
#   A. gate2.sh r3a-fused3sb with the SDPA tiers: timed session from a cool start (parent = c11 with the fused3
#      pair by variable, candidate = c11 as committed = fused3sb; same binary), tiers all / extended / full 12
#      passes each with the candidate environment, unmodified verify.sh with it, warm traces of both arms
#   B. r3b-final-dev15: timed session from a cool start (parent = head3 with no environment, candidate =
#      ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=780m-final; same binary), then unmodified verify.sh and one
#      pass of each SDPA tier with the candidate environment (for the dispatched kernel names)
# The six model files are read into the page cache before each part (owner decision 2026-10-06), for both arms
# alike. Each session is one hold unit. No profiler variable anywhere (gl.sh refuses them).
A=/home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-08; ET=/home/doremy/hmz-sarc/executorch; T=$A/tools
HW=/sys/class/hwmon/hwmon2; LOCK=00000000-c400-0000-0000-000000000000
env | grep -E '^(MESA_VK_TRACE|RADV_THREAD_TRACE|RADV_PROFILE_PSTATE|INTEL_MEASURE|MESA_GPU_TRACES)' && { echo "profiler variable set, refusing"; exit 97; }
export ART780M=$A; cd $A; eval "$($T/hold.sh vars)"
st() { echo "$(date -u +%FT%TZ) $*" >> logs/chain26.status; }
grep -q ' DONE$' logs/chain25.status || { st "REFUSED: chain25 not done"; exit 2; }
C11="ET_VK_SARC_DEV_PROFILE=780m-refine3 ET_VK_SARC_780M_PROFILE=c11"
FIN="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=780m-final"
pagecache() { cat /mnt/linux-share/models/llama-3.*/exported/*_vulkan_4w.pte /mnt/linux-share/models/llama-3.*/exported/*_vulkan_8da4w.pte > /dev/null; }
st "start"

pagecache; st "A: gate2.sh r3a-fused3sb started"
$T/gate2.sh r3a-fused3sb "$C11" sdpa > logs/gate-r3a.out 2>&1
st "A: gate done ($(cat stage/r3a-fused3sb/gate.done 2>/dev/null)); $(tail -1 stage/r3a-fused3sb/raw/summary.csv 2>/dev/null)"

S=$A/stage/r3b-final-dev15; pagecache
while :; do
  $T/hold.sh wait "chain26 r3b-final-dev15: timed session"
  t0=$SECONDS; while (( $(cat $HW/temp1_input) > 43000 && SECONDS - t0 < 1800 )); do sleep 10; done
  [[ -e $HOLD ]] || break
done
echo "prestart_temp=$(( $(cat $HW/temp1_input) / 1000 )) waited=$((SECONDS - t0))s $(date -u +%FT%TZ)" > $S/prestart.txt
st "B: session started"
$T/e2e5.sh --stage $S --out raw --lock $LOCK > $S/e2e5.out 2>&1
python3 $T/summarize.py $S/raw > $S/raw/summary.csv 2>&1
st "B: session done; $(tail -1 $S/raw/summary.csv)"
env $FIN $T/hold.sh run "verify.sh r3b-final-dev15" $ET/sarc/tools/verify.sh --dir $S --lock $LOCK \
  --models 1b,3b,8b --schemes 4w,8da4w --pdiff --out verify > $S/verify.out 2>&1
echo "VERIFY_DONE rc=$?" >> $S/verify.out
O=$S/sdpa-correctness; mkdir -p $O; : > $O/summary.txt
for tier in all extended full; do
  t0=$SECONDS; while (( $(cat $HW/temp1_input) > 55000 && SECONDS - t0 < 300 )); do sleep 5; done
  env $FIN $T/gl.sh $S/test_llama_microbench --sdpa-correctness-only --sdpa-tier=$tier > $O/cand-$tier-r1.log 2>&1
  echo "cand $tier r1 rc=$? passed=$(grep -c 'PASSED' $O/cand-$tier-r1.log) failed=$(grep -c 'FAILED' $O/cand-$tier-r1.log) pairing_not_ok=$(grep 'sdpa-kernels' $O/cand-$tier-r1.log | grep -vc 'pairing=ok')" >> $O/summary.txt
done
st "DONE"
