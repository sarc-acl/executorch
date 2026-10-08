#!/bin/bash
# chain28.sh (round 3, replacement validation on build head4, owner decision 2026-10-08 22:55 UTC): the
# measurements of chain26.sh again, one GPU job at a time, with the corrected tools (e2e5.sh and trace.sh read
# the model file before each cell and record its residency; e2e5.sh also compares the next token on r1304.txt).
#   A. r3a-fused3sb-head4: timed session from a cool start (parent = c11 with the fused3 pair by variable,
#      candidate = c11 as committed = fused3sb; same binary), SDPA tiers all / extended / full 12 passes each with
#      the candidate environment and one control pass each without it, the SDPA perf suite, unmodified verify.sh
#      with the candidate environment, warm traces of both arms (the steps of gate2.sh ... sdpa, in its order)
#   B. r3b-final-dev15-head4: timed session from a cool start (parent = head4 with no environment, candidate =
#      ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=780m-final; same binary), then unmodified verify.sh and one
#      pass of each SDPA tier with the candidate environment
# verify.sh is not edited and is run once per item, as the snapshot was: the six model files are read first and
# their residency (fincore) is written before and after it (verify-residency.txt).
# Each session is one hold unit. No profiler variable anywhere (gl.sh refuses them).
A=/home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-08; ET=/home/doremy/hmz-sarc/executorch; T=$A/tools
HW=/sys/class/hwmon/hwmon2; LOCK=00000000-c400-0000-0000-000000000000
env | grep -E '^(MESA_VK_TRACE|RADV_THREAD_TRACE|RADV_PROFILE_PSTATE|INTEL_MEASURE|MESA_GPU_TRACES)' && { echo "profiler variable set, refusing"; exit 97; }
export ART780M=$A; cd $A; eval "$($T/hold.sh vars)"
st() { echo "$(date -u +%FT%TZ) $*" >> logs/chain28.status; }
grep -q ' DONE$' logs/chain27.status || { st "REFUSED: chain27 not done"; exit 2; }
C11="ET_VK_SARC_DEV_PROFILE=780m-refine3 ET_VK_SARC_780M_PROFILE=c11"
FIN="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=780m-final"
MODELS="/mnt/linux-share/models/llama-3.*/exported/*_vulkan_4w.pte /mnt/linux-share/models/llama-3.*/exported/*_vulkan_8da4w.pte"
session() {  # session <stage dir>
  local S=$1 t0
  while :; do
    $T/hold.sh wait "chain28 $(basename $S): timed session"
    t0=$SECONDS; while (( $(cat $HW/temp1_input) > 43000 && SECONDS - t0 < 1800 )); do sleep 10; done
    [[ -e $HOLD ]] || break
  done
  echo "prestart_temp=$(( $(cat $HW/temp1_input) / 1000 )) waited=$((SECONDS - t0))s $(date -u +%FT%TZ)" > $S/prestart.txt
  $T/e2e5.sh --stage $S --out raw --lock $LOCK > $S/e2e5.out 2>&1
  python3 $T/summarize.py $S/raw > $S/raw/summary.csv 2>&1
}
tiers() {  # tiers <stage dir> "<env>" <passes> [control]
  local S=$1 E=$2 N=$3 O=$1/sdpa-correctness B=$1/test_llama_microbench tier i t0; mkdir -p $O; : > $O/summary.txt
  for tier in all extended full; do
    for i in $(seq 1 $N); do
      t0=$SECONDS; while (( $(cat $HW/temp1_input) > 55000 && SECONDS - t0 < 300 )); do sleep 5; done
      env $E $T/gl.sh $B --sdpa-correctness-only --sdpa-tier=$tier > $O/cand-$tier-r$i.log 2>&1
      echo "cand $tier r$i rc=$? passed=$(grep -c 'PASSED' $O/cand-$tier-r$i.log) failed=$(grep -c 'FAILED' $O/cand-$tier-r$i.log) pairing_not_ok=$(grep 'sdpa-kernels' $O/cand-$tier-r$i.log | grep -vc 'pairing=ok')" >> $O/summary.txt
    done
    [[ -n ${4:-} ]] || continue
    $T/gl.sh $B --sdpa-correctness-only --sdpa-tier=$tier > $O/table-$tier-r1.log 2>&1
    echo "table $tier r1 rc=$? passed=$(grep -c 'PASSED' $O/table-$tier-r1.log) failed=$(grep -c 'FAILED' $O/table-$tier-r1.log)" >> $O/summary.txt
  done
}
verify() {  # verify <stage dir> "<env>"
  local S=$1 E=$2
  { echo "before the read $(date -u +%FT%TZ)"; fincore -b $MODELS; cat $MODELS > /dev/null; echo "after the read, before verify.sh $(date -u +%FT%TZ)"; fincore -b $MODELS; } > $S/verify-residency.txt 2>&1
  env $E $T/hold.sh run "verify.sh $(basename $S)" $ET/sarc/tools/verify.sh --dir $S --lock $LOCK \
    --models 1b,3b,8b --schemes 4w,8da4w --pdiff --out verify > $S/verify.out 2>&1
  echo "VERIFY_DONE rc=$?" >> $S/verify.out
  { echo "after verify.sh $(date -u +%FT%TZ)"; fincore -b $MODELS; } >> $S/verify-residency.txt 2>&1
}
st "start"

S=$A/stage/r3a-fused3sb-head4; st "A: session started"
session $S; st "A: session done; $(tail -1 $S/raw/summary.csv)"
tiers $S "$C11" 12 control
O=$S/sdpa-correctness
env $C11 $T/gl.sh $S/test_llama_microbench --sdpa --json-out=$O/perf-cand.json > $O/perf-cand.log 2>&1; echo "perf cand rc=$?" >> $O/summary.txt
$T/gl.sh $S/test_llama_microbench --sdpa --json-out=$O/perf-table.json > $O/perf-table.log 2>&1; echo "perf table rc=$?" >> $O/summary.txt
st "A: tiers done"
verify $S "$C11"; st "A: $(tail -1 $S/verify.out)"
$T/trace.sh r3a-fused3sb-head4 1b,3b,8b 4w,8da4w "parent cand" > $S/trace.out 2>&1
echo GATE_DONE > $S/gate.done; st "A: gate done"

S=$A/stage/r3b-final-dev15-head4; st "B: session started"
session $S; st "B: session done; $(tail -1 $S/raw/summary.csv)"
verify $S "$FIN"; st "B: $(tail -1 $S/verify.out)"
tiers $S "$FIN" 1
st "DONE"
