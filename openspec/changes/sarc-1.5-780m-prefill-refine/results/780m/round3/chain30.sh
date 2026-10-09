#!/bin/bash
# chain30.sh = chain29.sh started again (chain29 was stopped by the actor at 03:27 UTC: the shell that launched it stayed alive with a word of the guard pattern in its command line, the first 10 rows were invalid, superseded/r6-actor-shell-matched-guard). (round 3, owner decision 2026-10-09 02:50 UTC, option (b)): the timed sessions of items A and B
# again on build head4, on the staged binaries of chain27.sh (stage/r3a-fused3sb-head4, r3b-final-dev15-head4),
# with tools/e2e5.sh sampling throttle_status, the guard's process list and the readable DRM clients during
# each run and asking for 5 clock samples. Output in <stage>/raw-r6b; the sessions of chain28.sh stay in
# <stage>/raw. Timed runs only (--no-check): the gate, the tiers, the next-token runs and the byte comparison
# are not repeated. Each session is one hold unit. No profiler variable anywhere.
A=/home/doremy/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-08; T=$A/tools
HW=/sys/class/hwmon/hwmon2; LOCK=00000000-c400-0000-0000-000000000000
env | grep -E '^(MESA_VK_TRACE|RADV_THREAD_TRACE|RADV_PROFILE_PSTATE|INTEL_MEASURE|MESA_GPU_TRACES)' && { echo "profiler variable set, refusing"; exit 97; }
export ART780M=$A; cd $A; eval "$($T/hold.sh vars)"
st() { echo "$(date -u +%FT%TZ) $*" >> logs/chain30.status; }
[[ -e logs/chain30.status ]] && { echo "chain30 was started before; a rerun takes a new name"; exit 2; }
st "start"
for S in stage/r3a-fused3sb-head4 stage/r3b-final-dev15-head4; do
  for b in parent cand; do
    [[ $(sha256sum < $S/$b/llama_main | cut -c1-64) == 578935be47dba52300012f20d093fcaa921c214842bbe8e8bc3d0b68763c69b5 &&
       $(sha256sum < $S/$b/libllama_runner.so | cut -c1-64) == f94c058ae42d3b55bea2f7eabafccfeedae0ad06ed70b74d332ece6bcf12ca7c ]] || { st "REFUSED: $S/$b is not the head4 binary"; exit 2; }
  done
  [[ -e $S/raw-r6b ]] && { st "REFUSED: $S/raw-r6b exists"; exit 2; }
done
session() {  # session <stage dir>
  local S=$1 t0
  while :; do
    $T/hold.sh wait "chain30 $(basename $S): timed session"
    t0=$SECONDS; while (( $(cat $HW/temp1_input) > 43000 && SECONDS - t0 < 1800 )); do sleep 10; done
    [[ -e $HOLD ]] || break
  done
  echo "prestart_temp=$(( $(cat $HW/temp1_input) / 1000 )) waited=$((SECONDS - t0))s $(date -u +%FT%TZ) monitors_found=[$(pgrep -x -a nvtop | tr '\n' ';')]" > $S/prestart-r6b.txt
  $T/e2e5.sh --stage $S --out raw-r6b --lock $LOCK --no-check > $S/e2e5-r6b.out 2>&1
  python3 $T/summarize.py $S/raw-r6b > $S/raw-r6b/summary.csv 2>&1
}
S=$A/stage/r3a-fused3sb-head4; st "A: session started"
session $S; st "A: session done; $(tail -1 $S/raw-r6b/summary.csv)"
S=$A/stage/r3b-final-dev15-head4; st "B: session started"
session $S; st "B: session done; $(tail -1 $S/raw-r6b/summary.csv)"
st "DONE"
