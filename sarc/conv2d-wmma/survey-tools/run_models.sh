#!/bin/sh
# run_models.sh LOCK NAME...: runs on the device, in the directory holding executor_runner, its Vulkan runtime
# library, models/NAME.pte and models/MODEL-input.bin. For each NAME (MODEL-PRECISION): 12 executions of the
# traced executor_runner on Vulkan device 0, ETDump to results/NAME.etdp, outputs results/NAME-output-<i>.bin,
# log results/NAME.log, exit status results/NAME.exit, wall-clock start/end in results/NAME.time.
# LOCK: gpu-lab lock file to hold (flock) for each run, or "-" for none (phones). One run at a time.
set -u
LOCK=$1; shift
mkdir -p results
export LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ETVK_DEVICE_INDEX=0
for name in "$@"; do
  model=${name%%-*}
  set -- ./executor_runner --model_path="models/$name.pte" --inputs="models/$model-input.bin" \
    --output_file="results/$name-output" --etdump_path="results/$name.etdp" \
    --num_executions=12 --print_output=none
  start=$(date +%s)
  status=0
  if [ "$LOCK" = "-" ]; then "$@" > "results/$name.log" 2>&1 || status=$?
  else flock "$LOCK" "$@" > "results/$name.log" 2>&1 || status=$?; fi
  printf '%s\n' "$status" > "results/$name.exit"
  printf '%s %s\n' "$start" "$(date +%s)" > "results/$name.time"
  printf '%s status=%s\n' "$name" "$status"
done
