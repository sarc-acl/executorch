#!/bin/bash
# matrix_verify.sh <session> <build tag> <environment file>: the unchanged six-cell gate, without extra attention tiers.
set -euo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"
[[ ! -e $A/DEVICE_LOST ]] || { echo "DEVICE_LOST blocks all later jobs" >&2; exit 77; }
NAME=${1:?session}; TAG=${2:?build tag}; ENVFILE=${3:?environment file}
D=$A/stage/$NAME
[[ ! -e $D/verify.out ]] || { echo "verification already exists: $NAME" >&2; exit 2; }
grep -qx BUILD_BOTH_OK "$A/build/$TAG.src.txt"
gpu_begin excl "verify $NAME"
foreign_wait "verify $NAME"
[[ -r $FREQ/act_freq && -r $HW/temp2_input ]] || { echo DEVICE_UNAVAILABLE >&2; exit 77; }
L=$A/build/$TAG/llama/examples/models/llama
LIB=$(find "$A/build/$TAG/llama" -name libllama_runner.so -print -quit)
if [[ -e $D ]]; then
  [[ -f $D/STAGE.md && -f $D/cand/env ]]
  cmp "$ENVFILE" "$D/cand/env"
  cmp "$L/llama_main" "$D/llama_main"
  cmp "$LIB" "$D/libllama_runner.so"
  cmp "$A/build/$TAG/tests/test_llama_microbench" "$D/test_llama_microbench"
else
  mkdir -p "$D"
  cp "$L/llama_main" "$A/build/$TAG/tests/test_llama_microbench" "$D/"
  cp "$LIB" "$D/"
  cp "$ET"/openspec/changes/sarc-1.5-e2e-benchmark/kit/prompts/prompt_*.txt "$D/"
  cp "${XE2_UNALIGNED:-$HOME/.cache/et-e2e/sarc15-r4/r1304.txt}" "$D/"
fi
cp "$ENVFILE" "$D/env"
if [[ -e $D/INPUTS.json ]]; then
  cmp "$A/preflight/revised-input-admission.json" "$D/INPUTS.json"
else
  cp "$A/preflight/revised-input-admission.json" "$D/INPUTS.json"
fi
mapfile -t ENVS < "$D/env"
{
  echo "build=$TAG"
  sed -n '/^commit=/p' "$A/build/$TAG.src.txt"
  sha256sum "$D/llama_main" "$D/test_llama_microbench" "$D/libllama_runner.so" "$D/env" "$ET/sarc/tools/verify.sh"
} > "$D/VERIFY-STAGE.md"
# All six models fit in RAM; fill the cache for the unchanged gate's internal cell loop.
for p in "$XE2_MODEL_ROOT"/llama-3.{1-8b,2-1b,2-3b}/exported/*_vulkan_{4w,8da4w}.pte; do
  cat "$p" > /dev/null
  fincore -n -b -o RES,SIZE "$p" >> "$D/model-cache.txt"
done
cool_start
set +e
guarded "$D/verify.others" env "${ENVS[@]}" "$ET/sarc/tools/verify.sh" --dir "$D" --lock "$LOCK" \
  --models 1b,3b,8b --schemes 4w,8da4w --pdiff --model-root "$XE2_MODEL_ROOT" --device-index 0 \
  > "$D/verify.out" 2>&1
rc=$?
set -e
echo "VERIFY_DONE rc=$rc" >> "$D/verify.out"
if rg -l 'VK_ERROR_DEVICE_LOST|ErrorDeviceLost' "$D/verify" > "$D/device-loss.txt"; then
  echo "Device loss in $NAME; no later job may start" > "$A/DEVICE_LOST"
  exit 77
fi
exit "$rc"
