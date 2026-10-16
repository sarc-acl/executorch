#!/bin/bash
# matrix_dispatch.sh <session> <build tag>: collect actual model dispatch names with the unchanged stage environments.
set -euo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"
S=$A/stage/${1:?session}; TAG=${2:?build tag}
python3 "$TOOLS/input_provenance.py" "$S/INPUTS.json" "$XE2_MODEL_ROOT"
gpu_begin excl "dispatch $1"
exec 9>>"$HOME/.cache/gpu-lab/lock-$LOCK"
flock -w 900 9 || exit 75
[[ ! -e $A/DEVICE_LOST ]]
FLATC=$A/build/$TAG/llama/third-party/flatc_ep/bin/flatc
SCHEMA=$A/src/$TAG/executorch/devtools/etdump/etdump_schema_flatcc.fbs
D=$S/dispatch; mkdir -p "$D"
declare -A STEM=([1b]=llama-3.2-1b:llama3_2-1b [3b]=llama-3.2-3b:llama3_2-3b [8b]=llama-3.1-8b:llama3_1-8b)
for model in 1b 3b 8b; do
  IFS=: read -r MD ST <<< "${STEM[$model]}"
  for scheme in 4w 8da4w; do
    PTE=$XE2_MODEL_ROOT/$MD/exported/${ST}_vulkan_$scheme.pte
    cat "$PTE" > /dev/null
    for arm in parent cand; do
      foreign_wait "dispatch $model $scheme $arm"
      cool_start
      B=$S/$arm-traced; mapfile -t ENVS < "$B/env"
      F=$D/$model-$scheme-$arm
      set +e
      guarded "$F.others" env "${ENVS[@]}" LD_LIBRARY_PATH=$B timeout 1800 "$B/llama_main" \
        --model_path "$PTE" --tokenizer_path "$XE2_MODEL_ROOT/$MD/original/tokenizer.model" \
        --prompt_file "$S/prompt_2048.txt" --max_new_tokens 1 --temperature 0 --warmup \
        --etdump_path "$F.etdp" > "$F.log" 2>&1 9>&-
      rc=$?
      set -e
      echo "$model,$scheme,$arm,$rc" >> "$D/rc.csv"
      if rg -q 'VK_ERROR_DEVICE_LOST|ErrorDeviceLost' "$F.log"; then
        echo "Device loss in dispatch $1" > "$A/DEVICE_LOST"
        exit 77
      fi
      [[ $rc == 0 ]]
      grep -q '"prompt_tokens":2048' "$F.log"
      [[ -s $F.etdp ]]
      "$FLATC" --json --strict-json --raw-binary --size-prefixed -o "$D" "$SCHEMA" -- "$F.etdp"
    done
  done
done
python3 "$TOOLS/dispatch_names.py" "$D"
date -u +%FT%TZ > "$D/dispatch.done"
