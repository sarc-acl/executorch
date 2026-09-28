#!/bin/bash
# optin_check.sh --lock UUID [--flat-models P]: 1B 4w and 8da4w prefill with the previous build
# with and without its opt-in env, to show the env actually switched the model path to WMMA.
LOCK=""; FLAT=""; MROOT=/mnt/linux-share/models
while [[ $# -gt 0 ]]; do case $1 in --lock) LOCK=$2; shift ;; --flat-models) FLAT=$2; shift ;; esac; shift; done
D=$(cd "$(dirname "$0")" && pwd); cd "$D"; exec 9>>"$HOME/.cache/gpu-lab/lock-$LOCK"; flock -w 900 9 || exit 75
export ETVK_DEVICE_INDEX=0; mkdir -p optin
for q in 4w 8da4w; do
  if [[ -n $FLAT ]]; then P=$FLAT/llama3_2-1b_vulkan_$q.pte; T=$FLAT/tokenizer.model
  else P=$MROOT/llama-3.2-1b/exported/llama3_2-1b_vulkan_$q.pte; T=$MROOT/llama-3.2-1b/original/tokenizer.model; fi
  for mode in env noenv; do
    E=(); [[ $mode == env && -f stock/env ]] && mapfile -t E < stock/env
    env "${E[@]}" LD_LIBRARY_PATH=$D/stock ./stock/llama_main --model_path $P --tokenizer_path $T --prompt_file prompt_real_2048.txt \
      --max_new_tokens 1 --temperature 0 --warmup < /dev/null > optin/1b-$q-$mode.log 2>&1 9>&-
    echo "optin 1b $q previous-build $mode tok_s=$(grep -o '"prefill_token_per_sec":[0-9.]*' optin/1b-$q-$mode.log | cut -d: -f2)"
  done
done
