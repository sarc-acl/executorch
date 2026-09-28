#!/bin/bash
# Dispatch timing evidence WITH --warmup (two executions per ETDump; analysis uses the last): one ETDump run per model x scheme x build with the traced
# binaries in <stage>/{stock,sarc}-traced (x86) or the tracer-enabled campaign binaries (Orin).
set -uo pipefail
LOCK=""; MODELS=1b,3b,8b; SCHEMES=4w,8da4w; DEV=0; MROOT=/mnt/linux-share/models; FLAT=""
while [[ $# -gt 0 ]]; do case $1 in
  --lock) LOCK=$2; shift ;; --models) MODELS=$2; shift ;; --schemes) SCHEMES=$2; shift ;;
  --device-index) DEV=$2; shift ;; --model-root) MROOT=$2; shift ;; --flat-models) FLAT=$2; shift ;;
  --gpu|--reps|--cool-max) shift ;; esac; shift; done
D=$(cd "$(dirname "$0")" && pwd); cd "$D"; O=$D/raw/trace2; mkdir -p "$O"
exec 9>>"$HOME/.cache/gpu-lab/lock-$LOCK"; flock -w 900 9 || exit 75
export ETVK_DEVICE_INDEX=$DEV
declare -A STEM=([1b]=llama-3.2-1b:llama3_2-1b [3b]=llama-3.2-3b:llama3_2-3b [8b]=llama-3.1-8b:llama3_1-8b)
IFS=, read -ra MS <<< "$MODELS"; IFS=, read -ra QS <<< "$SCHEMES"
for b in stock sarc; do
  BD=$D/$b-traced; [[ -x $BD/llama_main ]] || BD=$D/$b
  for m in "${MS[@]}"; do IFS=: read -r MD ST <<< "${STEM[$m]}"; for q in "${QS[@]}"; do
    if [[ -n $FLAT ]]; then P=$FLAT/${ST}_vulkan_$q.pte; T=$FLAT/tokenizer.model
    else P=$MROOT/$MD/exported/${ST}_vulkan_$q.pte; T=$MROOT/$MD/original/tokenizer.model; fi
    LD_LIBRARY_PATH=$BD timeout 1800 $BD/llama_main --model_path $P --tokenizer_path $T \
      --prompt_file prompt_2048.txt --max_new_tokens 1 --temperature 0 --warmup \
      --etdump_path $O/$m-$q-$b.etdp < /dev/null > $O/$m-$q-$b.log 2>&1 9>&-
    echo "trace $m $q $b rc=$? $(ls -s $O/$m-$q-$b.etdp 2>/dev/null | cut -d' ' -f1)K"
  done; done
done
echo TRACE2_DONE
