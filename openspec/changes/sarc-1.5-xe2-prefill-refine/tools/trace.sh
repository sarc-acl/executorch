#!/bin/bash
# trace.sh <session> [models=1b,3b] [schemes=4w,8da4w] [builds="parent cand"]: one warm ETDump per cell and arm
# with the traced binaries of stage/<session> (kit trace2.sh protocol), then kit/analysis/trace_analysis.py.
# Output: stage/<session>/trace/{raw/xe2/trace2/*.etdp, report/evidence/trace/{families,gemm,totals}.csv}
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"; S=$A/stage/$1; MODELS=${2:-1b,3b}; SCHEMES=${3:-4w,8da4w}; BUILDS=${4:-parent cand}
TR=$S/trace; O=$TR/raw/xe2/trace2; mkdir -p $O $TR/tools
cp -f $ET/openspec/changes/sarc-1.5-e2e-benchmark/kit/analysis/trace_analysis.py $TR/tools/
exec 9>>"$HOME/.cache/gpu-lab/lock-$LOCK"; flock -w 1800 9 || exit 75
export ETVK_DEVICE_INDEX=0; MROOT=/mnt/linux-share/models
declare -A STEM=([1b]=llama-3.2-1b:llama3_2-1b [3b]=llama-3.2-3b:llama3_2-3b [8b]=llama-3.1-8b:llama3_1-8b)
IFS=, read -ra MS <<< "$MODELS"; IFS=, read -ra QS <<< "$SCHEMES"
for b in $BUILDS; do BD=$S/$b-traced
  for m in "${MS[@]}"; do IFS=: read -r MD ST <<< "${STEM[$m]}"; for q in "${QS[@]}"; do
    benv=(); [[ -f $BD/env ]] && mapfile -t benv < $BD/env
    env "${benv[@]}" LD_LIBRARY_PATH=$BD timeout 1800 $BD/llama_main --model_path $MROOT/$MD/exported/${ST}_vulkan_$q.pte \
      --tokenizer_path $MROOT/$MD/original/tokenizer.model --prompt_file $S/prompt_2048.txt --max_new_tokens 1 \
      --temperature 0 --warmup --etdump_path $O/$m-$q-$b.etdp < /dev/null > $O/$m-$q-$b.log 2>&1 9>&-
    echo "trace $m $q $b rc=$? $(grep -o '"prefill_token_per_sec":[0-9.]*' $O/$m-$q-$b.log)"
  done; done
done
${XE2_PYTHON:-python3} $TR/tools/trace_analysis.py 2>&1 | grep -v Warning | tail -20
echo TRACE_DONE
