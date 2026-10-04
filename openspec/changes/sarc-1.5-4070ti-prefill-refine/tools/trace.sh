#!/bin/bash
# trace.sh <session> [models=1b,3b,8b] [schemes=4w,8da4w] [builds="parent cand"]: one warm ETDump per cell and arm
# with the traced binaries of stage/<session> (kit trace2.sh protocol), then kit/analysis/trace_analysis.py.
# Same rules as a timed run: under the gpu-lab lock, cooled to <= 50 C (max 120 s) before each run, no GPU process
# of another owner (exit 76), card alive before and after (exit 70). A run with rc != 0 or without 2048 prompt
# tokens fails the script (exit 4) after the remaining cells have run.
# Output: stage/<session>/trace/{raw/4070ti/trace2/*.etdp, report/evidence/trace/{families,gemm,totals}.csv}
set -uo pipefail
source "$(dirname "$0")/common.sh"; S=$A/stage/$1; MODELS=${2:-1b,3b,8b}; SCHEMES=${3:-4w,8da4w}; BUILDS=${4:-parent cand}
C=$S/trace; O=$C/raw/4070ti/trace2; need $S/prompt_2048.txt $KIT/analysis/trace_analysis.py
for b in $BUILDS; do need $S/$b-traced/llama_main $S/$b-traced/libllama_runner.so; done
mkdir -p $O $C/tools; cp -f $KIT/analysis/trace_analysis.py $C/tools/
take_lock; MROOT=/mnt/linux-share/models; BAD=0
declare -A STEM=([1b]=llama-3.2-1b:llama3_2-1b [3b]=llama-3.2-3b:llama3_2-3b [8b]=llama-3.1-8b:llama3_1-8b)
IFS=, read -ra MS <<< "$MODELS"; IFS=, read -ra QS <<< "$SCHEMES"
for b in $BUILDS; do BD=$S/$b-traced
  for m in "${MS[@]}"; do IFS=: read -r MD ST <<< "${STEM[$m]}"; for q in "${QS[@]}"; do
    cool_start 50 120; tp=$(gtemp) || gpu_gone "trace $m $q $b"; no_others "trace $m $q $b"
    benv=(); [[ -f $BD/env ]] && mapfile -t benv < $BD/env
    env "${benv[@]}" LD_LIBRARY_PATH=$BD timeout 1800 $BD/llama_main --model_path $MROOT/$MD/exported/${ST}_vulkan_$q.pte \
      --tokenizer_path $MROOT/$MD/original/tokenizer.model --prompt_file $S/prompt_2048.txt --max_new_tokens 1 \
      --temperature 0 --warmup --etdump_path $O/$m-$q-$b.etdp < /dev/null > $O/$m-$q-$b.log 2>&1 9>&-
    rc=$?; gone_check "trace $m $q $b rc=$rc"; oth=$(others)
    pt=$(grep -o '"prompt_tokens":[0-9]*' $O/$m-$q-$b.log | head -1 | cut -d: -f2)
    [[ $rc == 0 && $pt == 2048 && -s $O/$m-$q-$b.etdp ]] || BAD=1
    echo "trace $m $q $b rc=$rc prompt_tokens=$pt T=$tp->$(gtemp) $(grep -o '"prefill_token_per_sec":[0-9.]*' $O/$m-$q-$b.log)"
    [[ -n $oth ]] && no_others "during trace $m $q $b"
  done; done
done
# TRACE_PY: a python with the executorch devtools (ETDump inspector) installed.
( cd $C && ${TRACE_PY:-python3} tools/trace_analysis.py 2>&1 | grep -v Warning | tail -20; exit ${PIPESTATUS[0]} ) || BAD=1
echo "TRACE_DONE bad=$BAD"; exit $(( BAD ? 4 : 0 ))
