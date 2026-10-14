#!/bin/bash
# trace.sh <session> [models=1b,3b,8b] [schemes=4w,8da4w] [builds="parent cand"]: one warm ETDump per cell and arm
# with the binaries of stage/<session> (kit trace2.sh protocol). Device side; the analysis needs the ExecuTorch
# devtools and runs on the workstation after pull.sh (trace_analyze.sh).
# Same rules as a timed run: under the gpu-lab lock, cooled before each run (until the temperature stops falling,
# max 120 s), no GPU process of another owner before or during a run (watched every 0.5 s; exit 76 on what was
# captured), GPU sensors alive before and after (exit 70). A run with rc != 0 or without 2048 prompt tokens fails
# the script (exit 4) after the remaining cells have run.
# Output: stage/<session>/trace/raw/orin/trace2/*.{etdp,log}
set -uo pipefail
source "$(dirname "$0")/common.sh"; S=$A/stage/$1; MODELS=${2:-1b,3b,8b}; SCHEMES=${3:-4w,8da4w}; BUILDS=${4:-parent cand}
[[ $SIDE == device ]] || { echo "device only" >&2; exit 2; }
C=$S/trace; O=$C/raw/orin/trace2; need $S/prompt_2048.txt
for b in $BUILDS; do need $S/$b-traced/llama_main $S/$b-traced/libllama_runner.so; done
mkdir -p $O; take_lock; BAD=0
declare -A STEM=([1b]=llama3_2-1b [3b]=llama3_2-3b [8b]=llama3_1-8b)
IFS=, read -ra MS <<< "$MODELS"; IFS=, read -ra QS <<< "$SCHEMES"
for b in $BUILDS; do BD=$S/$b-traced
  for m in "${MS[@]}"; do ST=${STEM[$m]}; for q in "${QS[@]}"; do
    cat $MODELDIR/${ST}_vulkan_$q.pte > /dev/null   # D5: page cache
    cool_start 120; tp=$(gtemp) || gpu_gone "trace $m $q $b"; no_others "trace $m $q $b"
    benv=(); [[ -f $BD/env ]] && mapfile -t benv < $BD/env
    others_watch_start $O/$m-$q-$b.others
    env "${benv[@]}" LD_LIBRARY_PATH=$BD timeout 1800 $BD/llama_main --model_path $MODELDIR/${ST}_vulkan_$q.pte \
      --tokenizer_path $MODELDIR/tokenizer.model --prompt_file $S/prompt_2048.txt --max_new_tokens 1 \
      --temperature 0 --warmup --etdump_path $O/$m-$q-$b.etdp < /dev/null > $O/$m-$q-$b.log 2>&1 9>&-
    rc=$?; oth=$(others_watch_stop $O/$m-$q-$b.others); gone_check "trace $m $q $b rc=$rc"
    pt=$(grep -o '"prompt_tokens":[0-9]*' $O/$m-$q-$b.log | head -1 | cut -d: -f2)
    [[ $rc == 0 && $pt == 2048 && -s $O/$m-$q-$b.etdp ]] || BAD=1
    echo "trace $m $q $b rc=$rc prompt_tokens=$pt T=$tp->$(gtemp) $(grep -o '"prefill_token_per_sec":[0-9.]*' $O/$m-$q-$b.log)"
    [[ -n $oth ]] && abort_others "during trace $m $q $b (its ETDump is not used)" "$oth"
  done; done
done
echo "TRACE_DONE bad=$BAD"; exit $(( BAD ? 4 : 0 ))
