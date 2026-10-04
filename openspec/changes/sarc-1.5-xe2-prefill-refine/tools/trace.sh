#!/bin/bash
# trace.sh <session> [models=1b,3b] [schemes=4w,8da4w] [builds="parent cand"]: one warm ETDump per cell and arm
# with the traced binaries of stage/<session> (kit trace2.sh protocol), then tools/trace_analysis.py (the
# campaign-local copy of the kit analyzer, which reads raw/xe2).
# Output: stage/<session>/trace/{raw/xe2/trace2/*.etdp, report/evidence/trace/{families,gemm,totals}.csv} and
# trace.ok. Exit status 0 and trace.ok only if every run returned 0 with 2048 prompt tokens and a non-empty
# ETDump, no foreign GPU process was seen, and the analyzer produced one totals row per run.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"; S=$A/stage/$1; MODELS=${2:-1b,3b}; SCHEMES=${3:-4w,8da4w}; BUILDS=${4:-parent cand}
TR=$S/trace; O=$TR/raw/xe2/trace2; mkdir -p $O $TR/tools || exit 2
rm -f $TR/trace.ok; cp -f $TOOLS/trace_analysis.py $TR/tools/
exec 9>>"$HOME/.cache/gpu-lab/lock-$LOCK"; flock -w 1800 9 || exit 75
MROOT=/mnt/linux-share/models; bad=0; n=0
declare -A STEM=([1b]=llama-3.2-1b:llama3_2-1b [3b]=llama-3.2-3b:llama3_2-3b [8b]=llama-3.1-8b:llama3_1-8b)
IFS=, read -ra MS <<< "$MODELS"; IFS=, read -ra QS <<< "$SCHEMES"
for b in $BUILDS; do BD=$S/$b-traced
  for m in "${MS[@]}"; do IFS=: read -r MD ST <<< "${STEM[$m]}"; for q in "${QS[@]}"; do
    benv=(); [[ -f $BD/env ]] && mapfile -t benv < $BD/env
    n=$((n + 1)); rm -f $O/$m-$q-$b.etdp
    guarded $O/$m-$q-$b.others env "${benv[@]}" LD_LIBRARY_PATH=$BD timeout 1800 $BD/llama_main \
      --model_path $MROOT/$MD/exported/${ST}_vulkan_$q.pte --tokenizer_path $MROOT/$MD/original/tokenizer.model \
      --prompt_file $S/prompt_2048.txt --max_new_tokens 1 --temperature 0 --warmup \
      --etdump_path $O/$m-$q-$b.etdp < /dev/null > $O/$m-$q-$b.log 2>&1 9>&-
    rc=$?; st=ok
    [[ $rc == 0 && -s $O/$m-$q-$b.etdp ]] && grep -q '"prompt_tokens":2048' $O/$m-$q-$b.log || { st=FAILED; bad=$((bad + 1)); }
    echo "trace $m $q $b rc=$rc $st $(grep -o '"prefill_token_per_sec":[0-9.]*' $O/$m-$q-$b.log)"
    [[ $rc == 76 ]] && { echo "TRACE_ABORTED other GPU process"; exit 76; }
  done; done
done
[[ $bad == 0 ]] || { echo "TRACE_FAILED $bad of $n runs"; exit 1; }
${XE2_PYTHON:-python3} $TR/tools/trace_analysis.py trace2 $n 2>&1 | grep -v Warning | tail -20
[[ ${PIPESTATUS[0]} == 0 && $(($(wc -l < $TR/report/evidence/trace/totals.csv) - 1)) == "$n" ]] || { echo TRACE_FAILED analysis; exit 1; }
date -u > $TR/trace.ok; echo TRACE_DONE
