#!/bin/bash
# trace.sh <session> [models=1b,3b,8b] [schemes=4w,8da4w] [builds="parent cand"]: one warm ETDump per cell and arm
# with the traced binaries of stage/<session> (kit trace2.sh protocol: --warmup, two executions, the last is
# used), page cache filled per model (D5), then etdump_families.py.
# Output: stage/<session>/trace/{<m>-<q>-<b>.{etdp,log}, families.csv, kernels.csv, totals.csv}
set -uo pipefail
[[ -n ${SARC_HOLD_UNIT:-} ]] || exec env SARC_HOLD_UNIT=1 "$(dirname "$(readlink -f "$0")")/hold.sh" run "trace set trace.sh $*" "$0" "$@"
source "$(dirname "$(readlink -f "$0")")/env.sh"
S=$A/stage/$1; MODELS=${2:-1b,3b,8b}; SCHEMES=${3:-4w,8da4w}; BUILDS=${4:-parent cand}
O=$S/trace; mkdir -p $O
while [[ -e $A/.building ]]; do sleep 20; done
exec 9>>"$LOCKF"; flock -w 3600 9 || exit 75
declare -A STEM=([1b]=llama3_2-1b [3b]=llama3_2-3b [8b]=llama3_1-8b)
IFS=, read -ra MS <<< "$MODELS"; IFS=, read -ra QS <<< "$SCHEMES"
for m in "${MS[@]}"; do for q in "${QS[@]}"; do P=$MFLAT/${STEM[$m]}_vulkan_$q.pte; cat $P > /dev/null
  for b in $BUILDS; do BD=$S/$b-traced
    benv=(); [[ -f $BD/env ]] && mapfile -t benv < $BD/env
    t0=$SECONDS; while (( $(gtemp) > 60 && SECONDS - t0 < 300 )); do sleep 5; done
    env "${benv[@]}" LD_LIBRARY_PATH=$BD timeout 1800 $BD/llama_main --model_path $P --tokenizer_path $MFLAT/tokenizer.model \
      --prompt_file $S/prompt_2048.txt --max_new_tokens 1 --temperature 0 --warmup --etdump_path $O/$m-$q-$b.etdp \
      < /dev/null > $O/$m-$q-$b.log 2>&1 9>&-
    echo "trace $m $q $b rc=$? $(grep -o '"prefill_token_per_sec":[0-9.]*' $O/$m-$q-$b.log) others=$($T/others.sh)"
  done
done; done
# the ETDump analysis (flatc, schema of the exported tree) runs on the control workstation after pull-stage.sh: tools/trace-analyze.sh
[[ $WHERE == ws ]] && { $PY $T/etdump_families.py $O > $O/analysis.out 2>&1; tail -3 $O/analysis.out; }
echo TRACE_DONE
