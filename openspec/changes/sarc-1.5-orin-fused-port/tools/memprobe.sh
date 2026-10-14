#!/bin/bash
# memprobe.sh <out name> <build tag> <rounds> <label>=<VAR=VALUE,...> [...]: device side. What the fused node's
# K / V copies cost in memory: MemAvailable (8 GB shared between CPU and GPU) is sampled every 0.2 s while one
# prefill run (prompt_2048.txt, --warmup, one new token) executes, labels interleaved, 4w and 8da4w of each model;
# the model file is in the page cache before each cell (D5). One row per run in raw/<out name>/rows.csv:
# label,model,scheme,round,avail_before_mb,avail_min_mb,drop_mb,swapfree_before_mb,swapfree_min_mb,pswpout_delta,tok_s,rc
# Not a timed session (the sampler runs beside the job): the rates are recorded only.
T=$(dirname "$0"); source "$T/common.sh"; O=$A/raw/$1; BT=$2; R=$3; shift 3
[[ $SIDE == device ]] || { echo "device only" >&2; exit 2; }
L=$A/build/$BT/bundle/llama_main; SO=$A/build/$BT/bundle; need $L; mkdir -p $O/logs
[[ -f $O/rows.csv ]] || echo "label,model,scheme,round,avail_before_mb,avail_min_mb,drop_mb,swapfree_before_mb,swapfree_min_mb,pswpout_delta,tok_s,rc" > $O/rows.csv
declare -A STEM=([1b]=llama3_2-1b [3b]=llama3_2-3b [8b]=llama3_1-8b)
val() { awk -v k="$1:" '$1 == k {print int($2 / 1024)}' /proc/meminfo; }
for m in 1b 3b 8b; do for q in 4w 8da4w; do P=$MODELDIR/${STEM[$m]}_vulkan_$q.pte; cat $P > /dev/null
  for ((r = 1; r <= R; r++)); do for le in "$@"; do lab=${le%%=*}; IFS=, read -ra E <<< "${le#*=}"
    grep -q "^$lab,$m,$q,$r," $O/rows.csv && continue
    cool_start 120; a0=$(val MemAvailable); s0=$(val SwapFree); p0=$(awk '/^pswpout/ {print $2}' /proc/vmstat)
    ( exec 9>&-; while :; do echo "$(val MemAvailable) $(val SwapFree)"; sleep 0.2; done > $O/logs/$lab-$m-$q-r$r.mem ) & SP=$!
    env "${E[@]}" LD_LIBRARY_PATH=$SO $T/gl.sh $L --model_path $P --tokenizer_path $MODELDIR/tokenizer.model \
      --prompt_file $KIT/prompts/prompt_2048.txt --max_new_tokens 1 --temperature 0 --warmup < /dev/null > $O/logs/$lab-$m-$q-r$r.log 2>&1
    rc=$?; kill $SP 2>/dev/null; wait $SP 2>/dev/null; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "stopped rc=$rc"; exit $rc; }
    read -r amin smin < <(awk 'NR == 1 {a = $1; s = $2} {if ($1 < a) a = $1; if ($2 < s) s = $2} END {print a, s}' $O/logs/$lab-$m-$q-r$r.mem)
    p1=$(awk '/^pswpout/ {print $2}' /proc/vmstat)
    echo "$lab,$m,$q,$r,$a0,$amin,$((a0 - amin)),$s0,$smin,$((p1 - p0)),$(grep -o '"prefill_token_per_sec":[0-9.]*' $O/logs/$lab-$m-$q-r$r.log | cut -d: -f2),$rc" | tee -a $O/rows.csv
  done; done
done; done
echo MEMPROBE_DONE
