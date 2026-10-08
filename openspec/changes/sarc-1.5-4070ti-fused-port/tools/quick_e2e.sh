#!/bin/bash
# quick_e2e.sh <out name> <build tag> <reps> <label>=<env,env,...> [...]: an UNGATED look at end-to-end prefill
# (prompt_2048.txt, --warmup, one new token) for a few environments on one build, labels interleaved, all six
# cells. One row per run appended to raw/<out name>/rows.csv (resumable: a row that exists is skipped). Cools to
# 55 C (max 120 s) before each run. This is a screen for choosing what to gate, not evidence of a gain.
T=$(dirname "$0"); source "$T/common.sh"; O=$A/raw/$1; BT=$2; R=$3; shift 3
L=$A/build/$BT/llama/examples/models/llama/llama_main; SO=$(dirname "$(find $A/build/$BT/llama -name libllama_runner.so | head -1)"); need $L
mkdir -p $O/logs; [[ -f $O/rows.csv ]] || echo "label,model,scheme,rep,tok_s,rc,prompt_tokens,temp_pre" > $O/rows.csv
declare -A STEM=([1b]=llama-3.2-1b:llama3_2-1b [3b]=llama-3.2-3b:llama3_2-3b [8b]=llama-3.1-8b:llama3_1-8b)
for ((r = 1; r <= R; r++)); do for m in 1b 3b 8b; do IFS=: read -r MD ST <<< "${STEM[$m]}"; for q in 4w 8da4w; do for le in "$@"; do
  lab=${le%%=*}; IFS=, read -ra E <<< "${le#*=}"; [[ $le == *=* ]] || E=()
  grep -q "^$lab,$m,$q,$r," $O/rows.csv && continue
  hold_point "screen run of $(basename $O)"
  warm_model /mnt/linux-share/models/$MD/exported/${ST}_vulkan_$q.pte $O/warm.csv $lab-$m-$q-r$r   # owner decision D5
  cool_start 55 120; tp=$(gtemp)
  env "${E[@]}" LD_LIBRARY_PATH=$SO $T/gl.sh $L --model_path /mnt/linux-share/models/$MD/exported/${ST}_vulkan_$q.pte \
    --tokenizer_path /mnt/linux-share/models/$MD/original/tokenizer.model --prompt_file $KIT/prompts/prompt_2048.txt \
    --max_new_tokens 1 --temperature 0 --warmup < /dev/null > $O/logs/$lab-$m-$q-r$r.log 2>&1
  rc=$?; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "stopped rc=$rc"; exit $rc; }
  echo "$lab,$m,$q,$r,$(grep -o '"prefill_token_per_sec":[0-9.]*' $O/logs/$lab-$m-$q-r$r.log | cut -d: -f2),$rc,$(grep -o '"prompt_tokens":[0-9]*' $O/logs/$lab-$m-$q-r$r.log | cut -d: -f2),$tp" >> $O/rows.csv
done; done; done; done
echo QUICK_DONE
