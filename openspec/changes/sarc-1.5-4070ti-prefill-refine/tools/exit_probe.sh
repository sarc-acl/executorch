#!/bin/bash
# exit_probe.sh <out name> <model> <scheme> <reps> <label>:<build tag>[=<env,env,...>] [...]: how often does the
# runner fail AFTER it has produced its output (abort with rc 134, or a hang) on this card? Not a timing run.
# The standard prefill call (prompt_2048.txt, --warmup, one new token), labels interleaved, a fresh process per
# run, each under the gpu-lab lock with the foreign-process watch (gl.sh), 180 s limit per run (a hang reads
# rc 124). One row per run appended to raw/<out name>/rows.csv (resumable: a row that exists is skipped).
T=$(dirname "$0"); source "$T/common.sh"; O=$A/raw/$1; m=$2; q=$3; R=$4; shift 4
mkdir -p $O/logs; [[ -f $O/rows.csv ]] || echo "label,build,model,scheme,rep,rc,stats,echoed,prompt_tokens,tok_s,utc" > $O/rows.csv
declare -A STEM=([1b]=llama-3.2-1b:llama3_2-1b [3b]=llama-3.2-3b:llama3_2-3b [8b]=llama-3.1-8b:llama3_1-8b)
IFS=: read -r MD ST <<< "${STEM[$m]}"
for ((r = 1; r <= R; r++)); do for le in "$@"; do
  lb=${le%%=*}; lab=${lb%%:*}; BT=${lb#*:}; E=(); [[ $le == *=* ]] && IFS=, read -ra E <<< "${le#*=}"
  L=$A/build/$BT/llama/examples/models/llama/llama_main; SO=$(dirname "$(find $A/build/$BT/llama -name libllama_runner.so | head -1)"); need $L
  grep -q "^$lab,$BT,$m,$q,$r," $O/rows.csv && continue
  cool_start 60 60; f=$O/logs/$lab-$m-$q-r$r.log
  env "${E[@]}" LD_LIBRARY_PATH=$SO $T/gl.sh timeout 180 $L --model_path /mnt/linux-share/models/$MD/exported/${ST}_vulkan_$q.pte \
    --tokenizer_path /mnt/linux-share/models/$MD/original/tokenizer.model --prompt_file $KIT/prompts/prompt_2048.txt \
    --max_new_tokens 1 --temperature 0 --warmup < /dev/null > $f 2>&1
  rc=$?; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "stopped rc=$rc"; exit $rc; }
  echo "$lab,$BT,$m,$q,$r,$rc,$(grep -c '"prefill_token_per_sec"' $f),$(grep -c 'the the the' $f),$(grep -o '"prompt_tokens":[0-9]*' $f | head -1 | cut -d: -f2),$(grep -o '"prefill_token_per_sec":[0-9.]*' $f | head -1 | cut -d: -f2),$(date -u +%FT%TZ)" >> $O/rows.csv
  [[ $rc == 0 ]] && rm -f $f   # keep only the logs of the runs that failed
done; done
echo EXIT_PROBE_DONE
