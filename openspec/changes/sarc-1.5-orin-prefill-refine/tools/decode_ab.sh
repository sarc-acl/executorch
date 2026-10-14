#!/bin/bash
# decode_ab.sh <out name> <reps> <models: 1b,3b> <label>=<build tag>[:VAR=VALUE,...] [...]: device side. Decode is
# not what this campaign optimises, but a candidate must not slow it down unnoticed: the decode run of verify.sh
# (prompt_2048.txt, 32 new tokens, no warmup) repeated <reps> times per label, labels interleaved, both schemes.
# One row per run in raw/<out name>/rows.csv: label, model, scheme, rep, decode tok/s, generated tokens, prefill
# tok/s, rc. Resumable.
T=$(dirname "$0"); source "$T/common.sh"; [[ $SIDE == device ]] || { echo "device only" >&2; exit 2; }
O=$A/raw/$1; R=$2; IFS=, read -ra MS <<< "$3"; shift 3; mkdir -p $O/logs
[[ -f $O/rows.csv ]] || echo "label,model,scheme,rep,decode_tok_s,generated_tokens,prefill_tok_s,rc" > $O/rows.csv
declare -A STEM=([1b]=llama3_2-1b [3b]=llama3_2-3b [8b]=llama3_1-8b)
for ((r = 1; r <= R; r++)); do for m in "${MS[@]}"; do for q in 4w 8da4w; do for le in "$@"; do
  lab=${le%%=*}; spec=${le#*=}; bt=${spec%%:*}; E=(); [[ $spec == *:* ]] && IFS=, read -ra E <<< "${spec#*:}"
  grep -q "^$lab,$m,$q,$r," $O/rows.csv && continue
  BD=$A/build/$bt/bundle; need $BD/llama_main; L=$O/logs/$lab-$m-$q-r$r.log; cool_start 120
  env "${E[@]}" LD_LIBRARY_PATH=$BD $T/gl.sh $BD/llama_main --model_path $MODELDIR/${STEM[$m]}_vulkan_$q.pte --tokenizer_path $MODELDIR/tokenizer.model \
    --prompt_file $KIT/prompts/prompt_2048.txt --max_new_tokens 32 --temperature 0 < /dev/null > $L 2>&1
  rc=$?; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "stopped rc=$rc"; exit $rc; }
  g() { grep -o "\"$1\":[0-9.]*" $L | head -1 | cut -d: -f2; }
  echo "$lab,$m,$q,$r,$(g decode_token_per_sec),$(g generated_tokens),$(g prefill_token_per_sec),$rc" | tee -a $O/rows.csv
done; done; done; done
echo DECODE_AB_DONE
