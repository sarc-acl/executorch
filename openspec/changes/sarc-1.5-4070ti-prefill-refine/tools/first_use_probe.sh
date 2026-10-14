#!/bin/bash
# first_use_probe.sh <out name> <model> <scheme> <trials> <label>:<build tag>[=<env,env,...>] [...]: does the FIRST
# candidate process after the model changes abort after its output (rc 134), as `prefill 8b 4w cand r1` did in the
# gates s8-c4 and s8-c4b? Not a timing run and not a gate. Per trial and label, the order of the timed session at
# that position:
#   evict   the model file is dropped from the page cache (posix_fadvise DONTNEED through dd, no privilege);
#   parent  the pristine parent, as `parent r1`;
#   first   the label's build and environment, as `cand r1`;
#   warm    the same again, as `cand r2`.
# With STEPS=cold the trial is instead one run of the label's build right after the eviction (the label's own
# process reads the model from the share), which is what a slow first load looks like for that build.
# The standard prefill call (prompt_2048.txt, --warmup, one new token), a fresh process per run, each under the
# gpu-lab lock with the foreign-process watch (gl.sh), 300 s limit. One row per run appended to
# raw/<out name>/rows.csv (resumable); cached_pct is the share of the model file in the page cache before the run.
T=$(dirname "$0"); source "$T/common.sh"; O=$A/raw/$1; m=$2; q=$3; R=$4; shift 4
mkdir -p $O/logs; [[ -f $O/rows.csv ]] || echo "trial,label,step,build,model,scheme,cached_pct,rc,stats,echoed,abort_msg,tok_s,wall_s,utc" > $O/rows.csv
declare -A STEM=([1b]=llama-3.2-1b:llama3_2-1b [3b]=llama-3.2-3b:llama3_2-3b [8b]=llama-3.1-8b:llama3_1-8b)
IFS=: read -r MD ST <<< "${STEM[$m]}"; PTE=/mnt/linux-share/models/$MD/exported/${ST}_vulkan_$q.pte
cached() { fincore -n -b -o RES,SIZE "$PTE" 2>/dev/null | awk '{printf "%.0f", 100 * $1 / $2}'; }
one() {  # one <trial> <label> <step> <build tag> [env...]
  local t=$1 lab=$2 step=$3 BT=$4; shift 4
  grep -q "^$t,$lab,$step," $O/rows.csv && return 0
  local L=$A/build/$BT/llama/examples/models/llama/llama_main SO f=$O/logs/$lab-t$t-$step.log c t0 rc
  SO=$(dirname "$(find $A/build/$BT/llama -name libllama_runner.so | head -1)"); need $L
  cool_start 60 60; c=$(cached); t0=$SECONDS
  env "$@" LD_LIBRARY_PATH=$SO $T/gl.sh timeout 300 $L --model_path $PTE \
    --tokenizer_path /mnt/linux-share/models/$MD/original/tokenizer.model --prompt_file $KIT/prompts/prompt_2048.txt \
    --max_new_tokens 1 --temperature 0 --warmup < /dev/null > $f 2>&1
  rc=$?; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "stopped rc=$rc"; exit $rc; }
  echo "$t,$lab,$step,$BT,$m,$q,$c,$rc,$(grep -ac '"prefill_token_per_sec"' $f),$(grep -ac 'the the the' $f),$(grep -aoE 'corrupted [a-z -]+|free\(\)[a-z :-]+|malloc[a-z():_ -]+' $f | head -1),$(grep -ao '"prefill_token_per_sec":[0-9.]*' $f | head -1 | cut -d: -f2),$((SECONDS - t0)),$(date -u +%FT%TZ)" >> $O/rows.csv
  [[ $rc == 0 ]] && rm -f $f   # keep only the logs of the runs that failed
}
for ((t = 1; t <= R; t++)); do for le in "$@"; do
  lb=${le%%=*}; lab=${lb%%:*}; BT=${lb#*:}; E=(); [[ $le == *=* ]] && IFS=, read -ra E <<< "${le#*=}"
  if [[ ${STEPS:-} == cold ]]; then
    grep -q "^$t,$lab,cold," $O/rows.csv || dd if=$PTE iflag=nocache count=0 status=none
    one $t $lab cold $BT "${E[@]}"; continue
  fi
  grep -q "^$t,$lab,parent," $O/rows.csv || dd if=$PTE iflag=nocache count=0 status=none
  one $t $lab parent parent; one $t $lab first $BT "${E[@]}"; one $t $lab warm $BT "${E[@]}"
done; done
echo FIRST_USE_PROBE_DONE
