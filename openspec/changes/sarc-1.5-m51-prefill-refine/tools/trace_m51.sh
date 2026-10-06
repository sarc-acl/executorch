#!/bin/bash
# trace_m51.sh <session> [models=1b,3b] [schemes=4w,8da4w] [arms="parent cand"]: one warm ETDump per cell and arm
# with the ETDump binaries of stage/m51-LOCAL-ONLY/<session>/<arm>-etdump (kit trace2.sh protocol: --warmup, the
# timed prompt, one new token), pulled to <stage>/trace/<model>-<scheme>-<arm>.etdp, then trace_families.py.
# One coordinator-hold unit; device-state guard before every run; a board that disappears writes ABORTED.
set -uo pipefail
[[ -n ${SARC_HOLD_UNIT:-} ]] || exec env SARC_HOLD_UNIT=1 "$(dirname "$(readlink -f "$0")")/hold.sh" run "trace set trace_m51.sh $*" "$0" "$@"
source "$(dirname "$(readlink -f "$0")")/dev.sh"
SES=$1; MODELS=${2:-1b,3b}; SCHEMES=${3:-4w,8da4w}; ARMS=${4:-parent cand}
S=$ART/stage/$LOC/$SES; DS=$DEV_ROOT/stage/$SES; O=$S/trace; mkdir -p "$O"
declare -A STEM=([1b]=llama3_2_1b [3b]=llama3_2_3b [8b]=llama3_1_8b)
IFS=, read -ra MS <<< "$MODELS"; IFS=, read -ra QS <<< "$SCHEMES"
for b in $ARMS; do for m in "${MS[@]}"; do for q in "${QS[@]}"; do
  st=$(device_state); [[ $st == ok ]] || { echo "board not fit before $m $q $b: $st"; [[ $st == device_gone ]] && echo "board gone before trace $m $q $b" >> "$ART/ABORTED"; exit 3; }
  t=$m-$q-$b; benv=$(tr '\n' ' ' < "$S/$b-etdump/env")
  A shell "cd $DS/$b-etdump && rm -f $t.etdp && $benv LD_LIBRARY_PATH=$DS/$b-etdump timeout 1190 ./llama_main --model_path=$DEV_ROOT/models/${STEM[$m]}_${q}_embq_ctx3072.pte --tokenizer_path=$DEV_ROOT/models/tokenizer.model --prompt_file=prompt_2048.txt --max_new_tokens=1 --temperature=0 --warmup --etdump_path=$t.etdp < /dev/null > $t.log 2>&1; echo RC=\$? >> $t.log" < /dev/null > /dev/null 2>&1
  alive || { echo "board gone during trace $t $(date -u +%FT%TZ)" | tee -a "$ART/ABORTED"; exit 3; }
  A pull "$DS/$b-etdump/$t.log" "$O/" > /dev/null; A pull "$DS/$b-etdump/$t.etdp" "$O/" > /dev/null 2>&1
  echo "trace $t $(tail -1 "$O/$t.log") $(grep -o '"prefill_token_per_sec":[0-9.]*' "$O/$t.log")"
done; done; done
"$ART/venv/m51/bin/python" "$TOOLS/trace_families.py" "$O" > "$O/families.csv" 2> "$O/families.err"; echo "families: $O/families.csv"
echo TRACE_DONE
