#!/bin/bash
# session.sh: one interleaved cross-runtime prefill session on one device.
#
# Arms of ExecuTorch (llama_main) and of llama.cpp (llama-completion on the real prompt, llama-bench) are run
# in one rotation per model, under the device's gpu-lab lock, with the clock, temperature and foreign engine
# time sampled during every run. The device side (lock, build lock, foreign-workload guard, sensors, sampler)
# is the tuning campaign's: this script sources its tools/host.sh and uses its sampler.py, so the validity
# rules and the calibrated thresholds (clock floor, foreign-busy ceiling) are the campaign's own.
# Implemented against the host.sh of the Intel campaigns (functions gpu_shared, guarded, gpu_others,
# drm_clients; variables HW, LOCK, CLKMIN_FILE, BUSYMAX_FILE, *_TOP). A device whose campaign has no such
# file gets an adapter with the same names under kit/hosts/<device>/host.sh (optionally gtemp, dev_sampler,
# DEV_CLKMIN, DEV_BUSYMAX), written from that campaign's own session script.
#
# usage: session.sh --tools DIR --stage DIR --out NAME [--reps 5] [--extra 3] [--models 1b,3b,8b]
#                   [--arms FILE] [--tokens 2048] [--check-tokens 1972] [--no-check]
#   DIR (stage) holds prompt_2048.txt, prompt_check.txt, the ExecuTorch arm directories and arms.tsv:
#     <name> et <subdir> <scheme>                 subdir/{llama_main,libllama_runner.so,env}; scheme 4w|8da4w
#     <name> lc <binary> <gguf suffix> <flags...> llama-completion; flags = the tier's settings
#     <name> lb <binary> <gguf suffix> <flags...> llama-bench; one process, 5 repetitions inside it
#   Lines starting with # are ignored. Output in DIR/NAME/: runs.csv, checks.csv, logs/, env.txt, done.txt.
#
# Per model: every arm once per repetition; the order rotates by one arm per repetition and is reversed on
# even repetitions. ExecuTorch runs use --warmup (its warm-up pass is in the process). A llama-completion
# process has no full-prompt warm-up, so repetition 0 of every `lc` arm is run and discarded. `lb` arms run
# once. A run that is not valid is kept in runs.csv with its reason and replaced, at most EXTRA times.
# Then one `check` run per et and lc arm on prompt_check.txt: the text each arm generates next.
set -uo pipefail
TOOLS=""; STAGE=""; OUTN=raw; REPS=5; EXTRA=3; MODELS=1b,3b,8b; ARMS=""; TOKENS=2048; CHKTOK=1972; LCCHKTOK=1980; CHECK=1
MROOT=${MROOT:-/mnt/linux-share/models}; GROOT=${GROOT:-/mnt/linux-share/hmz-campaigns/compare/models}; COOLMAX=120
while [[ $# -gt 0 ]]; do
  case $1 in
    --tools) TOOLS=$2; shift ;; --stage) STAGE=$2; shift ;; --out) OUTN=$2; shift ;; --reps) REPS=$2; shift ;;
    --extra) EXTRA=$2; shift ;; --models) MODELS=$2; shift ;; --arms) ARMS=$2; shift ;; --tokens) TOKENS=$2; shift ;;
    --check-tokens) CHKTOK=$2; shift ;; --lc-check-tokens) LCCHKTOK=$2; shift ;; --no-check) CHECK=0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done
[[ -n $TOOLS && -n $STAGE ]] || { sed -n '2,27p' "$0"; exit 2; }
KIT=$(dirname "$(readlink -f "$0")"); D=$(cd "$STAGE" && pwd); cd "$D" || exit 2
ARMS=${ARMS:-$D/arms.tsv}; [[ -s $ARMS ]] || { echo "no arms file $ARMS" >&2; exit 2; }
O=$D/$OUTN; mkdir -p "$O/logs"
CT=$(readlink -f "$TOOLS"); . "$CT/host.sh"
TOP=$$; export B580_TOP=$TOP XE2_TOP=$TOP
exec 9>>"$HOME/.cache/gpu-lab/lock-$LOCK"; flock -w 900 9 || { echo "gpu-lab lock busy"; exit 75; }
gpu_shared || exit 75
# Thresholds: the campaign's calibration files where it has them (Intel), else the values its host file sets
# (DEV_CLKMIN; DEV_BUSYMAX may be empty where the device has no per-client engine accounting).
CLKMIN=${DEV_CLKMIN:-}; BUSYMAX=${DEV_BUSYMAX:-}
[[ -z $CLKMIN && -s ${CLKMIN_FILE:-/nonexistent} ]] && { CLKMIN=$(<"$CLKMIN_FILE"); BUSYMAX=$(<"$BUSYMAX_FILE"); }
[[ -n $CLKMIN && ${CLKMIN%.*} -gt 0 ]] || { echo "no clock floor from the campaign" >&2; exit 2; }
declare -F gtemp > /dev/null || gtemp() { echo $(( $(<$HW/temp2_input) / 1000 )); }
declare -F dev_sampler > /dev/null || dev_sampler() { exec python3 "$CT/sampler.py" "$1" 0.01 $TOP; }
finish() { local st=$1; shift; { date -u; echo "$st $*"; } > "$O/done.txt"; echo "$st $*"; [[ $st == SESSION_OK ]] && exit 0; [[ $st == SESSION_ABORTED ]] && exit 76; exit 1; }
declare -A STEM=([1b]=llama-3.2-1b:llama3_2-1b [3b]=llama-3.2-3b:llama3_2-3b [8b]=llama-3.1-8b:llama3_1-8b)
declare -A GSTEM=([1b]=llama3_2_1b [3b]=llama3_2_3b [8b]=llama3_1_8b)
pte()  { local MD ST; IFS=: read -r MD ST <<< "${STEM[$1]}"; echo "$MROOT/$MD/exported/${ST}_vulkan_$2.pte"; }
tokz() { local MD ST; IFS=: read -r MD ST <<< "${STEM[$1]}"; echo "$MROOT/$MD/original/tokenizer.model"; }
gguf() { echo "$GROOT/${GSTEM[$1]}_$2.gguf"; }
mapfile -t ARM < <(grep -v '^\s*#' "$ARMS" | grep -v '^\s*$')
{
  date -u; hostname; uname -r; echo "lock=$LOCK reps=$REPS extra=$EXTRA tokens=$TOKENS clkmin=$CLKMIN busymax=$BUSYMAX models=$MODELS"
  echo "arms ($ARMS):"; printf '  %s\n' "${ARM[@]}"
  for l in "${ARM[@]}"; do read -r n k x _ <<< "$l"
    case $k in et) sha256sum "$D/$x/llama_main" "$D/$x/libllama_runner.so"; echo "$n env: $(tr '\n' ' ' < "$D/$x/env" 2>/dev/null) commit: $(cat "$D/$x/COMMIT" 2>/dev/null)" ;;
               *) sha256sum "$x" ;; esac; done | sort -u
  sha256sum prompt_2048.txt prompt_check.txt
  vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName|driverInfo|apiVersion'
  echo "others: $(gpu_others)"; echo "drm clients not ours: $(drm_clients)"
} > "$O/env.txt" 2>&1
sleep 60; IDLE=$(gtemp); echo "idle_temp=$IDLE" >> "$O/env.txt"
cool() { local t0=$SECONDS t prev=999 flat=0
  while :; do t=$(gtemp); [[ $t -le $((IDLE + 5)) || $((SECONDS - t0)) -ge $COOLMAX ]] && break
    if [[ $t -ge $prev ]]; then flat=$((flat + 1)); [[ $flat -ge 2 ]] && break; else flat=0; fi; prev=$t; sleep 5; done; }
CSV=$O/runs.csv
[[ -f $CSV ]] || echo "model,arm,kind,tag,rep,slot,tok_s,prompt_tokens,prefill_ms,clk_n,clk_med_mhz,clk_min_mhz,throttled_n,power_avg_w,temp_max,valid,reason,fbusy_pct,samples,rc,temp_pre,temp_post,cool_s,others,utc,log" > "$CSV"
run1() {  # run1 <model> <arm line> <rep> <slot> <tag: prefill|discard|check>
  local m=$1 line=$2 r=$3 s=$4 tag=$5 name kind x rest log t0 cs tp tq rc oth sp us0 us1 want=$TOKENS prompt=prompt_2048.txt
  local tok pt ms n cm cmin thr pw tmax valid reason fb smp
  read -r name kind x rest <<< "$line"
  # the check prompt is real text: llama.cpp's tokenizer splits it into 1980 tokens, ExecuTorch's into 1972
  [[ $tag == check ]] && { want=$CHKTOK; prompt=prompt_check.txt; [[ $kind == lc ]] && want=$LCCHKTOK; }
  log="logs/$tag-$m-$name-r$r.log"
  t0=$SECONDS; cool; cs=$((SECONDS - t0)); tp=$(gtemp); oth=$(gpu_others | tr ',' ';')
  [[ -n $oth ]] && finish SESSION_ABORTED "other GPU workload before $tag $m $name r$r: $oth"
  dev_sampler "$O/${log%.log}.clk" 9>&- 8>&- & sp=$!
  us0=$(date +%s%6N)
  case $kind in
    et) local benv=(); [[ -f $D/$x/env ]] && mapfile -t benv < "$D/$x/env"
        guarded "$O/${log%.log}.others" env "${benv[@]}" LD_LIBRARY_PATH=$D/$x timeout 1800 "$D/$x/llama_main" \
          --model_path "$(pte $m $rest)" --tokenizer_path "$(tokz $m)" --prompt_file "$prompt" --max_new_tokens 1 \
          --temperature 0 $([[ $tag != check ]] && echo --warmup) < /dev/null > "$O/$log" 2>&1 9>&- ;;
    lc) local q=${rest%% *} fl=${rest#* }; [[ $fl == "$rest" ]] && fl=""
        # one new token after the prompt (check: 8); greedy; context of the ExecuTorch model; the automatic
        # beginning-of-sequence token off so that the prompt is the same 2048 tokens; -fit off so that no unset
        # argument depends on the memory free at launch. Generated text on stdout, log and timings on stderr.
        guarded "$O/${log%.log}.others" timeout 1800 bash -c 'date +%s%6N > "$0"; exec "$@"' "$O/.start" "$x" -m "$(gguf $m $q)" -f "$prompt" \
          -n $([[ $tag == check ]] && echo 8 || echo 1) --temp 0 --top-k 1 -c 2560 -fit off -no-cnv --no-display-prompt \
          --ignore-eos -s 1234 --perf --override-kv tokenizer.ggml.add_bos_token=bool:false $fl \
          < /dev/null > "$O/${log%.log}.out" 2> "$O/$log" 9>&- ;;
    lb) local q=${rest%% *} fl=${rest#* }; [[ $fl == "$rest" ]] && fl=""
        guarded "$O/${log%.log}.others" timeout 1800 "$x" -m "$(gguf $m $q)" -p $TOKENS -n 0 -r 5 -o json $fl \
          < /dev/null > "$O/$log" 2> "$O/${log%.log}.err" 9>&- ;;
    *) echo "unknown arm kind $kind" >&2; kill $sp; exit 2 ;;
  esac
  rc=$?; us1=$(date +%s%6N); kill $sp 2>/dev/null; wait $sp 2>/dev/null; tq=$(gtemp)
  # llama-completion's log times count from its own start, so the window needs the moment of the exec, not the
  # moment before the guard's checks
  [[ $kind == lc && -s $O/.start ]] && { us0=$(<"$O/.start"); rm -f "$O/.start"; }
  oth=$(cut -d' ' -f2- "$O/${log%.log}.others" 2>/dev/null | sort -u | tr -d '\n' | tr ',' ';')
  python3 "$KIT/row.py" $kind "$O/$log" "$O/${log%.log}.clk" "$want" "$CLKMIN" "$rc" "$oth" "$tag" "$BUSYMAX" $us0 $us1 > "$O/.row" 2> "$O/.row.err" \
    || echo ",,,,,,,,,0,row_parse_failed,," > "$O/.row"
  echo "$m,$name,$kind,$tag,$r,$s,$(<"$O/.row"),$rc,$tp,$tq,$cs,$oth,$(date -u +%FT%TZ),$log" >> "$CSV"
  IFS=, read -r tok pt ms n cm cmin thr pw tmax valid reason fb smp < "$O/.row"
  echo "$tag $m $name r$r slot$s tok_s=$tok tokens=$pt ms=$ms rc=$rc T=$tp->$tq cool=${cs}s clk=$cm/$cmin n=$n busy=${fb}% valid=$valid $reason"
  [[ -n $oth ]] && finish SESSION_ABORTED "other GPU workload during $tag $m $name r$r: $oth"
  return 0
}
nvalid() { awk -F, -v m=$1 -v a=$2 'NR > 1 && $1 == m && $2 == a && $4 == "prefill" && $16 == 1 {n++} END {print n + 0}' "$CSV"; }
FAILS=(); IFS=, read -ra MS <<< "$MODELS"; N=${#ARM[@]}
for m in "${MS[@]}"; do
  for l in "${ARM[@]}"; do read -r n k x rest <<< "$l"
    case $k in et) f=$(pte $m $rest) ;; *) f=$(gguf $m ${rest%% *}) ;; esac
    [[ -f $f ]] || finish SESSION_INCOMPLETE "missing model file $f"
    echo "model $m $n $(stat -c %s "$f") $f" >> "$O/env.txt"; done
  i=0; for l in "${ARM[@]}"; do read -r n k _ <<< "$l"; [[ $k == lc ]] && { run1 $m "$l" 0 $i discard; i=$((i + 1)); }; done
  for (( r = 1; r <= REPS + EXTRA; r++ )); do
    todo=0
    for (( j = 0; j < N; j++ )); do
      if (( r % 2 )); then idx=$(( (j + r - 1) % N )); else idx=$(( (N - 1 - j + r - 1) % N )); fi
      l=${ARM[$idx]}; read -r n k _ <<< "$l"
      need=$REPS; [[ $k == lb ]] && need=1
      (( $(nvalid $m $n) >= need )) && continue
      run1 $m "$l" $r $j prefill; todo=1
    done
    (( r >= REPS && todo == 0 )) && break
  done
  for l in "${ARM[@]}"; do read -r n k _ <<< "$l"; need=$REPS; [[ $k == lb ]] && need=1
    (( $(nvalid $m $n) >= need )) || { echo "CELL $m $n: fewer than $need valid runs"; FAILS+=("$m-$n:incomplete"); }; done
  if [[ $CHECK == 1 ]]; then
    [[ -f $O/checks.csv ]] || echo "model,arm,valid,text" > "$O/checks.csv"
    for l in "${ARM[@]}"; do read -r n k _ <<< "$l"; [[ $k == lb ]] && continue
      run1 $m "$l" 0 0 check
      v=$(tail -1 "$CSV" | cut -d, -f16)
      if [[ $k == et ]]; then t=$(grep -v 'PyTorchObserver\|^[IWE] \|^\[sarc_dev\]' "$O/logs/check-$m-$n-r0.log" | tr '\n,' '  ' | tail -c 60)
      else t=$(tr '\n,' '  ' < "$O/logs/check-$m-$n-r0.out" | head -c 60); fi
      echo "$m,$n,$v,\"$t\"" >> "$O/checks.csv"; [[ $v == 1 ]] || FAILS+=("$m-$n:check_invalid")
    done
  fi
done
[[ ${#FAILS[@]} == 0 ]] && finish SESSION_OK
finish SESSION_INCOMPLETE "${FAILS[*]}"
