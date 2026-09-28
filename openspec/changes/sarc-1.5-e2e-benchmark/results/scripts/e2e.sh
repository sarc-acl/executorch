#!/bin/bash
# End-to-end prefill campaign on one GPU host: build "stock" vs build "sarc" (see STAGE.md for what they are).
# Runs ON THE GPU HOST from the staged directory:
#   <stage>/{stock,sarc}/{llama_main,libllama_runner.so}, prompt_2048.txt, prompt_check.txt, e2e.sh
#
# usage: e2e.sh --gpu NAME --lock UUID [--reps 5] [--models 1b,3b,8b] [--schemes 4w,8da4w]
#               [--model-root P | --flat-models P] [--device-index N] [--cool-max 120]
#
# Protocol: for each model, scheme, repeat r = 1..reps the two builds run back to back,
# stock->sarc on odd r and sarc->stock on even r. Each run is a fresh llama_main process
# (--warmup, 2048-token prompt, 1 new token, temperature 0); prefill tok/s comes from the
# PyTorchObserver stats. Before each run the GPU cools to within 5 C of its idle baseline
# (max --cool-max s). Per cell, both builds also run prompt_check.txt once and their next
# token is compared. Every run lands in runs.csv; failed runs are kept and retried at the end.
set -uo pipefail

GPU=""; LOCK=""; REPS=5; MODELS=1b,3b,8b; SCHEMES=4w,8da4w; DEV=0; COOLMAX=120
MROOT=/mnt/linux-share/models; FLAT=""; PROMPT=prompt_2048.txt; OUTN=raw; CHECK=1
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu) GPU=$2; shift ;;
    --lock) LOCK=$2; shift ;;
    --reps) REPS=$2; shift ;;
    --models) MODELS=$2; shift ;;
    --schemes) SCHEMES=$2; shift ;;
    --device-index) DEV=$2; shift ;;
    --model-root) MROOT=$2; shift ;;
    --flat-models) FLAT=$2; shift ;;
    --cool-max) COOLMAX=$2; shift ;;
    --prompt) PROMPT=$2; shift ;;
    --out) OUTN=$2; shift ;;
    --no-check) CHECK=0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[[ -n $GPU && -n $LOCK ]] || { sed -n '2,14p' "$0"; exit 2; }
D=$(cd "$(dirname "$0")" && pwd); cd "$D" || exit 2
O=$D/$OUTN; mkdir -p "$O/logs"
exec 9>>"$HOME/.cache/gpu-lab/lock-$LOCK"; flock -w 900 9 || { echo "gpu-lab lock busy"; exit 75; }
export ETVK_DEVICE_INDEX=$DEV

# ---- device probes (best effort; recorded, never used to change state) ----
# nvidia-smi is usable only on discrete GPUs; on Jetson it reports [N/A].
nvok() { command -v nvidia-smi >/dev/null &&
  [[ $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1) =~ ^[0-9]+$ ]]; }
NVOK=0; nvok && NVOK=1
gtemp() {  # GPU temperature in C, or empty
  if [[ $NVOK == 1 ]]; then
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1; return; fi
  local z
  for z in /sys/class/thermal/thermal_zone*; do
    [[ $(cat $z/type 2>/dev/null) == gpu-thermal ]] && { echo $(( $(cat $z/temp) / 1000 )); return; }
  done
  local h best=""
  for h in /sys/class/hwmon/hwmon*; do
    case $(cat $h/name 2>/dev/null) in
      amdgpu|xe|i915)
        [[ $GPU == 780m && $(cat $h/name) != amdgpu ]] && continue
        [[ $GPU == b580 || $GPU == b70 ]] && [[ $(cat $h/name) == amdgpu ]] && continue
        for f in $h/temp*_input; do
          [[ -r $f ]] || continue; local t=$(( $(cat $f) / 1000 ))
          [[ -z $best || $t -gt $best ]] && best=$t
        done ;;
    esac
  done
  echo "$best"
}
clocks() {  # one-line clock/power snapshot
  if [[ $NVOK == 1 ]]; then
    nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,pstate --format=csv,noheader | head -1; return; fi
  local f out=""
  for f in /sys/class/devfreq/*gpu*/cur_freq; do
    [[ -r $f ]] && out+="devfreq=$(cat $f) "; done
  for f in /sys/class/drm/card*/device/pp_dpm_sclk; do
    [[ -r $f ]] && out+="sclk=$(grep '\*' $f | awk '{print $2}') "; done
  for f in /sys/class/drm/card*/device/tile0/gt0/freq0/act_freq; do
    [[ -r $f ]] && out+="xe_act=$(cat $f) "; done
  echo "${out:-n/a}"
}
others() {  # other GPU compute users (nvidia) or known GPU services
  if [[ $NVOK == 1 ]]; then
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v llama_main | tr '\n' ';'; fi
  pgrep -af 'llama-server|ComfyUI|comfyui|ollama|vllm|llama_main' | grep -v "$D" | grep -v pgrep | tr '\n' ';'
}

{
  date -u; hostname; uname -r; echo "gpu=$GPU lock=$LOCK dev=$DEV reps=$REPS prompt=$PROMPT"
  env | grep -E '^(ET_VK|ETVK)' | sort
  sha256sum stock/* sarc/* prompt_*.txt; for b in stock sarc; do echo "$b env: $(cat $b/env 2>/dev/null | tr '\n' ' ')"; done; cat STAGE.md 2>/dev/null
  command -v vulkaninfo >/dev/null && vulkaninfo --summary 2>/dev/null | grep -E 'GPU[0-9]|deviceName|driverName|driverInfo|apiVersion'
  command -v nvidia-smi >/dev/null && nvidia-smi 2>/dev/null | head -12
  [[ -r /etc/nv_tegra_release ]] && cat /etc/nv_tegra_release
  command -v nvpmodel >/dev/null && nvpmodel -q 2>/dev/null
  echo "clocks: $(clocks)"; echo "others: $(others)"
} > "$O/env.txt" 2>&1

declare -A STEM=([1b]=llama-3.2-1b:llama3_2-1b [3b]=llama-3.2-3b:llama3_2-3b [8b]=llama-3.1-8b:llama3_1-8b)
IFS=, read -ra MS <<< "$MODELS"; IFS=, read -ra QS <<< "$SCHEMES"
pte() { local MD ST; IFS=: read -r MD ST <<< "${STEM[$1]}"
  if [[ -n $FLAT ]]; then echo "$FLAT/${ST}_vulkan_$2.pte"; else echo "$MROOT/$MD/exported/${ST}_vulkan_$2.pte"; fi; }
tokz() { local MD ST; IFS=: read -r MD ST <<< "${STEM[$1]}"
  if [[ -n $FLAT ]]; then echo "$FLAT/tokenizer.model"; else echo "$MROOT/$MD/original/tokenizer.model"; fi; }
for m in "${MS[@]}"; do for q in "${QS[@]}"; do
  echo "model $m $q $(sha256sum "$(pte $m $q)" | cut -c1-16) $(pte $m $q)" >> "$O/env.txt"; done; done

sleep 60; IDLE=$(gtemp); echo "idle_temp=$IDLE" >> "$O/env.txt"
cool() { local t0=$SECONDS t
  [[ $IDLE =~ ^[0-9]+$ ]] || return
  while :; do t=$(gtemp); [[ ! $t =~ ^[0-9]+$ || $t -le $((IDLE + 5)) || $((SECONDS - t0)) -ge $COOLMAX ]] && break; sleep 5; done; }

CSV=$O/runs.csv
[[ -f $CSV ]] || echo "gpu,host,model,scheme,build,rep,slot,tok_s,rc,temp_pre,temp_post,cool_s,clocks,others,utc,log" > "$CSV"
run1() {  # run1 <model> <scheme> <build> <rep> <slot> <prompt> <tag>
  local m=$1 q=$2 b=$3 r=$4 s=$5 p=$6 tag=$7 log t0 tp tq rc tok cl oth
  log="logs/$tag-$m-$q-$b-r$r.log"
  t0=$SECONDS; cool; local cs=$((SECONDS - t0))
  tp=$(gtemp); oth=$(others | tr ',' ';')
  # per-build environment (e.g. the opt-in flags the previous kernels needed): <build>/env, KEY=VALUE per line
  local benv=(); [[ -f $D/$b/env ]] && mapfile -t benv < "$D/$b/env"
  env "${benv[@]}" LD_LIBRARY_PATH=$D/$b timeout 1800 "$D/$b/llama_main" --model_path "$(pte $m $q)" \
    --tokenizer_path "$(tokz $m)" --prompt_file "$p" --max_new_tokens 1 --temperature 0 \
    $([[ $tag == prefill ]] && echo --warmup) < /dev/null > "$O/$log" 2>&1 9>&-
  rc=$?; cl=$(clocks | tr ',' ';'); tq=$(gtemp)
  tok=$(grep -o '"prefill_token_per_sec":[0-9.]*' "$O/$log" | head -1 | cut -d: -f2)
  echo "$GPU,$(hostname),$m,$q,$b,$r,$s,${tok},$rc,$tp,$tq,$cs,$cl,$oth,$(date -u +%FT%TZ),$log" >> "$CSV"
  echo "$tag $m $q $b r$r slot$s tok_s=$tok rc=$rc T=$tp->$tq cool=${cs}s"
}
gen() { grep -v 'PyTorchObserver\|^[IWE] \|^\[sarc_dev\]' "$O/$1"; }

for m in "${MS[@]}"; do for q in "${QS[@]}"; do
  [[ -f $(pte $m $q) ]] || { echo "missing $(pte $m $q)"; continue; }
  for ((r = 1; r <= REPS; r++)); do
    if (( r % 2 )); then order=(stock sarc); else order=(sarc stock); fi
    run1 $m $q ${order[0]} $r 1 $PROMPT prefill
    run1 $m $q ${order[1]} $r 2 $PROMPT prefill
  done
  if [[ $CHECK == 1 ]]; then
  run1 $m $q stock 0 0 prompt_check.txt check
  run1 $m $q sarc 0 0 prompt_check.txt check
  if cmp -s <(gen "logs/check-$m-$q-stock-r0.log") <(gen "logs/check-$m-$q-sarc-r0.log"); then r=SAME; else r=DIFFER; fi
  echo "$m,$q,$r" >> "$O/nexttoken.csv"; echo "check $m $q next token sarc vs stock: $r"
  fi
done; done

# Retry failed prefill runs once (the failed rows stay in the CSV).
FAILED=$(awk -F, 'NR > 1 && $16 ~ /^logs\/prefill/ && ($9 != 0 || $8 == "") {print $3, $4, $5, $6}' "$CSV")
[[ -n $FAILED ]] && while read -r m q b r; do echo "retry $m $q $b r$r"; run1 $m $q $b "${r}x" 9 $PROMPT prefill; done <<< "$FAILED"
echo "clocks_end: $(clocks) others_end: $(others)" >> "$O/env.txt"
date -u > "$O/done.txt"; echo E2E_DONE
