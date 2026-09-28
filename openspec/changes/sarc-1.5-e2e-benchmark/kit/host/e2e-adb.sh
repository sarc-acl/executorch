#!/usr/bin/env bash
# e2e-adb.sh: the M1 / M2 / M4 protocols of e2e.sh, trace2.sh and probe.sh, driven from a Linux host over
# adb for an Android phone. Every run is one `adb shell` invocation (a fresh process on the phone).
#
# usage: e2e-adb.sh --serial SERIAL --gpu NAME --out LOCAL_DIR [--mode e2e|trace|probe]
#          [--device-dir /data/local/tmp/e2eb] [--model-dir <device-dir>/models] [--stage DIR]
#          [--prompt prompt_2048.txt] [--reps 5] [--models 1b,3b,8b] [--schemes 4w,8da4w] [--no-check]
#          [--pacing skin|idle] [--temp-zone SYSFS_TEMP] [--cool-max 120]
#          [--guard-md5 DEVICE_FILE=MD5] [--guard-value DEVICE_FILE=VALUE] [--cmd-log FILE] [--host-label NAME]
#
# Device dir (--device-dir) layout:
#   {stock,sarc}/llama_main            M1 binaries (plus any .so they need: LD_LIBRARY_PATH is that dir)
#   {stock,sarc}-traced/llama_main     M2 (ETDump) binaries        probe-{stock,sarc}/logits_probe  M4
#   tokenizer.model, the kit prompts (prompt_2048.txt, prompt_check.txt, prompt_real_2048.txt),
#   logits_probe/{real,check}_ids.txt, and in --model-dir: <stem>_vulkan_<scheme>.pte with stem
#   llama3_2-1b, llama3_2-3b, llama3_1-8b (symlinks are fine).
# Per-build env: <stage>/{stock,sarc}/env (KEY=VALUE lines, as e2e.sh); stage defaults to dirname of --out.
#
# Modes:
#   e2e    M1 (as e2e.sh): per model/scheme, rep r runs stock->sarc (odd r) / sarc->stock (even r), --warmup,
#          1 new token, temperature 0 -> <out>/runs.csv (the kit schema) + logs/; the check runs on
#          prompt_check.txt (rep 0, slot 0) -> nexttoken.csv; failed prefill runs are retried once as rep
#          "<r>x", slot 9.
#   trace  M2 (as trace2.sh): one --warmup prefill per cell with {stock,sarc}-traced/llama_main and
#          --etdump_path -> <out>/trace2/<m>-<q>-<b>.{etdp,log}
#   probe  M4 (as probe.sh): probe-<b>/logits_probe on logits_probe/{real,check}_ids.txt
#          -> <out>/probe/<m>-<q>-<b>-<real|check>.json (+ .log)
#
# Pacing before every run (at most 300 s for skin, --cool-max s for idle):
#   skin  (phones whose thermal HAL has no GPU sensor, e.g. Adreno) dumpsys thermalservice SKIN < 38.5 C and
#         the hottest gpuss-* thermal zone <= 45 C;
#   idle  the --temp-zone temperature back within 5 C of its idle value (read after a 60 s settle).
# The clocks column of runs.csv holds the guarded state after each run, plus the kgsl GPU clock, thermal cap
# and reset counter where /sys/class/kgsl is readable (Adreno, no root needed).
# Guards (checked before every run; the campaign stops on a change, so device states never mix):
#   --guard-md5 /vendor/lib64/hw/vulkan.X.so=<md5>   driver binary; --guard-value /sys/.../min_freq=<v>   pin.
# adb server: honours ADB_SERVER_SOCKET (e.g. a tunnelled remote server). Never uses the default device.
set -uo pipefail
S=""; GPU=""; OUTD=""; MODE=e2e; D=/data/local/tmp/e2eb; MD=""; STAGE=""; PROMPT=prompt_2048.txt; REPS=5
MODELS=1b,3b,8b; SCHEMES=4w,8da4w; CHECK=1; PACING=skin; TZ=""; COOLMAX=120; GMD5=(); GVAL=(); CMDLOG=/dev/null
HOSTN=$(hostname)
while [[ $# -gt 0 ]]; do
  case $1 in
    --serial) S=$2; shift ;; --gpu) GPU=$2; shift ;; --out) OUTD=$2; shift ;; --mode) MODE=$2; shift ;;
    --device-dir) D=$2; shift ;; --model-dir) MD=$2; shift ;; --stage) STAGE=$2; shift ;;
    --prompt) PROMPT=$2; shift ;; --reps) REPS=$2; shift ;; --models) MODELS=$2; shift ;;
    --schemes) SCHEMES=$2; shift ;; --no-check) CHECK=0 ;; --pacing) PACING=$2; shift ;;
    --temp-zone) TZ=$2; shift ;; --cool-max) COOLMAX=$2; shift ;; --guard-md5) GMD5+=("$2"); shift ;;
    --guard-value) GVAL+=("$2"); shift ;; --cmd-log) CMDLOG=$2; shift ;; --host-label) HOSTN=$2; shift ;;
    -h|--help) sed -n '2,37p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done
[[ -n $S && -n $GPU && -n $OUTD ]] || { sed -n '2,37p' "$0"; exit 2; }
[[ $MODE =~ ^(e2e|trace|probe)$ && $PACING =~ ^(skin|idle)$ ]] || { echo "bad --mode/--pacing" >&2; exit 2; }
[[ $PACING == idle && -z $TZ ]] && { echo "--pacing idle needs --temp-zone" >&2; exit 2; }
MD=${MD:-$D/models}; STAGE=${STAGE:-$(dirname "$OUTD")}
A() { adb -s "$S" "$@"; }
alive() { A get-state 2>/dev/null | grep -q device; }
alive || { echo "device $S not found (adb devices -l)" >&2; exit 3; }
mkdir -p "$OUTD/logs"

gtemp() {  # temperature in C (integer) of --temp-zone, else the hottest gpuss-* zone; empty if none
  if [[ -n $TZ ]]; then A shell "cat $TZ" | tr -d '\r' | awk '{printf "%d", $1/1000}'
  else A shell 'for z in /sys/class/thermal/thermal_zone*; do case $(cat $z/type) in gpuss-*) cat $z/temp;; esac; done' \
         | tr -d '\r' | sort -n | tail -1 | awk '{printf "%d", $1/1000}'; fi; }
skin() { A shell dumpsys thermalservice | grep -m1 -oE 'mValue=[0-9.]+, mType=3' | grep -oE '[0-9.]+' | head -1; }
state() {  # the guarded device state, one line
  local g f v; for g in "${GMD5[@]}"; do f=${g%%=*}; echo -n "md5($f)=$(A shell md5sum "$f" | cut -d' ' -f1) "; done
  for g in "${GVAL[@]}"; do f=${g%%=*}; v=$(A shell cat "$f" | tr -d '\r'); echo -n "$f=$v "; done
  # Adreno (kgsl): current and thermally capped GPU clock, readable without root.
  A shell 'k=/sys/class/kgsl/kgsl-3d0; [ -r $k/clock_mhz ] && echo -n "kgsl clock_mhz=$(cat $k/clock_mhz) max_clock_mhz=$(cat $k/max_clock_mhz) thermal_pwrlevel=$(cat $k/thermal_pwrlevel) reset_count=$(cat $k/reset_count)"' | tr -d '\r'; echo; }
others() { A shell "pgrep -a -f 'llama_main|test_llama|igpu|roofline' | grep -v pgrep" 2>/dev/null | tr -d '\r' | tr '\n' ';' | tr ',' ';'; }
guard() {  # stop on a lost device or a changed guarded state
  alive || { echo "DEVICE GONE -- stopping"; exit 3; }
  local g f v
  for g in "${GMD5[@]}"; do f=${g%%=*}; v=$(A shell md5sum "$f" | cut -d' ' -f1)
    [[ $v == "${g#*=}" ]] || { echo "GUARD md5($f)=$v, want ${g#*=} -- stopping"; exit 4; }; done
  for g in "${GVAL[@]}"; do f=${g%%=*}; v=$(A shell cat "$f" | tr -d '\r')
    [[ $v == "${g#*=}" ]] || { echo "GUARD $f=$v, want ${g#*=} -- stopping"; exit 5; }; done
}
declare -A STEM=([1b]=llama3_2-1b [3b]=llama3_2-3b [8b]=llama3_1-8b)
IFS=, read -ra MS <<< "$MODELS"; IFS=, read -ra QS <<< "$SCHEMES"
benv() { [[ -f $STAGE/$1/env ]] && tr '\n' ' ' < "$STAGE/$1/env"; }
{
  echo "=== invocation $(date -u +%FT%TZ): mode=$MODE models=$MODELS schemes=$SCHEMES"
  date -u; echo "host=$HOSTN serial=$S gpu=$GPU reps=$REPS prompt=$PROMPT pacing=$PACING device_dir=$D"
  A shell "getprop ro.product.model; getprop ro.soc.model; getprop ro.build.fingerprint; uname -r"
  echo "state: $(state)"
  A shell "cd $D && sha256sum stock/llama_main sarc/llama_main $PROMPT prompt_check.txt tokenizer.model"
  for b in stock sarc; do echo "$b env: $(benv $b)"; done
  for m in "${MS[@]}"; do for q in "${QS[@]}"; do
    echo "model $m $q $(A shell "sha256sum $MD/${STEM[$m]}_vulkan_$q.pte" | cut -c1-16) ${STEM[$m]}_vulkan_$q.pte"; done; done
  echo "others: $(others)"
} >> "$OUTD/env.txt" 2>&1

sleep 60; IDLE=$(gtemp); echo "idle_temp=$IDLE" >> "$OUTD/env.txt"
cool() { local t0=$SECONDS t g
  if [[ $PACING == skin ]]; then
    # SKIN alone let runs start with a 50 C+ GPU on the S26, hence the gpuss-* bound too.
    while :; do t=$(skin); g=$(gtemp)
      awk -v t="$t" -v g="$g" 'BEGIN{exit !(t<38.5 && (g == "" || g<=45))}' && break
      (( SECONDS - t0 >= 300 )) && break; sleep 10; done
  else
    [[ $IDLE =~ ^[0-9]+$ ]] || return
    while :; do t=$(gtemp); [[ ! $t =~ ^[0-9]+$ || $t -le $((IDLE + 5)) || $((SECONDS - t0)) -ge $COOLMAX ]] && break; sleep 5; done
  fi; }

if [[ $MODE != e2e ]]; then
  mkdir -p "$OUTD/trace2" "$OUTD/probe"
  for m in "${MS[@]}"; do for q in "${QS[@]}"; do for b in stock sarc; do
    if [[ $MODE == trace ]]; then
      cool; guard; t=$m-$q-$b
      echo "# e2e-adb $GPU $S trace $t: $(benv $b) $b-traced/llama_main" >> "$CMDLOG"
      A shell "cd $D && mkdir -p trace2 && $(benv $b) LD_LIBRARY_PATH=$D/$b-traced timeout 1800 $D/$b-traced/llama_main --model_path=$MD/${STEM[$m]}_vulkan_$q.pte \
        --tokenizer_path=tokenizer.model --prompt_file=prompt_2048.txt --max_new_tokens=1 --temperature=0 --warmup \
        --etdump_path=trace2/$t.etdp < /dev/null > trace2/$t.log 2>&1; echo RC=\$? >> trace2/$t.log" < /dev/null
      alive || { echo "DEVICE GONE during trace $t -- stopping"; exit 3; }
      A pull $D/trace2/$t.log "$OUTD/trace2/" > /dev/null; A pull $D/trace2/$t.etdp "$OUTD/trace2/" > /dev/null 2>&1
      echo "trace $t $(tail -n1 "$OUTD/trace2/$t.log") etdp=$([[ -s $OUTD/trace2/$t.etdp ]] && echo yes || echo NO)"
    else
      for pr in real check; do
        cool; guard; t=$m-$q-$b-$pr
        echo "# e2e-adb $GPU $S probe $t: $(benv $b) probe-$b/logits_probe" >> "$CMDLOG"
        A shell "cd $D && mkdir -p probe && $(benv $b) LD_LIBRARY_PATH=$D/$b timeout 1800 $D/probe-$b/logits_probe $MD/${STEM[$m]}_vulkan_$q.pte \
          logits_probe/${pr}_ids.txt probe/$t.json 6062 45647 < /dev/null > probe/$t.log 2>&1; echo RC=\$? >> probe/$t.log" < /dev/null
        alive || { echo "DEVICE GONE during probe $t -- stopping"; exit 3; }
        A pull $D/probe/$t.log "$OUTD/probe/" > /dev/null; A pull $D/probe/$t.json "$OUTD/probe/" > /dev/null 2>&1
        echo "probe $t $(grep -m1 '^top1' "$OUTD/probe/$t.log") $(tail -n1 "$OUTD/probe/$t.log")"
      done
    fi
  done; done; done
  echo "state_end: $(state) others_end: $(others)" >> "$OUTD/env.txt"; echo "${MODE^^}_DONE"; exit 0
fi

CSV=$OUTD/runs.csv
[[ -f $CSV ]] || echo "gpu,host,model,scheme,build,rep,slot,tok_s,rc,temp_pre,temp_post,cool_s,clocks,others,utc,log" > "$CSV"
run1() {  # run1 <model> <scheme> <build> <rep> <slot> <prompt> <tag>
  local m=$1 q=$2 b=$3 r=$4 s=$5 p=$6 tag=$7 log t0 tp tq rc tok cl oth cs
  log="logs/$tag-$m-$q-$b-r$r.log"
  t0=$SECONDS; cool; cs=$((SECONDS - t0)); guard
  tp=$(gtemp); oth=$(others)
  echo "# e2e-adb $GPU $S $tag $m $q $b r$r: $(benv $b) $b/llama_main $p" >> "$CMDLOG"
  A shell "cd $D && $(benv $b) LD_LIBRARY_PATH=$D/$b timeout 1800 $D/$b/llama_main --model_path=$MD/${STEM[$m]}_vulkan_$q.pte \
    --tokenizer_path=tokenizer.model --prompt_file=$p --max_new_tokens=1 --temperature=0 \
    $([[ $tag == prefill ]] && echo --warmup) < /dev/null > run.log 2>&1; echo RC=\$? >> run.log" < /dev/null
  alive || { echo "DEVICE GONE during $tag $m $q $b r$r -- stopping"; exit 3; }
  A pull $D/run.log "$OUTD/$log" > /dev/null
  rc=$(grep -oE '^RC=[0-9]+' "$OUTD/$log" | tail -1 | cut -d= -f2); cl=$(state | tr ',' ';'); tq=$(gtemp)
  [[ -n ${cl// /} ]] || cl="n/a"
  tok=$(grep -o '"prefill_token_per_sec":[0-9.]*' "$OUTD/$log" | head -1 | cut -d: -f2)
  echo "$GPU,$HOSTN,$m,$q,$b,$r,$s,${tok},${rc:-255},$tp,$tq,$cs,$cl,$oth,$(date -u +%FT%TZ),$log" >> "$CSV"
  echo "$tag $m $q $b r$r slot$s tok_s=$tok rc=$rc T=$tp->$tq cool=${cs}s prompt_tokens=$(grep -o '"prompt_tokens":[0-9]*' "$OUTD/$log" | cut -d: -f2)"
}
gen() { grep -v 'PyTorchObserver\|^[IWE] \|^\[sarc_dev\]\|^RC=' "$OUTD/$1"; }

for m in "${MS[@]}"; do for q in "${QS[@]}"; do
  for ((r = 1; r <= REPS; r++)); do
    if (( r % 2 )); then order=(stock sarc); else order=(sarc stock); fi
    run1 $m $q ${order[0]} $r 1 $PROMPT prefill
    run1 $m $q ${order[1]} $r 2 $PROMPT prefill
  done
  if [[ $CHECK == 1 ]]; then
    run1 $m $q stock 0 0 prompt_check.txt check
    run1 $m $q sarc 0 0 prompt_check.txt check
    if cmp -s <(gen "logs/check-$m-$q-stock-r0.log") <(gen "logs/check-$m-$q-sarc-r0.log"); then r=SAME; else r=DIFFER; fi
    echo "$m,$q,$r" >> "$OUTD/nexttoken.csv"; echo "check $m $q next token sarc vs stock: $r"
  fi
done; done

FAILED=$(awk -F, -v ms=",$MODELS," -v qs=",$SCHEMES," 'NR > 1 && $16 ~ /^logs\/prefill/ && ($9 != 0 || $8 == "") && index(ms, ","$3",") && index(qs, ","$4",") {print $3, $4, $5, $6}' "$CSV")
[[ -n $FAILED ]] && while read -r m q b r; do echo "retry $m $q $b r$r"; run1 $m $q $b "${r}x" 9 $PROMPT prefill; done <<< "$FAILED"
echo "state_end: $(state) others_end: $(others)" >> "$OUTD/env.txt"
date -u > "$OUTD/done.txt"; echo E2E_DONE
