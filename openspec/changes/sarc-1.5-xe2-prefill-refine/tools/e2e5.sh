#!/bin/bash
# e2e5.sh: parent vs candidate end-to-end prefill session on the Arc Pro B70, card b70-0 (fedora-gpu-eval).
# Protocol = openspec/changes/sarc-1.5-e2e-benchmark/kit/host/e2e.sh (fresh llama_main per run, --warmup,
# 1 new token, temperature 0, cool to idle+5 C max 120 s, builds interleaved parent->cand on odd repeats and
# cand->parent on even ones, failed runs kept), plus what this task adds:
#   - the GT clock (freq0/act_freq), throttle status, card energy and package temperature are sampled every 0.1 s during each run
#     (logs/<run>.clk) and summarised over the measured execution window of that run;
#   - a run is VALID only if rc = 0, tok/s present, prompt_tokens = <expected>, generated_tokens = 0, no other GPU
#     process, the median clock in the measured window >= CLKMIN MHz and no throttled sample.
#   - CLKMIN is required: --clkmin, or the value the calibration session stored (host.sh CLKMIN_FILE). Only
#     --calibrate (the baseline / A-A session, started from a cool idle card) runs without one; when all its
#     cells are complete it stores the idle temperature and CLKMIN = 97 % of the lowest per-run median clock.
#   - a GPU process this campaign did not start aborts the session (exit 76) before a launch, and a run during
#     which one appears is invalid and also aborts it.
#   - next tokens are compared only between runs that ran to completion on the expected prompt and printed text.
#   - exit status 0 and "E2E5_OK" in done.txt only if every cell has REPS valid runs per build and every
#     next-token comparison is SAME; otherwise 1 and the reasons. Invalid runs stay in runs.csv with the
#     reason; cells with fewer than REPS valid runs per build get extra interleaved pairs (at most EXTRA).
# One GPU job at a time: everything runs under the gpu-lab lock.
#
# usage: e2e5.sh --stage DIR --out NAME --lock UUID [--reps 5] [--extra 3] [--models 1b,3b,8b] [--schemes 4w,8da4w]
#                [--prompt prompt_2048.txt] [--tokens 2048] [--clkmin MHz | --calibrate] [--no-check]
#   DIR/{parent,cand}/{llama_main,libllama_runner.so,[env]}, DIR/prompt_*.txt; output in DIR/NAME/
set -uo pipefail
STAGE=""; OUTN=raw; LOCK=""; REPS=5; EXTRA=3; MODELS=1b,3b,8b; SCHEMES=4w,8da4w
PROMPT=prompt_2048.txt; TOKENS=2048; CLKMIN=""; CALIB=0; CHECK=1; COOLMAX=120; MROOT=/mnt/linux-share/models
while [[ $# -gt 0 ]]; do
  case $1 in
    --stage) STAGE=$2; shift ;; --out) OUTN=$2; shift ;; --lock) LOCK=$2; shift ;;
    --reps) REPS=$2; shift ;; --extra) EXTRA=$2; shift ;; --models) MODELS=$2; shift ;;
    --schemes) SCHEMES=$2; shift ;; --prompt) PROMPT=$2; shift ;; --tokens) TOKENS=$2; shift ;;
    --clkmin) CLKMIN=$2; shift ;; --calibrate) CALIB=1 ;; --no-check) CHECK=0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done
[[ -n $STAGE && -n $LOCK ]] || { sed -n '2,25p' "$0"; exit 2; }
D=$(cd "$STAGE" && pwd); cd "$D" || exit 2
O=$D/$OUTN; mkdir -p "$O/logs"
exec 9>>"$HOME/.cache/gpu-lab/lock-$LOCK"; flock -w 900 9 || { echo "gpu-lab lock busy"; exit 75; }
. "$(dirname "$(readlink -f "$0")")/host.sh"
[[ -z $CLKMIN && $CALIB == 0 && -s $CLKMIN_FILE ]] && CLKMIN=$(<$CLKMIN_FILE)
[[ $CALIB == 1 ]] && CLKMIN=${CLKMIN:-0}
[[ -n $CLKMIN ]] || { echo "no clock threshold: run the calibration session first (--calibrate) or pass --clkmin" >&2; exit 2; }
[[ $CALIB == 0 && $CLKMIN -le 0 ]] && { echo "clock threshold must be > 0 outside --calibrate" >&2; exit 2; }
FAILS=()
finish() { # finish <status> [reason...]: record the session status and exit
  local st=$1; shift; echo "others_end: $(others)" >> "$O/env.txt"
  { date -u; echo "$st $*"; } > "$O/done.txt"; echo "$st $*"; [[ $st == E2E5_OK ]]; exit; }
gtemp() { echo $(( $(<$HW/temp2_input) / 1000 )); }
others() { gpu_others; }
sampler() {  # sampler <file>: epoch_us act_freq_MHz throttle_status card_energy_uJ pkg_temp_mC, every 0.1 s until killed
  while :; do
    printf '%s %s %s %s %s\n' "${EPOCHREALTIME/./}" "$(<$FREQ/act_freq)" "$(<$FREQ/throttle/status)" "$(<$HW/energy1_input)" "$(<$HW/temp2_input)"
    sleep 0.1
  done > "$1" 2>/dev/null
}
{
  date -u; hostname; uname -r; echo "lock=$LOCK reps=$REPS extra=$EXTRA prompt=$PROMPT tokens=$TOKENS clkmin=$CLKMIN calibrate=$CALIB"
  for b in parent cand; do sha256sum $b/llama_main $b/libllama_runner.so; echo "$b env: $(cat $b/env 2>/dev/null | tr '\n' ' ')"; cat $b/COMMIT 2>/dev/null; done
  sha256sum prompt_*.txt; cat STAGE.md 2>/dev/null
  vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName|driverInfo|apiVersion'
  echo "freq0 MHz: min=$(<$FREQ/min_freq) max=$(<$FREQ/max_freq) rp0=$(<$FREQ/rp0_freq) rpe=$(<$FREQ/rpe_freq) rpn=$(<$FREQ/rpn_freq) profile=$(<$FREQ/power_profile) ETVK_DEVICE_INDEX=$ETVK_DEVICE_INDEX"
  echo "others: $(others)"
} > "$O/env.txt" 2>&1
declare -A STEM=([1b]=llama-3.2-1b:llama3_2-1b [3b]=llama-3.2-3b:llama3_2-3b [8b]=llama-3.1-8b:llama3_1-8b)
IFS=, read -ra MS <<< "$MODELS"; IFS=, read -ra QS <<< "$SCHEMES"
pte()  { local MD ST; IFS=: read -r MD ST <<< "${STEM[$1]}"; echo "$MROOT/$MD/exported/${ST}_vulkan_$2.pte"; }
tokz() { local MD ST; IFS=: read -r MD ST <<< "${STEM[$1]}"; echo "$MROOT/$MD/original/tokenizer.model"; }
for m in "${MS[@]}"; do for q in "${QS[@]}"; do
  echo "model $m $q $(sha256sum "$(pte $m $q)" | cut -c1-16) $(pte $m $q)" >> "$O/env.txt"; done; done
sleep 60; IDLE=$(gtemp); echo "idle_temp=$IDLE" >> "$O/env.txt"
cool() { local t0=$SECONDS t; while :; do t=$(gtemp); [[ $t -le $((IDLE + 5)) || $((SECONDS - t0)) -ge $COOLMAX ]] && break; sleep 5; done; }
CSV=$O/runs.csv
[[ -f $CSV ]] || echo "gpu,host,model,scheme,build,rep,slot,tok_s,rc,temp_pre,temp_post,cool_s,clocks,others,utc,log,prompt_tokens,generated_tokens,prefill_ms,clk_n,clk_med_mhz,clk_min_mhz,throttled_n,power_avg_w,temp_max,valid,reason" > "$CSV"
run1() {  # run1 <model> <scheme> <build> <rep> <slot> <prompt> <tag> <expected tokens>
  local m=$1 q=$2 b=$3 r=$4 s=$5 p=$6 tag=$7 want=$8 log t0 tp tq rc oth cs sp
  log="logs/$tag-$m-$q-$b-r$r.log"
  t0=$SECONDS; cool; cs=$((SECONDS - t0)); tp=$(gtemp); oth=$(others | tr ',' ';')
  [[ -n $oth ]] && finish E2E5_ABORTED "other GPU process before $tag $m $q $b r$r: $oth"
  others_watch "$O/${log%.log}.others" 9>&- & local wp=$!
  local benv=(); [[ -f $D/$b/env ]] && mapfile -t benv < "$D/$b/env"
  sampler "$O/${log%.log}.clk" 9>&- & sp=$!
  env "${benv[@]}" LD_LIBRARY_PATH=$D/$b timeout 1800 "$D/$b/llama_main" --model_path "$(pte $m $q)" \
    --tokenizer_path "$(tokz $m)" --prompt_file "$p" --max_new_tokens 1 --temperature 0 \
    $([[ $tag == prefill ]] && echo --warmup) < /dev/null > "$O/$log" 2>&1 9>&-
  rc=$?; kill $sp $wp 2>/dev/null; wait $sp $wp 2>/dev/null; tq=$(gtemp)
  oth=$(cut -d' ' -f2- "$O/${log%.log}.others" | sort -u | tr -d '\n' | tr ',' ';')
  python3 - "$O/$log" "$O/${log%.log}.clk" "$want" "$CLKMIN" "$rc" "$oth" "$tag" <<'PY' > "$O/.row"
import json, re, statistics as st, sys
log, clk, want, clkmin, rc, oth, tag = sys.argv[1:8]
obs = None
for line in open(log, errors="replace"):
    i = line.find("PyTorchObserver")
    if i >= 0:
        try: obs = json.loads(line[line.index("{", i):])
        except Exception: pass
tok = pt = gt = ms = ""
rows = []
if obs:
    tok = obs.get("prefill_token_per_sec", ""); pt = obs.get("prompt_tokens", ""); gt = obs.get("generated_tokens", "")
    a, b = obs.get("model_execution_start_ms"), obs.get("model_execution_end_ms")
    # prefill window: from inference start to prompt-eval end when available, else the execution window
    a = obs.get("inference_start_ms", a); b2 = obs.get("prompt_eval_end_ms", b)
    if a and b2:
        ms = b2 - a
        for l in open(clk):
            f = l.split()
            if len(f) == 5 and a * 1000 <= int(f[0]) <= b2 * 1000: rows.append([int(x) for x in f])
n = len(rows)
med = lambda k, d: (round(st.median(r[k] for r in rows) / d, 1) if rows else "")
cm = med(1, 1); cmin = min(r[1] for r in rows) if rows else ""
# power: card energy counter (uJ) over the sampled window (time in us); thr = samples with throttle/status != 0
pw = round((rows[-1][3] - rows[0][3]) / (rows[-1][0] - rows[0][0]), 1) if n > 1 else ""
thr = sum(1 for r in rows if r[2])
reason = []
if rc != "0": reason.append("rc")
if tok == "": reason.append("no_tok_s")
if str(pt) != want: reason.append("prompt_tokens")
if tag == "prefill" and str(gt) != "0": reason.append("generated_tokens")
if oth: reason.append("other_gpu_process")
if n < 2: reason.append("clock_unsampled")
elif cm < float(clkmin): reason.append("clock_low")
elif thr: reason.append("throttled")
valid = 0 if reason else 1
print(",".join(str(x) for x in [tok, pt, gt, ms, n, cm, cmin, thr, pw, round(max(r[4] for r in rows) / 1000) if rows else "", valid, "+".join(reason)]))
PY
  IFS=, read -r tok pt gt ms n cm cmin bm pw tmax valid reason < "$O/.row"
  echo "xe2-b70,$(hostname),$m,$q,$b,$r,$s,$tok,$rc,$tp,$tq,$cs,act_med=${cm}MHz,$oth,$(date -u +%FT%TZ),$log,$pt,$gt,$ms,$n,$cm,$cmin,$bm,$pw,$tmax,$valid,$reason" >> "$CSV"
  echo "$tag $m $q $b r$r slot$s tok_s=$tok ms=$ms rc=$rc T=$tp->$tq cool=${cs}s clk=$cm/$cmin MHz n=$n throttled=$bm valid=$valid $reason"
  [[ -n $oth ]] && finish E2E5_ABORTED "other GPU process during $tag $m $q $b r$r: $oth"
}
# toklog <tag> <model> <scheme> <build>: log of the first run that completed on the expected prompt (clock and
# throttle do not matter for a token comparison); empty if there is none.
toklog() { awk -F, -v p="logs/$1-$2-$3-$4-r" 'NR > 1 && index($16, p) == 1 && $27 !~ /rc|no_tok_s|prompt_tokens|generated_tokens|other_gpu_process/ {print $16; exit}' "$CSV"; }
cmptok() { # cmptok <tag> <model> <scheme>: SAME / DIFFER / INVALID (a side has no completed run or printed no text)
  local a b ga gb; a=$(toklog $1 $2 $3 parent); b=$(toklog $1 $2 $3 cand)
  [[ -n $a && -n $b ]] || { echo INVALID; return; }
  ga=$(gen "$a"); gb=$(gen "$b"); [[ -n $ga && -n $gb ]] || { echo INVALID; return; }
  [[ $ga == "$gb" ]] && echo SAME || echo DIFFER; }
nvalid() { awk -F, -v m=$1 -v q=$2 -v b=$3 'NR > 1 && $3 == m && $4 == q && $5 == b && $16 ~ /^logs\/prefill/ && $26 == 1 {n++} END {print n + 0}' "$CSV"; }
gen() { grep -v 'PyTorchObserver\|^[IWE] \|^\[sarc_dev\]' "$O/$1"; }
for m in "${MS[@]}"; do for q in "${QS[@]}"; do
  [[ -f $(pte $m $q) ]] || { echo "missing $(pte $m $q)"; FAILS+=("$m-$q:missing_model"); continue; }
  r=1
  while :; do
    if (( r % 2 )); then order=(parent cand); else order=(cand parent); fi
    run1 $m $q ${order[0]} $r 1 $PROMPT prefill $TOKENS
    run1 $m $q ${order[1]} $r 2 $PROMPT prefill $TOKENS
    r=$((r + 1))
    (( r <= REPS )) && continue
    (( $(nvalid $m $q parent) >= REPS && $(nvalid $m $q cand) >= REPS )) && break
    (( r > REPS + EXTRA )) && { echo "CELL $m $q: fewer than $REPS valid runs after $EXTRA extra pairs"; FAILS+=("$m-$q:incomplete"); break; }
  done
  if [[ $CHECK == 1 ]]; then
    run1 $m $q parent 0 0 prompt_check.txt check 1972
    run1 $m $q cand 0 0 prompt_check.txt check 1972
    x=$(cmptok check $m $q); y=$(cmptok prefill $m $q)
    echo "$m,$q,$PROMPT:$y,prompt_check.txt:$x" >> "$O/nexttoken.csv"; echo "nexttoken $m $q $PROMPT=$y check=$x"
    [[ $x == SAME && $y == SAME ]] || FAILS+=("$m-$q:nexttoken_${y}_$x")
  fi
done; done
if [[ $CALIB == 1 && ${#FAILS[@]} == 0 ]]; then
  echo $((IDLE * 1000)) > "$IDLE_FILE"
  awk -F, 'NR > 1 && $16 ~ /^logs\/prefill/ && $26 == 1 && $21 != "" {if (m == "" || $21 + 0 < m) m = $21 + 0} END {printf "%d\n", m * 0.97}' "$CSV" > "$CLKMIN_FILE"
  echo "calibration: idle_temp_mc=$(<$IDLE_FILE) clkmin_mhz=$(<$CLKMIN_FILE)" | tee -a "$O/env.txt"
  [[ $(<$CLKMIN_FILE) -gt 0 ]] || FAILS+=("calibration:clock_not_readable")
fi
[[ ${#FAILS[@]} == 0 ]] && finish E2E5_OK
finish E2E5_INCOMPLETE "${FAILS[*]}"
