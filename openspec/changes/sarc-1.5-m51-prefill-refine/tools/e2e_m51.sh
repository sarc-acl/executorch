#!/bin/bash
# e2e_m51.sh: parent vs candidate end-to-end prefill session on the M51 board (adapted from the 780M's e2e5.sh).
# Protocol: fresh llama_main per run on the board, --warmup, 1 new token, temperature 0; arms interleaved (parent
# first on odd repeats, cand first on even ones); before every run the board cools until the GPU (G3D) temperature
# is at most idle + COOLD C, or has stopped falling for 30 s, or COOLMAX s have passed; failed runs are kept.
# Per run, a sampler on the board records at a short fixed interval: time, gpu_clock, gpu_busy, G3D temperature, the GPU
# thermal cooling state; and the run records gpu_clock_stats (active time per frequency) and gpu_reset_count
# before and after. A run is VALID only if: rc 0; tok/s present; prompt_tokens = expected; generated_tokens = 0;
# the device-state guard passes before and after (driver md5, PAL cfg aside, clock pins read back); no other GPU
# process; at least CLKN samples inside the prefill window; their median gpu_clock >= CLKMIN kHz; no active time
# at any frequency other than the pinned one; GPU cooling state 0 throughout; no GPU reset. Cells with fewer than REPS valid runs per
# arm get extra interleaved pairs (at most EXTRA). Whether the model load was slow is recorded (load_ms).
# One coordinator-hold unit. A board that disappears writes ABORTED and stops the session.
#
# usage: e2e_m51.sh --stage DIR [--out raw] [--reps 5] [--extra 3] [--models 1b,3b,8b] [--schemes 4w,8da4w]
#                   [--prompt prompt_2048.txt] [--tokens 2048] [--clkmin <kHz>] [--clkn 5] [--no-check]
set -uo pipefail
[[ -n ${SARC_HOLD_UNIT:-} ]] || exec env SARC_HOLD_UNIT=1 "$(dirname "$(readlink -f "$0")")/hold.sh" run "timed session e2e_m51.sh $*" "$0" "$@"
source "$(dirname "$(readlink -f "$0")")/dev.sh"
STAGE=""; OUTN=raw; REPS=5; EXTRA=3; MODELS=1b,3b,8b; SCHEMES=4w,8da4w; PROMPT=prompt_2048.txt; TOKENS=2048
CLKMIN=${CLKMIN:-}; CLKN=5; CHECK=1; COOLMAX=${COOLMAX:-300}; COOLD=${COOLD:-5}
while [[ $# -gt 0 ]]; do
  case $1 in
    --stage) STAGE=$2; shift ;; --out) OUTN=$2; shift ;; --reps) REPS=$2; shift ;; --extra) EXTRA=$2; shift ;;
    --models) MODELS=$2; shift ;; --schemes) SCHEMES=$2; shift ;; --prompt) PROMPT=$2; shift ;;
    --tokens) TOKENS=$2; shift ;; --clkmin) CLKMIN=$2; shift ;; --clkn) CLKN=$2; shift ;; --no-check) CHECK=0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done
[[ -n $STAGE && -n $CLKMIN ]] || { sed -n "2,17p" "$0"; echo "(--clkmin: kHz, from the local thresholds)"; exit 2; }
D=$(cd "$STAGE" && pwd); SES=$(basename "$D"); DS=$DEV_ROOT/stage/$SES; O=$D/$OUTN; mkdir -p "$O/logs"
st=$(device_state); [[ $st == ok ]] || { echo "board not fit: $st"; exit 4; }
declare -A STEM=([1b]=llama3_2_1b [3b]=llama3_2_3b [8b]=llama3_1_8b)
IFS=, read -ra MS <<< "$MODELS"; IFS=, read -ra QS <<< "$SCHEMES"
{
  date -u; echo "reps=$REPS extra=$EXTRA prompt=$PROMPT tokens=$TOKENS clkmin=$CLKMIN clkn=$CLKN coolmax=$COOLMAX coold=$COOLD"
  for b in parent cand; do echo "$b env: $(cat $D/$b/env | tr '\n' ' ')"; echo "$b commit: $(cat $D/$b/COMMIT)"; done
  A shell "cd $DS && sha256sum parent/llama_main cand/llama_main parent/$PROMPT parent/prompt_check.txt parent/r1304.txt" < /dev/null
  A shell "getprop ro.soc.model; uname -r; md5sum /vendor/lib64/hw/vulkan.samsung.so; cat /sys/class/devfreq/23400000.sgpu/governor; cat /sys/kernel/gpu/gpu_reset_count" < /dev/null
  echo "clocks: $(clocks)"; echo "others: $(gpu_others)"
  A shell 'for p in /sys/devices/system/cpu/cpufreq/policy*; do echo "$p $(cat $p/scaling_governor) min=$(cat $p/scaling_min_freq) max=$(cat $p/scaling_max_freq)"; done' < /dev/null
  for m in "${MS[@]}"; do for q in "${QS[@]}"; do echo "model $m $q $(A shell "ls -l $DEV_ROOT/models/${STEM[$m]}_${q}_embq_ctx3072.pte" < /dev/null | awk '{print $5}')"; done; done
} > "$O/env.txt" 2>&1
sleep 60; IDLE=$(gtemp); echo "idle_temp=$IDLE" >> "$O/env.txt"
cool() { local t0=$SECONDS t best=999 tb=$SECONDS
  while :; do t=$(gtemp); [[ $t =~ ^[0-9]+$ ]] || break
    (( t < best )) && { best=$t; tb=$SECONDS; }
    (( t <= IDLE + COOLD || SECONDS - tb >= 30 || SECONDS - t0 >= COOLMAX )) && break; sleep 5; done; }
CSV=$O/runs.csv
[[ -f $CSV ]] || echo "gpu,model,scheme,build,rep,slot,tok_s,rc,temp_pre,temp_post,cool_s,guard_pre,guard_post,others,utc,log,prompt_tokens,generated_tokens,prefill_ms,load_ms,clk_n,clk_med_khz,clk_min_khz,busy_med,temp_max,throttle_max,other_freq_us,resets,valid,reason" > "$CSV"
run1() {  # run1 <model> <scheme> <build> <rep> <slot> <prompt> <tag> <expected tokens>
  local m=$1 q=$2 b=$3 r=$4 s=$5 p=$6 tag=$7 want=$8 log t0 tp tq oth cs g0 g1 benv w
  log="logs/$tag-$m-$q-$b-r$r.log"
  t0=$SECONDS; cool; cs=$((SECONDS - t0)); tp=$(gtemp); oth=$(gpu_others); g0=$(device_state)
  [[ $g0 == device_gone || $g0 == marker* ]] && { echo "board gone before $log" | tee -a "$ART/ABORTED"; exit 3; }
  benv=$(tr '\n' ' ' < "$D/$b/env"); w=""; [[ $tag == prefill ]] && w=--warmup
  A shell "cd $DS/$b && rm -f run.log run.clk run.rc run.pre run.post
    (cat /sys/kernel/gpu/gpu_clock_stats; cat /sys/kernel/gpu/gpu_reset_count) > run.pre
    (while :; do echo \$(date +%s%N) \$(cat /sys/kernel/gpu/gpu_clock) \$(cat /sys/kernel/gpu/gpu_busy) \$(cat /sys/class/thermal/thermal_zone4/temp) \$(cat /sys/class/thermal/cooling_device7/cur_state); sleep 0.05; done > run.clk) &
    sp=\$!
    $benv LD_LIBRARY_PATH=$DS/$b timeout 1190 ./llama_main --model_path=$DEV_ROOT/models/${STEM[$m]}_${q}_embq_ctx3072.pte --tokenizer_path=$DEV_ROOT/models/tokenizer.model --prompt_file=$p --max_new_tokens=1 --temperature=0 $w < /dev/null > run.log 2>&1
    echo \$? > run.rc; kill \$sp; wait \$sp 2>/dev/null
    (cat /sys/kernel/gpu/gpu_clock_stats; cat /sys/kernel/gpu/gpu_reset_count) > run.post" < /dev/null > /dev/null 2>&1
  alive || { echo "board gone during $log $(date -u +%FT%TZ)" | tee -a "$ART/ABORTED"; exit 3; }
  A pull "$DS/$b/run.log" "$O/$log" > /dev/null; A pull "$DS/$b/run.clk" "$O/${log%.log}.clk" > /dev/null
  A pull "$DS/$b/run.pre" "$O/${log%.log}.pre" > /dev/null; A pull "$DS/$b/run.post" "$O/${log%.log}.post" > /dev/null
  local rc; rc=$(A shell "cat $DS/$b/run.rc" < /dev/null | tr -d '\r\n'); tq=$(gtemp); g1=$(device_state)
  python3 - "$O/$log" "$O/${log%.log}" "$want" "$CLKMIN" "$CLKN" "${rc:-255}" "$oth" "$tag" "$g0" "$g1" "$M51_GPU_KHZ" <<'PY' > "$O/.row"
import json, statistics as st, sys
log, base, want, clkmin, clkn, rc, oth, tag, g0, g1, gpu = sys.argv[1:12]; gpu = int(gpu)
obs = None
for line in open(log, errors="replace"):
    i = line.find("PyTorchObserver")
    if i >= 0:
        try: obs = json.loads(line[line.index("{", i):])
        except Exception: pass
tok = pt = gt = ms = lm = ""; rows = []
if obs:
    tok = obs.get("prefill_token_per_sec", ""); pt = obs.get("prompt_tokens", ""); gt = obs.get("generated_tokens", "")
    if obs.get("model_load_start_ms") and obs.get("model_load_end_ms"): lm = obs["model_load_end_ms"] - obs["model_load_start_ms"]
    a = obs.get("inference_start_ms"); b = obs.get("prompt_eval_end_ms")
    if a and b:
        ms = b - a
        for l in open(base + ".clk"):
            f = l.split()
            if len(f) == 5 and all(x.lstrip("-").isdigit() for x in f) and a * 1e6 <= int(f[0]) <= b * 1e6: rows.append([int(x) for x in f])
def stats(p):
    fr, rs = {}, None
    try:
        for l in open(p):
            f = l.split()
            if len(f) == 2 and f[0].isdigit(): fr[int(f[0])] = int(f[1])
            elif len(f) == 1 and f[0].isdigit(): rs = int(f[0])
    except OSError: pass
    return fr, rs
f0, r0 = stats(base + ".pre"); f1, r1 = stats(base + ".post")
other = sum(f1.get(k, 0) - f0.get(k, 0) for k in f1 if k != gpu) if f0 and f1 else ""
resets = (r1 - r0) if r0 is not None and r1 is not None else ""
n = len(rows); act = [r for r in rows if r[1] > 0]
cm = st.median(r[1] for r in act) if act else ""; cmin = min(r[1] for r in act) if act else ""
bm = st.median(r[2] for r in rows) if rows else ""; tmax = round(max(r[3] for r in rows) / 1000, 1) if rows else ""
thr = max(r[4] for r in rows) if rows else ""
reason = []
if rc != "0": reason.append("rc")
if tok == "": reason.append("no_tok_s")
if want != "any" and str(pt) != want: reason.append("prompt_tokens")
if tag == "prefill" and str(gt) != "0": reason.append("generated_tokens")
if g0 != "ok" or g1 != "ok": reason.append("guard")
if oth: reason.append("other_gpu_process")
if len(act) < int(clkn): reason.append("clock_unsampled")
elif cm < float(clkmin): reason.append("clock_low")
if other == "" or other > 0: reason.append("other_frequency")
if thr == "" or thr > 0: reason.append("thermal_throttle")
if resets == "" or resets > 0: reason.append("gpu_reset")
print(",".join(str(x) for x in [tok, pt, gt, ms, lm, len(act), cm, cmin, bm, tmax, thr, other, resets, 0 if reason else 1, "+".join(reason)]))
PY
  local tok pt gt ms lm n cm cmin bm tmax thr other resets valid reason
  IFS=, read -r tok pt gt ms lm n cm cmin bm tmax thr other resets valid reason < "$O/.row"
  echo "m51,$m,$q,$b,$r,$s,$tok,${rc:-255},$tp,$tq,$cs,${g0// /_},${g1// /_},$oth,$(date -u +%FT%TZ),$log,$pt,$gt,$ms,$lm,$n,$cm,$cmin,$bm,$tmax,$thr,$other,$resets,$valid,$reason" >> "$CSV"
  echo "$tag $m $q $b r$r slot$s tok_s=$tok ms=$ms load=$lm rc=$rc T=$tp->$tq cool=${cs}s clk=$cm/$cmin n=$n busy=$bm thr=$thr other=$other valid=$valid $reason"
  [[ $g1 == device_gone ]] && { echo "board gone after $log" | tee -a "$ART/ABORTED"; exit 3; }
}
nvalid() { awk -F, -v m=$1 -v q=$2 -v b=$3 'NR > 1 && $2 == m && $3 == q && $4 == b && $16 ~ /^logs\/prefill/ && $29 == 1 {n++} END {print n + 0}' "$CSV"; }
gen() { grep -v 'PyTorchObserver\|^[IWE] \|^\[sarc_dev\]' "$O/$1"; }
for m in "${MS[@]}"; do for q in "${QS[@]}"; do
  r=1
  while :; do
    if (( r % 2 )); then order=(parent cand); else order=(cand parent); fi
    run1 $m $q ${order[0]} $r 1 $PROMPT prefill $TOKENS
    run1 $m $q ${order[1]} $r 2 $PROMPT prefill $TOKENS
    r=$((r + 1))
    (( r <= REPS )) && continue
    (( $(nvalid $m $q parent) >= REPS && $(nvalid $m $q cand) >= REPS )) && break
    (( r > REPS + EXTRA )) && { echo "CELL $m $q: fewer than $REPS valid runs after $EXTRA extra pairs"; break; }
  done
  if [[ $CHECK == 1 ]]; then
    for c in prompt_check.txt:check:1972 r1304.txt:unaligned:any; do IFS=: read -r cp ct cn <<< "$c"
      run1 $m $q parent 0 0 $cp $ct "$cn"; run1 $m $q cand 0 0 $cp $ct "$cn"; done
    x=(); for k in "prefill:r1" "check:r0" "unaligned:r0"; do IFS=: read -r kt kr <<< "$k"
      if cmp -s <(gen "logs/$kt-$m-$q-parent-$kr.log") <(gen "logs/$kt-$m-$q-cand-$kr.log"); then x+=(SAME); else x+=(DIFFER); fi; done
    echo "$m,$q,$PROMPT:${x[0]},prompt_check.txt:${x[1]},r1304.txt:${x[2]}" >> "$O/nexttoken.csv"; echo "nexttoken $m $q ${x[*]}"
  fi
done; done
echo "others_end: $(gpu_others)" >> "$O/env.txt"; echo "clocks_end: $(clocks)" >> "$O/env.txt"; date -u > "$O/done.txt"; echo E2E_DONE
