#!/bin/bash
# e2e5.sh: parent vs candidate end-to-end prefill session on the Radeon RX 7900 XTX (runs on the GPU host).
# Protocol = openspec/changes/sarc-1.5-e2e-benchmark/kit/host/e2e.sh (fresh llama_main per run, --warmup,
# 1 new token, temperature 0, cool to idle + 5 C, arms interleaved parent->cand on odd repeats and cand->parent
# on even ones, failed runs kept), plus:
#   - the model file of a cell is read into the page cache before the first process of the cell (owner
#     decision D5); the model load time of every run is recorded (load_ms);
#   - sclk, busy %, power, max(edge, junction, mem) temperature and gpu_metrics indep_throttle_status are sampled
#     every sample_period_s (thresholds.txt, 5 ms: a 1B prefill is ~90 ms) during each run (logs/<run>.clk; sampler.py) and summarised over the prefill window;
#   - foreign GPU users (fuser on /dev/dri, runner programs) are checked before and after each run and polled
#     every 0.5 s during it; host builds (compilers, linkers, ninja) of anyone are waited out before each run
#     and polled during it;
#   - the card drives the host's display: gpu_busy_percent is sampled 10 x 50 ms before every run (busy_pre_max column);
#     a run above busy_pre_max of thresholds.txt (set from the A/A session; 100 = not rejecting) is invalid (foreign_busy);
#   - a run is VALID only if rc = 0, tok/s present, prompt_tokens = <expected>, generated_tokens = 0, no foreign
#     GPU user and no host build before / during / after it, >= CLKN clock samples in the prefill window, a
#     median clock >= CLKMIN MHz there, and no thermal throttle bit (indep_throttle_status bits 32-47, masked by
#     thermal_mask of thresholds.txt) in it.
#     Invalid runs stay in runs.csv with the reason; cells with fewer than REPS valid runs per arm get extra
#     interleaved pairs (at most EXTRA).
#   - next token parent vs candidate on the timed prompt (repeat 1), prompt_real_2048.txt and prompt_check.txt.
# One GPU job at a time: everything runs under the gpu-lab lock. One unit of the coordinator hold.
#
# usage: e2e5.sh --stage DIR --out NAME [--reps 5] [--extra 3] [--models 1b,3b,8b] [--schemes 4w,8da4w]
#                [--prompt prompt_2048.txt] [--tokens 2048] [--clkmin MHz] [--clkn 5] [--no-check]
#   DIR/{parent,cand}/{llama_main,libllama_runner.so,[env]}, DIR/prompt_*.txt; output in DIR/NAME/
set -uo pipefail
[[ -n ${SARC_HOLD_UNIT:-} ]] || exec env SARC_HOLD_UNIT=1 "$(dirname "$(readlink -f "$0")")/hold.sh" run "timed session e2e5.sh $*" "$0" "$@"
source "$(dirname "$(readlink -f "$0")")/env.sh"
STAGE=""; OUTN=raw; REPS=5; EXTRA=3; MODELS=1b,3b,8b; SCHEMES=4w,8da4w
PROMPT=prompt_2048.txt; TOKENS=2048; CLKMIN=$(sed -n 's/^clkmin=//p' $T/thresholds.txt 2>/dev/null); TMASK=$(sed -n 's/^thermal_mask=//p' $T/thresholds.txt 2>/dev/null); CLKN=5; CHECK=1; COOLMAX=300
SPER=$(sed -n 's/^sample_period_s=//p' $T/thresholds.txt 2>/dev/null); SPER=${SPER:-0.005}
BUSYMAX=$(sed -n 's/^busy_pre_max=//p' $T/thresholds.txt 2>/dev/null); BUSYMAX=${BUSYMAX:-100}
while [[ $# -gt 0 ]]; do
  case $1 in
    --stage) STAGE=$2; shift ;; --out) OUTN=$2; shift ;;
    --reps) REPS=$2; shift ;; --extra) EXTRA=$2; shift ;; --models) MODELS=$2; shift ;;
    --schemes) SCHEMES=$2; shift ;; --prompt) PROMPT=$2; shift ;; --tokens) TOKENS=$2; shift ;;
    --clkmin) CLKMIN=$2; shift ;; --clkn) CLKN=$2; shift ;; --no-check) CHECK=0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done
[[ -n $STAGE ]] || { sed -n '2,26p' "$0"; exit 2; }
CLKMIN=${CLKMIN:-0}; TMASK=${TMASK:-0xffff}
D=$(cd "$STAGE" && pwd); cd "$D" || exit 2
O=$D/$OUTN; mkdir -p "$O/logs"
while [[ -e $A/.building ]]; do sleep 30; done
exec 9>>"$LOCKF"; flock -w 1800 9 || { echo "gpu-lab lock busy"; exit 75; }
{
  date -u; hostname; uname -r; echo "lock=$LOCK reps=$REPS extra=$EXTRA prompt=$PROMPT tokens=$TOKENS clkmin=$CLKMIN clkn=$CLKN thermal_mask=$TMASK"
  for b in parent cand; do sha256sum $b/llama_main $b/libllama_runner.so; echo "$b env: $(tr '\n' ' ' < $b/env 2>/dev/null)"; cat $b/COMMIT 2>/dev/null; done
  sha256sum prompt_*.txt; cat STAGE.md 2>/dev/null
  echo "VK_ICD_FILENAMES=$VK_ICD_FILENAMES ETVK_DEVICE_INDEX=$ETVK_DEVICE_INDEX"
  vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName|driverInfo|apiVersion'
  echo "dpm: $(<$CARD/power_dpm_force_performance_level) profile: $(grep '\*' $CARD/pp_power_profile_mode | awk '{print $1, $2}')"
  echo "sclk levels: $(tr '\n' ' ' < $CARD/pp_dpm_sclk) mclk levels: $(tr '\n' ' ' < $CARD/pp_dpm_mclk)"
  echo "power cap: $(<$HW/power1_cap) uW; vram used $(<$CARD/mem_info_vram_used) of $(<$CARD/mem_info_vram_total)"
  echo "others: $($T/others.sh)"; echo "load: $(</proc/loadavg)"; free -g | sed -n 2p
} > "$O/env.txt" 2>&1
declare -A STEM=([1b]=llama3_2-1b [3b]=llama3_2-3b [8b]=llama3_1-8b)
IFS=, read -ra MS <<< "$MODELS"; IFS=, read -ra QS <<< "$SCHEMES"
pte() { echo "$MFLAT/${STEM[$1]}_vulkan_$2.pte"; }
TOK=$MFLAT/tokenizer.model
for m in "${MS[@]}"; do for q in "${QS[@]}"; do
  echo "model $m $q $(stat -L -c %s "$(pte $m $q)") $(readlink -f "$(pte $m $q)")" >> "$O/env.txt"; done; done
sleep 60; IDLE=$(gtemp); echo "idle_temp=$IDLE" >> "$O/env.txt"
cool() {  # until idle + 5 C, or the temperature has not fallen for 30 s, or COOLMAX
  local t0=$SECONDS t best=999 tb=$SECONDS
  while :; do
    t=$(gtemp)
    (( t < best )) && { best=$t; tb=$SECONDS; }
    (( t <= IDLE + 5 || SECONDS - tb >= 30 || SECONDS - t0 >= COOLMAX )) && break
    sleep 2
  done
}
waitbuild() { local t0=$SECONDS; while [[ $($T/others.sh) == *build=?* ]] && (( SECONDS - t0 < 7200 )); do sleep 20; done; echo $((SECONDS - t0)); }
CSV=$O/runs.csv
[[ -f $CSV ]] || echo "gpu,host,model,scheme,build,rep,slot,tok_s,rc,temp_pre,temp_post,cool_s,clocks,others,utc,log,prompt_tokens,generated_tokens,prefill_ms,clk_n,clk_med_mhz,clk_min_mhz,busy_med,power_med_w,temp_max,valid,reason,load_ms,throttle,load1,buildwait_s,vram_pre_mb,busy_pre_max" > "$CSV"
run1() {  # run1 <model> <scheme> <build> <rep> <slot> <prompt> <tag> <expected tokens>
  local m=$1 q=$2 b=$3 r=$4 s=$5 p=$6 tag=$7 want=$8 log t0 tp tq rc oth oth2 cs sp gp lp bw vr
  log="logs/$tag-$m-$q-$b-r$r.log"
  bw=$(waitbuild)
  t0=$SECONDS; cool; cs=$((SECONDS - t0)); tp=$(gtemp); oth=$($T/others.sh); vr=$(( $(<$CARD/mem_info_vram_used) / 1048576 ))
  local benv=(); [[ -f $D/$b/env ]] && mapfile -t benv < "$D/$b/env"
  local clk=$O/${log%.log}.clk bp=0 i x; rm -f "$clk"
  for i in 1 2 3 4 5 6 7 8 9 10; do x=$(<$CARD/gpu_busy_percent); (( x > bp )) && bp=$x; sleep 0.05; done   # foreign busy % before the run (the card drives the display)
  /usr/bin/python3 $T/sampler.py "$clk" $SPER 9>&- & sp=$!
  while [[ ! -s $clk ]] && kill -0 $sp 2>/dev/null; do sleep 0.01; done
  env "${benv[@]}" LD_LIBRARY_PATH=$D/$b timeout 1800 "$D/$b/llama_main" --model_path "$(pte $m $q)" \
    --tokenizer_path "$TOK" --prompt_file "$p" --max_new_tokens 1 --temperature 0 \
    $([[ $tag == prefill ]] && echo --warmup) < /dev/null > "$O/$log" 2>&1 9>&- &
  lp=$!
  ( while kill -0 $lp 2>/dev/null; do echo "$(date +%s%N | cut -c1-16) $($T/others.sh $lp)"; sleep 0.5; done ) > "$O/${log%.log}.oth" 9>&- &
  gp=$!
  wait $lp; rc=$?
  wait $gp; kill $sp 2>/dev/null; wait $sp 2>/dev/null; tq=$(gtemp); oth2=$($T/others.sh)
  /usr/bin/python3 - "$O/$log" "$clk" "$want" "$CLKMIN" "$rc" "$oth" "$oth2" "$tag" "$CLKN" "$O/${log%.log}.oth" "$TMASK" "$bp" "$BUSYMAX" <<'PY' > "$O/.row"
import json, statistics as st, sys
log, clk, want, clkmin, rc, oth, oth2, tag, clkn, othf, tmask, bp, busymax = sys.argv[1:14]
obs = None
for line in open(log, errors="replace"):
    i = line.find("PyTorchObserver")
    if i >= 0:
        try: obs = json.loads(line[line.index("{", i):])
        except Exception: pass
tok = pt = gt = ms = lms = ""
rows = []
if obs:
    tok = obs.get("prefill_token_per_sec", ""); pt = obs.get("prompt_tokens", ""); gt = obs.get("generated_tokens", "")
    a, b = obs.get("inference_start_ms"), obs.get("prompt_eval_end_ms")
    if obs.get("model_load_start_ms") and obs.get("model_load_end_ms"):
        lms = obs["model_load_end_ms"] - obs["model_load_start_ms"]
    if a and b:
        ms = b - a
        for l in open(clk):
            f = l.split()
            if len(f) == 7 and a * 1000 <= int(f[0]) <= b * 1000:
                rows.append([int(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[4]), int(f[5], 16), int(f[6])])
n = len(rows)
med = lambda k, d: (round(st.median(r[k] for r in rows) / d, 1) if rows else "")
cm = med(1, 1e6); cmin = round(min(r[1] for r in rows) / 1e6, 1) if rows else ""
thr = 0
for r in rows: thr |= max(r[5], 0)
during = [l.split(" ", 1)[1].strip() for l in open(othf) if l.strip()]
fg = [x for x in [oth, oth2] + during if not x.startswith("gpu= ")]
fb = [x for x in [oth, oth2] + during if not x.endswith("build=")]
reason = []
if rc != "0": reason.append("rc")
if tok == "": reason.append("no_tok_s")
if str(pt) != want: reason.append("prompt_tokens")
if tag == "prefill" and str(gt) != "0": reason.append("generated_tokens")
if fg: reason.append("other_gpu_process")
if fb: reason.append("host_build")
if n < int(clkn): reason.append("clock_unsampled")
elif cm < float(clkmin): reason.append("clock_low")
if (thr >> 32) & int(tmask, 16): reason.append("thermal_throttle")
if int(bp) > int(busymax): reason.append("foreign_busy")
valid = 0 if reason else 1
print(",".join(str(x) for x in [tok, pt, gt, ms, n, cm, cmin, med(2, 1), med(3, 1e6), round(max(r[4] for r in rows) / 1000) if rows else "", valid, "+".join(reason), lms, hex(thr)]))
PY
  IFS=, read -r tok pt gt ms n cm cmin bm pw tmax valid reason lms thr < "$O/.row"
  local l1; l1=$(cut -d' ' -f1 /proc/loadavg)
  echo "7900xtx,<gpu-host>,$m,$q,$b,$r,$s,$tok,$rc,$tp,$tq,$cs,sclk_med=${cm}MHz,$(echo "$oth" | tr ', ' ';_'),$(date -u +%FT%TZ),$log,$pt,$gt,$ms,$n,$cm,$cmin,$bm,$pw,$tmax,$valid,$reason,$lms,$thr,$l1,$bw,$vr,$bp" >> "$CSV"
  echo "$tag $m $q $b r$r slot$s tok_s=$tok ms=$ms rc=$rc T=$tp->$tq cool=${cs}s clk=$cm/$cmin MHz n=$n busy=$bm load_ms=$lms thr=$thr vram=${vr}MB valid=$valid $reason"
}
nvalid() { awk -F, -v m=$1 -v q=$2 -v b=$3 'NR > 1 && $3 == m && $4 == q && $5 == b && $16 ~ /^logs\/prefill/ && $26 == 1 {n++} END {print n + 0}' "$CSV"; }
gen() { grep -v 'PyTorchObserver\|^[IWE] \|^\[sarc_dev\]' "$O/$1"; }
for m in "${MS[@]}"; do for q in "${QS[@]}"; do
  [[ -f $(pte $m $q) ]] || { echo "missing $(pte $m $q)"; continue; }
  cat "$(pte $m $q)" > /dev/null
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
    for k in real check; do
      [[ $k == real ]] && { pf=prompt_real_2048.txt; nt=2048; } || { pf=prompt_check.txt; nt=1972; }
      run1 $m $q parent 0 0 $pf $k $nt
      run1 $m $q cand 0 0 $pf $k $nt
    done
    same() { cmp -s <(gen "logs/$1-$m-$q-parent-r$2.log") <(gen "logs/$1-$m-$q-cand-r$2.log") && echo SAME || echo DIFFER; }
    x=$(same check 0); y=$(same prefill 1); z=$(same real 0)
    echo "$m,$q,$PROMPT:$y,prompt_real_2048.txt:$z,prompt_check.txt:$x" >> "$O/nexttoken.csv"; echo "nexttoken $m $q $PROMPT=$y real=$z check=$x"
  fi
done; done
echo "others_end: $($T/others.sh)" >> "$O/env.txt"; date -u > "$O/done.txt"; echo E2E5_DONE
