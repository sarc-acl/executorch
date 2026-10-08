#!/bin/bash
# e2e5.sh: parent vs candidate end-to-end prefill session on the Jetson Orin Nano (duck-naughty). Device side.
# Protocol = openspec/changes/sarc-1.5-e2e-benchmark/kit/host/e2e.sh (fresh llama_main per run, --warmup,
# 1 new token, temperature 0, cool to idle+5 C max 120 s, builds interleaved parent->cand on odd repeats and
# cand->parent on even ones, failed runs kept), plus what this task adds:
#   - the GPU clock (devfreq cur_freq), load, module power (VDD_IN) and gpu-thermal are sampled every 0.1 s during
#     each run (logs/<run>.clk) and summarised over the measured execution window of that run;
#   - the cooling wait also ends when the temperature has stopped falling (common.sh cool_to);
#   - available memory and the swap counters are recorded before and after each run (logs/<run>.mem; 8 GB are
#     shared between CPU and GPU and the 8B model is 4.4 GB); swap-out during a run is reported, not judged;
#   - a GPU process this campaign did not start ends the session (exit 76): it is looked for before each run,
#     every 0.5 s while the run executes (logs/<run>.others) and once after it; the abort uses what was captured,
#     without asking again, and a run it overlapped is kept in runs.csv as invalid. GPU sensors not answering
#     end it with exit 70;
#   - a run is VALID only if rc = 0, tok/s present, prompt_tokens = <expected>, generated_tokens = 0, no other GPU
#     process, and the median clock in the measured window >= CLKMIN MHz. The threshold comes from --clkmin-file
#     (calibrate_clock.py on the baseline and A/A sessions); --calibrate runs record-only (clkmin 0 in runs.csv)
#     and is only for those sessions: gate_check.py rejects such a session as a candidate gate. Invalid runs stay
#     in runs.csv with the reason; cells with fewer than REPS valid runs per build get extra interleaved pairs
#     (at most EXTRA).
# One GPU job at a time: everything runs under the gpu-lab lock.
#
# After the timed runs of a cell, parent and candidate each run prompt_real_2048.txt (2048 tokens, aligned real
# text), prompt_check.txt (1972) and r1304.txt (1792) once, and nexttoken.py compares the next token per prompt
# (and the timed prompt's r1 logs); nexttoken.csv keeps rc, prompt tokens, the token and the output hashes.
#
# usage: e2e5.sh --stage DIR --out NAME --lock UUID (--clkmin-file JSON | --calibrate) [--reps 5] [--extra 3]
#                [--models 1b,3b,8b] [--schemes 4w,8da4w] [--prompt prompt_2048.txt] [--tokens 2048] [--no-check]
#   DIR/{parent,cand}/{llama_main,libllama_runner.so,[env]}, DIR/prompt_*.txt; output in DIR/NAME/
set -uo pipefail
STAGE=""; OUTN=raw; LOCK=""; REPS=5; EXTRA=3; MODELS=1b,3b,8b; SCHEMES=4w,8da4w
PROMPT=prompt_2048.txt; TOKENS=2048; CLKFILE=""; CALIBRATE=0; CHECK=1; COOLMAX=120
while [[ $# -gt 0 ]]; do
  case $1 in
    --stage) STAGE=$2; shift ;; --out) OUTN=$2; shift ;; --lock) LOCK=$2; shift ;;
    --reps) REPS=$2; shift ;; --extra) EXTRA=$2; shift ;; --models) MODELS=$2; shift ;;
    --schemes) SCHEMES=$2; shift ;; --prompt) PROMPT=$2; shift ;; --tokens) TOKENS=$2; shift ;;
    --clkmin-file) CLKFILE=$(realpath "$2"); shift ;; --calibrate) CALIBRATE=1 ;; --no-check) CHECK=0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done
[[ -n $STAGE && -n $LOCK ]] || { sed -n '2,32p' "$0"; exit 2; }
[[ ( -n $CLKFILE && $CALIBRATE == 0 ) || ( -z $CLKFILE && $CALIBRATE == 1 ) ]] || { echo "exactly one of --clkmin-file and --calibrate is required" >&2; exit 2; }
TOOLS=$(cd "$(dirname "$0")" && pwd)
D=$(cd "$STAGE" && pwd); cd "$D" || exit 2
O=$D/$OUTN; mkdir -p "$O/logs"
source "$TOOLS/common.sh"   # gtemp, others, no_others, gone_check, take_lock, sampler_start, cool_to
[[ $SIDE == device ]] || { echo "device only" >&2; exit 2; }
take_lock 900; gone_check e2e5-start; no_others e2e5-start
need parent/llama_main parent/libllama_runner.so cand/llama_main cand/libllama_runner.so "$PROMPT" prompt_check.txt prompt_real_2048.txt r1304.txt
# Per-cell clock threshold (MHz): from the calibration file, or 0 = record only (--calibrate).
declare -A CLK
if [[ -n $CLKFILE ]]; then need "$CLKFILE"
  while read -r cell v; do CLK[$cell]=$v; done < <(python3 -c 'import json, sys
for k, v in json.load(open(sys.argv[1]))["cells"].items(): print(k, v["clkmin_mhz"])' "$CLKFILE")
fi
clkmin() { if [[ $CALIBRATE == 1 ]]; then echo 0; else local v=${CLK[$1-$2]:-}; [[ $v =~ ^[1-9][0-9]*$ ]] || { echo "no clock threshold for cell $1 $2 in $CLKFILE" >&2; exit 77; }; echo $v; fi; }
{
  date -u; hostname; uname -r; echo "lock=$LOCK reps=$REPS extra=$EXTRA prompt=$PROMPT tokens=$TOKENS calibrate=$CALIBRATE clkmin_file=$CLKFILE $([[ -n $CLKFILE ]] && sha256sum < "$CLKFILE" | cut -c1-16)"
  for b in parent cand; do sha256sum $b/llama_main $b/libllama_runner.so; echo "$b env: $(cat $b/env 2>/dev/null | tr '\n' ' ')"; cat $b/COMMIT 2>/dev/null; done
  sha256sum prompt_*.txt r1304.txt; cat STAGE.md 2>/dev/null
  vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName|driverInfo|apiVersion'
  head -1 /etc/nv_tegra_release; nvpmodel -q 2>/dev/null | tr '\n' ' '; echo
  echo "devfreq: governor=$(cat $GPUDEV/governor) cur=$(cat $GPUDEV/cur_freq) min=$(cat $GPUDEV/min_freq) max=$(cat $GPUDEV/max_freq) available=[$(cat $GPUDEV/available_frequencies)]"
  echo "fan: pwm=$(cat /sys/class/hwmon/hwmon*/pwm1 2>/dev/null | head -1) rpm=$(cat /sys/class/hwmon/hwmon*/rpm 2>/dev/null | head -1)"
  echo "temp: gpu=$(gtemp_m) mC; load=$(cat $GPULOAD); mem: $(mem_line)"
  echo "others: $(others)"
} > "$O/env.txt" 2>&1
declare -A STEM=([1b]=llama-3.2-1b:llama3_2-1b [3b]=llama-3.2-3b:llama3_2-3b [8b]=llama-3.1-8b:llama3_1-8b)
IFS=, read -ra MS <<< "$MODELS"; IFS=, read -ra QS <<< "$SCHEMES"
pte()  { local MD ST; IFS=: read -r MD ST <<< "${STEM[$1]}"; echo "$MODELDIR/${ST}_vulkan_$2.pte"; }
tokz() { echo "$MODELDIR/tokenizer.model"; }
for m in "${MS[@]}"; do for q in "${QS[@]}"; do
  echo "model $m $q $(sha256sum "$(pte $m $q)" | cut -c1-16) $(pte $m $q)" >> "$O/env.txt"; done; done
sleep 60; IDLE=$(gtemp) || gpu_gone idle; echo "idle_temp=$IDLE" >> "$O/env.txt"
cool() { cool_to $(( (IDLE + 5) * 1000 )) $COOLMAX; }
CSV=$O/runs.csv
[[ -f $CSV ]] || echo "gpu,host,model,scheme,build,rep,slot,tok_s,rc,temp_pre,temp_post,cool_s,clocks,others,utc,log,prompt_tokens,generated_tokens,prefill_ms,clk_n,clk_med_mhz,clk_min_mhz,busy_med,power_med_w,temp_max,valid,reason,clkmin" > "$CSV"
run1() {  # run1 <model> <scheme> <build> <rep> <slot> <prompt> <tag> <expected tokens>
  local m=$1 q=$2 b=$3 r=$4 s=$5 p=$6 tag=$7 want=$8 log t0 tp tq rc oth cs sp cmin_cell=0
  [[ $tag == prefill ]] && { cmin_cell=$(clkmin $m $q) || exit 77; }
  log="logs/$tag-$m-$q-$b-r$r.log"
  t0=$SECONDS; cool; cs=$((SECONDS - t0)); tp=$(gtemp) || gpu_gone "before $log"; no_others "before $log"
  local benv=(); [[ -f $D/$b/env ]] && mapfile -t benv < "$D/$b/env"
  echo "pre $(mem_line)load=$(cat $GPULOAD)" > "$O/${log%.log}.mem"
  others_watch_start "$O/${log%.log}.others"; sampler_start "$O/${log%.log}.clk"; sleep 0.1
  env "${benv[@]}" LD_LIBRARY_PATH=$D/$b timeout 1800 "$D/$b/llama_main" --model_path "$(pte $m $q)" \
    --tokenizer_path "$(tokz $m)" --prompt_file "$p" --max_new_tokens 1 --temperature 0 \
    $([[ $tag == prefill ]] && echo --warmup) < /dev/null > "$O/$log" 2>&1 9>&-
  rc=$?; oth=$(others_watch_stop "$O/${log%.log}.others" | tr ',' ';'); sampler_stop; echo "post $(mem_line)" >> "$O/${log%.log}.mem"; tq=$(gtemp) || gpu_gone "after $log rc=$rc"
  python3 $TOOLS/runrow.py "$O/$log" "$O/${log%.log}.clk" "$want" "$cmin_cell" "$rc" "$oth" "$tag" > "$O/.row"
  IFS=, read -r tok pt gt ms n cm cmin bm pw tmax valid reason < "$O/.row"
  echo "orin,$(hostname),$m,$q,$b,$r,$s,$tok,$rc,$tp,$tq,$cs,devfreq_med=${cm}MHz,$oth,$(date -u +%FT%TZ),$log,$pt,$gt,$ms,$n,$cm,$cmin,$bm,$pw,$tmax,$valid,$reason,$cmin_cell" >> "$CSV"
  echo "$tag $m $q $b r$r slot$s tok_s=$tok ms=$ms rc=$rc T=$tp->$tq cool=${cs}s clk=$cm/$cmin MHz n=$n busy=$bm valid=$valid $reason"
  [[ -n $oth ]] && abort_others "during $log (the run is kept in runs.csv as invalid)" "$oth"
  LAST_RC=$rc; return 0
}
nvalid() { awk -F, -v m=$1 -v q=$2 -v b=$3 'NR > 1 && $3 == m && $4 == q && $5 == b && $16 ~ /^logs\/prefill/ && $26 == 1 {n++} END {print n + 0}' "$CSV"; }
NT_HEADER=$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import nexttoken; print(",".join(nexttoken.FIELDS))' $TOOLS)
for m in "${MS[@]}"; do for q in "${QS[@]}"; do
  [[ -f $(pte $m $q) ]] || { echo "missing $(pte $m $q)"; INCOMPLETE=1; continue; }
  r=1
  while :; do
    if (( r % 2 )); then order=(parent cand); else order=(cand parent); fi
    run1 $m $q ${order[0]} $r 1 $PROMPT prefill $TOKENS
    run1 $m $q ${order[1]} $r 2 $PROMPT prefill $TOKENS
    r=$((r + 1))
    (( r <= REPS )) && continue
    (( $(nvalid $m $q parent) >= REPS && $(nvalid $m $q cand) >= REPS )) && break
    (( r > REPS + EXTRA )) && { echo "CELL $m $q: fewer than $REPS valid runs after $EXTRA extra pairs"; INCOMPLETE=1; break; }
  done
  if [[ $CHECK == 1 ]]; then
    [[ -f $O/nexttoken.csv ]] || echo "$NT_HEADER" > "$O/nexttoken.csv"
    for pc in real:prompt_real_2048.txt:2048 check:prompt_check.txt:1972 r1304:r1304.txt:1792; do
      IFS=: read -r tag pf want <<< "$pc"
      run1 $m $q parent 0 0 $pf $tag $want; prc=$LAST_RC
      run1 $m $q cand 0 0 $pf $tag $want; crc=$LAST_RC
      row=$(python3 $TOOLS/nexttoken.py $D/$pf $want "$O/logs/$tag-$m-$q-parent-r0.log" "$O/logs/$tag-$m-$q-cand-r0.log" $prc $crc)
      echo "$m,$q,$row" >> "$O/nexttoken.csv"; echo "nexttoken $m $q $pf ${row##*,}"
    done
    # The timed prompt, from the first timed pair (rc is re-read from runs.csv by gate_check.py).
    rcof() { awk -F, -v l="logs/prefill-$m-$q-$1-r1.log" '$16 == l {print $9}' "$CSV" | head -1; }
    row=$(python3 $TOOLS/nexttoken.py $D/$PROMPT $TOKENS "$O/logs/prefill-$m-$q-parent-r1.log" "$O/logs/prefill-$m-$q-cand-r1.log" "$(rcof parent)" "$(rcof cand)" --timed)
    echo "$m,$q,$row" >> "$O/nexttoken.csv"; echo "nexttoken $m $q $PROMPT ${row##*,}"
  fi
done; done
echo "others_end: $(others)" >> "$O/env.txt"; date -u > "$O/done.txt"; echo "E2E5_DONE incomplete=${INCOMPLETE:-0}"
exit $(( ${INCOMPLETE:-0} ? 3 : 0 ))
