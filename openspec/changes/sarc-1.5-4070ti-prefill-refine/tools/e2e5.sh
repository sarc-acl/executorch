#!/bin/bash
# e2e5.sh: parent vs candidate end-to-end prefill session on the RTX 4070 Ti SUPER (gpu-dev-4004).
# Protocol = openspec/changes/sarc-1.5-e2e-benchmark/kit/host/e2e.sh (fresh llama_main per run, --warmup,
# 1 new token, temperature 0, cool to idle+5 C max 120 s, builds interleaved parent->cand on odd repeats and
# cand->parent on even ones, failed runs kept), plus what this task adds:
#   - the GPU clock (nvidia-smi clocks.gr), busy %, power and temperature are sampled every 20 ms during each run
#     (logs/<run>.clk) and summarised over the measured execution window of that run;
#   - a GPU process this campaign did not start, seen before or after a run, ends the session (exit 76); a run
#     it overlapped is kept in runs.csv as invalid. The card not answering nvidia-smi ends it with exit 70;
#   - a run is VALID only if rc = 0, tok/s present, prompt_tokens = <expected>, generated_tokens = 0, no other GPU
#     process, and the median clock in the measured window >= CLKMIN MHz (0 = record only; set it from the
#     baseline session, the idle clock is 210 MHz and the maximum 3120 MHz). 1B prefill is about 100 ms, so its
#     window holds only a few samples; the card must also still answer nvidia-smi (Xid 79 rule: stop, no retry). Invalid runs stay in runs.csv with the
#     reason; cells with fewer than REPS valid runs per build get extra interleaved pairs (at most EXTRA).
# One GPU job at a time: everything runs under the gpu-lab lock.
#
# usage: e2e5.sh --stage DIR --out NAME --lock UUID [--reps 5] [--extra 3] [--models 1b,3b,8b] [--schemes 4w,8da4w]
#                [--prompt prompt_2048.txt] [--tokens 2048] [--clkmin 0] [--no-check]
#   DIR/{parent,cand}/{llama_main,libllama_runner.so,[env]}, DIR/prompt_*.txt; output in DIR/NAME/
set -uo pipefail
STAGE=""; OUTN=raw; LOCK=""; REPS=5; EXTRA=3; MODELS=1b,3b,8b; SCHEMES=4w,8da4w
PROMPT=prompt_2048.txt; TOKENS=2048; CLKMIN=0; CHECK=1; COOLMAX=120; MROOT=/mnt/linux-share/models
while [[ $# -gt 0 ]]; do
  case $1 in
    --stage) STAGE=$2; shift ;; --out) OUTN=$2; shift ;; --lock) LOCK=$2; shift ;;
    --reps) REPS=$2; shift ;; --extra) EXTRA=$2; shift ;; --models) MODELS=$2; shift ;;
    --schemes) SCHEMES=$2; shift ;; --prompt) PROMPT=$2; shift ;; --tokens) TOKENS=$2; shift ;;
    --clkmin) CLKMIN=$2; shift ;; --no-check) CHECK=0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done
[[ -n $STAGE && -n $LOCK ]] || { sed -n '2,17p' "$0"; exit 2; }
TOOLS=$(cd "$(dirname "$0")" && pwd)
D=$(cd "$STAGE" && pwd); cd "$D" || exit 2
O=$D/$OUTN; mkdir -p "$O/logs"
source "$TOOLS/common.sh"   # gtemp, others, no_others, gone_check, take_lock
take_lock 900; gone_check e2e5-start; no_others e2e5-start
need parent/llama_main parent/libllama_runner.so cand/llama_main cand/libllama_runner.so "$PROMPT" prompt_check.txt
sampler_start() {  # sampler_start <file>: epoch_us clock_MHz busy% power_W temp_C every 20 ms; stop with kill $SP
  stdbuf -oL nvidia-smi --query-gpu=clocks.gr,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits -lms 20 \
    > >(while IFS= read -r l; do echo "${EPOCHREALTIME/./} ${l//,/}"; done > "$1") 2>/dev/null 9>&- & SP=$!
}
{
  date -u; hostname; uname -r; echo "lock=$LOCK reps=$REPS extra=$EXTRA prompt=$PROMPT tokens=$TOKENS clkmin=$CLKMIN"
  for b in parent cand; do sha256sum $b/llama_main $b/libllama_runner.so; echo "$b env: $(cat $b/env 2>/dev/null | tr '\n' ' ')"; cat $b/COMMIT 2>/dev/null; done
  sha256sum prompt_*.txt; cat STAGE.md 2>/dev/null
  vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName|driverInfo|apiVersion'
  nvidia-smi --query-gpu=name,driver_version,pstate,clocks.gr,clocks.max.gr,clocks.mem,power.draw,power.limit,temperature.gpu,fan.speed --format=csv
  echo "others: $(others)"
} > "$O/env.txt" 2>&1
declare -A STEM=([1b]=llama-3.2-1b:llama3_2-1b [3b]=llama-3.2-3b:llama3_2-3b [8b]=llama-3.1-8b:llama3_1-8b)
IFS=, read -ra MS <<< "$MODELS"; IFS=, read -ra QS <<< "$SCHEMES"
pte()  { local MD ST; IFS=: read -r MD ST <<< "${STEM[$1]}"; echo "$MROOT/$MD/exported/${ST}_vulkan_$2.pte"; }
tokz() { local MD ST; IFS=: read -r MD ST <<< "${STEM[$1]}"; echo "$MROOT/$MD/original/tokenizer.model"; }
for m in "${MS[@]}"; do for q in "${QS[@]}"; do
  echo "model $m $q $(sha256sum "$(pte $m $q)" | cut -c1-16) $(pte $m $q)" >> "$O/env.txt"; done; done
sleep 60; IDLE=$(gtemp) || gpu_gone idle; echo "idle_temp=$IDLE" >> "$O/env.txt"
cool() { local t0=$SECONDS t; while :; do t=$(gtemp) || gpu_gone cool; [[ $t -le $((IDLE + 5)) || $((SECONDS - t0)) -ge $COOLMAX ]] && break; sleep 5; done; }
CSV=$O/runs.csv
[[ -f $CSV ]] || echo "gpu,host,model,scheme,build,rep,slot,tok_s,rc,temp_pre,temp_post,cool_s,clocks,others,utc,log,prompt_tokens,generated_tokens,prefill_ms,clk_n,clk_med_mhz,clk_min_mhz,busy_med,power_med_w,temp_max,valid,reason" > "$CSV"
run1() {  # run1 <model> <scheme> <build> <rep> <slot> <prompt> <tag> <expected tokens>
  local m=$1 q=$2 b=$3 r=$4 s=$5 p=$6 tag=$7 want=$8 log t0 tp tq rc oth cs sp
  log="logs/$tag-$m-$q-$b-r$r.log"
  t0=$SECONDS; cool; cs=$((SECONDS - t0)); tp=$(gtemp) || gpu_gone "before $log"; no_others "before $log"
  local benv=(); [[ -f $D/$b/env ]] && mapfile -t benv < "$D/$b/env"
  sampler_start "$O/${log%.log}.clk"; sleep 0.1
  env "${benv[@]}" LD_LIBRARY_PATH=$D/$b timeout 1800 "$D/$b/llama_main" --model_path "$(pte $m $q)" \
    --tokenizer_path "$(tokz $m)" --prompt_file "$p" --max_new_tokens 1 --temperature 0 \
    $([[ $tag == prefill ]] && echo --warmup) < /dev/null > "$O/$log" 2>&1 9>&-
  rc=$?; oth=$(others | tr ',' ';'); kill $SP 2>/dev/null; wait $SP 2>/dev/null; sleep 0.1; tq=$(gtemp) || gpu_gone "after $log rc=$rc"
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
            if len(f) == 5 and a * 1000 <= int(f[0]) <= b2 * 1000: rows.append([float(x) for x in f])
n = len(rows)
med = lambda k, d: (round(st.median(r[k] for r in rows) / d, 1) if rows else "")
cm = med(1, 1); cmin = round(min(r[1] for r in rows), 1) if rows else ""
reason = []
if rc != "0": reason.append("rc")
if tok == "": reason.append("no_tok_s")
if str(pt) != want: reason.append("prompt_tokens")
if tag == "prefill" and str(gt) != "0": reason.append("generated_tokens")
if oth: reason.append("other_gpu_process")
if n < 2: reason.append("clock_unsampled")
elif cm < float(clkmin): reason.append("clock_low")
valid = 0 if reason else 1
print(",".join(str(x) for x in [tok, pt, gt, ms, n, cm, cmin, med(2, 1), med(3, 1), round(max(r[4] for r in rows)) if rows else "", valid, "+".join(reason)]))
PY
  IFS=, read -r tok pt gt ms n cm cmin bm pw tmax valid reason < "$O/.row"
  echo "4070ti,$(hostname),$m,$q,$b,$r,$s,$tok,$rc,$tp,$tq,$cs,gr_med=${cm}MHz,$oth,$(date -u +%FT%TZ),$log,$pt,$gt,$ms,$n,$cm,$cmin,$bm,$pw,$tmax,$valid,$reason" >> "$CSV"
  echo "$tag $m $q $b r$r slot$s tok_s=$tok ms=$ms rc=$rc T=$tp->$tq cool=${cs}s clk=$cm/$cmin MHz n=$n busy=$bm valid=$valid $reason"
  [[ -n $oth ]] && no_others "during $log"
  return 0
}
nvalid() { awk -F, -v m=$1 -v q=$2 -v b=$3 'NR > 1 && $3 == m && $4 == q && $5 == b && $16 ~ /^logs\/prefill/ && $26 == 1 {n++} END {print n + 0}' "$CSV"; }
gen() { grep -v 'PyTorchObserver\|^[IWE] \|^\[sarc_dev\]' "$O/$1"; }
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
    run1 $m $q parent 0 0 prompt_check.txt check 1972
    run1 $m $q cand 0 0 prompt_check.txt check 1972
    if cmp -s <(gen "logs/check-$m-$q-parent-r0.log") <(gen "logs/check-$m-$q-cand-r0.log"); then x=SAME; else x=DIFFER; fi
    if cmp -s <(gen "logs/prefill-$m-$q-parent-r1.log") <(gen "logs/prefill-$m-$q-cand-r1.log"); then y=SAME; else y=DIFFER; fi
    echo "$m,$q,$PROMPT:$y,prompt_check.txt:$x" >> "$O/nexttoken.csv"; echo "nexttoken $m $q $PROMPT=$y check=$x"
  fi
done; done
echo "others_end: $(others)" >> "$O/env.txt"; date -u > "$O/done.txt"; echo "E2E5_DONE incomplete=${INCOMPLETE:-0}"
exit $(( ${INCOMPLETE:-0} ? 3 : 0 ))
