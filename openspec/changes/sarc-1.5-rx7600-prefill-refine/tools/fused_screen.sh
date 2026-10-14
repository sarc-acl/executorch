#!/bin/bash
# fused_screen.sh <session> <rounds> <out.csv> <base env> <pair>...: kernel-level screen of the fused attention variants
# on this device. A pair is "<d64 variant>,<d128 variant>" (ET_VK_SARC_780M_SDPA_FUSED; written "<d64>+<d128>" in <out.csv>); per round, every pair once
# (order rotated per round), test_llama_microbench --sdpa with ET_VK_SDPA_PERF_RUNS=20,8 (steady clock), one gl.sh
# job each; the median of the 8 timed totals (fused kernel + copy pass) per model at S = 2048 goes to <out.csv>.
# Resumable: a (round, pair) already in <out.csv> is skipped.
source "$(dirname "$(readlink -f "$0")")/env.sh"
S=$A/stage/$1; R=$2; OUT=$3; BASE=$4; shift 4; PAIRS=("$@"); L=$S/fused-screen; mkdir -p $L
[[ -f $OUT ]] || echo "round,pair,model,median_us,runs_us,kernels,utc" > $OUT
n=${#PAIRS[@]}
for r in $(seq 1 $R); do for i in $(seq 0 $((n - 1))); do p=${PAIRS[$(( (i + r - 1) % n ))]}
  grep -q "^$r,${p//,/+}," $OUT && continue
  log=$L/r$r-${p//,/+}.log
  t0=$SECONDS; while (( $(gtemp) > 55 && SECONDS - t0 < 300 )); do sleep 5; done
  env $BASE ET_VK_SARC_780M_SDPA_FUSED=$p ET_VK_SDPA_PERF_RUNS=20,8 $T/gl.sh $S/test_llama_microbench --sdpa --json-out=${log%.log}.json > $log 2>&1
  /usr/bin/python3 - "$log" "$r" "$p" >> $OUT <<'PY'
import re, statistics as st, sys, time, json
log, r, p = sys.argv[1:4]
txt = open(log).read()
try: recs = json.load(open(log[:-4] + ".json")).get("cases", [])
except Exception: recs = []
for m in re.finditer(r"\[sdpa-runs\] (\S+) prefill coopmat total_us ([0-9. e+-]+)", txt):
    v = [float(x) for x in m.group(2).split()]
    k = ";".join(sorted({x.get("kernel", "") for x in recs if x.get("model") == m.group(1) and x.get("regime") == "prefill"}))
    print(f"{r},{p.replace(',', '+')},{m.group(1)},{st.median(v):.1f},{' '.join(f'{x:.0f}' for x in v)},{k},{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
PY
done; done
