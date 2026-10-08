#!/bin/bash
# linear_screen.sh <session> <4w|8da4w> <rounds> <out.csv> <base env> <kernel base|table>...: kernel-level screen of
# linear kernels on the twelve real prefill shapes (test_llama_microbench --linear --regime=prefill --storage=texture3d,
# the model path; 3 warm-up + 5 timed runs, a steady clock on this suite), one gl.sh job per (round, kernel), order
# rotated per round. "table" = no override (the incumbent row); any other name is selected exactly with
# ET_VK_SARC_780M_{Q4,DQ} (a shape it does not fit keeps the table kernel and is recorded with dispatched=0).
# Resumable: a (round, kernel) already in <out.csv> is skipped.
source "$(dirname "$(readlink -f "$0")")/env.sh"
S=$A/stage/$1; F=$2; R=$3; OUT=$4; BASE=$5; shift 5; KS=("$@"); L=$S/linear-screen-$F; mkdir -p $L
declare -A V=([4w]=ET_VK_SARC_780M_Q4 [8da4w]=ET_VK_SARC_780M_DQ)
[[ -f "$OUT" ]] || echo "round,cand,model,op,M,N,K,kernel,dispatched,kernel_median_us,kernel_cov,utc" > "$OUT"
n=${#KS[@]}
for r in $(seq 1 $R); do for i in $(seq 0 $((n - 1))); do k=${KS[$(( (i + r - 1) % n ))]}
  grep -q "^$r,$k," $OUT && continue
  log=$L/r$r-$k.log; e=""; [[ $k != table ]] && e="${V[$F]}=$k"
  t0=$SECONDS; while (( $(gtemp) > 55 && SECONDS - t0 < 300 )); do sleep 5; done
  env $(cat "$BASE") $e $T/gl.sh $S/test_llama_microbench --linear --regime=prefill --scheme=$F --storage=texture3d --skip-correctness \
    --json-out=${log%.log}.json > $log 2>&1
  /usr/bin/python3 - "${log%.log}.json" "$r" "$k" >> $OUT <<'PY'
import json, sys, time
j, r, k = sys.argv[1:4]
try: recs = json.load(open(j)).get("cases", [])
except Exception: recs = []
for x in recs:
    if x.get("regime") != "prefill" or "coopmat" not in x.get("kernel", "") and "sarc" not in x.get("kernel", ""): continue
    kn = x.get("kernel", "")
    disp = 1 if k == "table" or kn.startswith(k + "_") else 0
    print(f'{r},{k},{x.get("model")},{x.get("op")},{x.get("M")},{x.get("N")},{x.get("K")},{kn},{disp},{x.get("kernel_median_us")},{x.get("kernel_cov")},{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}')
PY
done; done
