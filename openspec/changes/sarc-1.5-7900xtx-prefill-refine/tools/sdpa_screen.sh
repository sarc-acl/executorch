#!/bin/bash
# sdpa_screen.sh <session> <rounds> <out.csv> <base env file> <profile|table>...: kernel-level screen of the unfused QK^T / attn*V kernels
# (named ET_VK_SARC_DEV_PROFILE profiles of the dev zone; "table" = no profile = the incumbent row). test_llama_microbench --sdpa
# (ET_VK_SDPA_PERF_RUNS=20,8), one gl.sh job per (round, profile), order rotated per round; per model at S = 2048 the mean time of each of the
# three ops (qk, softmax, av) and their total, from the json. Resumable: a (round, profile) already in <out.csv> is skipped.
source "$(dirname "$(readlink -f "$0")")/env.sh"
S=$A/stage/$1; R=$2; OUT=$3; BASE=$4; shift 4; PS=("$@"); L=$S/sdpa-screen; mkdir -p $L
[[ -f $OUT ]] || echo "round,profile,model,qk_us,softmax_us,av_us,total_us,utc" > $OUT
n=${#PS[@]}
for r in $(seq 1 $R); do for i in $(seq 0 $((n - 1))); do p=${PS[$(( (i + r - 1) % n ))]}
  grep -q "^$r,$p," $OUT && continue
  log=$L/r$r-$p.log; e=""; [[ $p != table ]] && e="ET_VK_SARC_DEV_PROFILE=$p"
  t0=$SECONDS; while (( $(gtemp_core) > 55 && SECONDS - t0 < 300 )); do sleep 5; done
  env $(cat "$BASE") $e ET_VK_SDPA_PERF_RUNS=20,8 $T/gl.sh $S/test_llama_microbench --sdpa --json-out=${log%.log}.json > $log 2>&1
  /usr/bin/python3 - "${log%.log}.json" "$r" "$p" >> $OUT <<'PY'
import json, sys, time
j, r, p = sys.argv[1:4]
try: recs = json.load(open(j)).get("cases", [])
except Exception: recs = []
d = {}
for x in recs:
    if x.get("regime") == "prefill" and x.get("variant") == "coopmat": d.setdefault(x["model"], {})[x["op"]] = x.get("op_mean_us")
for m, v in sorted(d.items()):
    print(f'{r},{p},{m},{v.get("qk")},{v.get("softmax")},{v.get("av")},{v.get("total")},{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}')
PY
done; done
