#!/bin/bash
# sdpa_screen2.sh <session> <rounds> <out.csv> <base env file> <QK incumbent> <AV incumbent> <item>...: the second attention screen, by exact
# kernel name (ET_VK_SARC_7900XTX_QK / _AV). An item is "table" (both incumbents), "qk-<kernel_base>" (that QK^T kernel with the AV incumbent) or
# "av-<kernel_base>" (that attn*V kernel with the QK^T incumbent). test_llama_microbench --sdpa (ET_VK_SDPA_PERF_RUNS=20,8), one gl.sh job per
# (round, item), order rotated per round; per model at S = 2048 the mean time of the qk, softmax, av ops and their total. The rows keep the
# item as "profile" (sdpa_pick.py reads them unchanged). Resumable.
source "$(dirname "$(readlink -f "$0")")/env.sh"
S=$A/stage/$1; R=$2; OUT=$3; BASE=$4; QI=$5; AI=$6; shift 6; PS=("$@"); L=$S/sdpa-screen2; mkdir -p $L
[[ -f $OUT ]] || echo "round,profile,model,qk_us,softmax_us,av_us,total_us,utc" > $OUT
n=${#PS[@]}
for r in $(seq 1 $R); do for i in $(seq 0 $((n - 1))); do p=${PS[$(( (i + r - 1) % n ))]}
  grep -q "^$r,$p," $OUT && continue
  log=$L/r$r-$p.log; qk=$QI; av=$AI
  case $p in qk-*) qk=${p#qk-} ;; av-*) av=${p#av-} ;; esac
  t0=$SECONDS; while (( $(gtemp) > 55 && SECONDS - t0 < 300 )); do sleep 5; done
  env $(cat "$BASE") ET_VK_SARC_7900XTX_QK=$qk ET_VK_SARC_7900XTX_AV=$av ET_VK_SDPA_PERF_RUNS=20,8 $T/gl.sh $S/test_llama_microbench --sdpa --json-out=${log%.log}.json > $log 2>&1
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
