#!/bin/bash
# confirm_sdpa.sh <artifact dir>: the repeat stage of the QK^T / attn*V enumeration (raw/space/results-steady.csv).
#   1. the 10 fastest configurations per model and family among those that passed the 8 correctness cases on their
#      own kernel, plus the table kernels and the 780m-refine3 choices: space/confirm-sdpa/top.csv
#   2. four more timing repeats of those at the steady clock (20 warm-up + 8 timed runs): raw/confirm-sdpa/full-r2..r5.csv
#      (repeat 1 is the enumeration row); each repeat is also one pass of the extended correctness tier
#   3. seven more correctness passes per configuration (--sdpa-correctness-only --sdpa-tier=extended, the
#      configuration alone, the other op on its table kernel), 12 in all: raw/confirm-sdpa/passes.csv
# One GPU job at a time under the gpu-lab lock; resumable; honours PAUSE and the coordinator's HOLD.
set -uo pipefail
D=$(realpath "$1"); T=$(cd "$(dirname "$0")" && pwd); O=$D/raw/confirm-sdpa; mkdir -p $O/passes $D/space/confirm-sdpa; cd $D
[[ -f space/confirm-sdpa/top.csv ]] || python3 - <<'PY' 2> $O/select-top.txt
import collections, csv, sys
ALWAYS = {"qk": ["t128x64k32g22s64", "t128x64k32g22s64nf"], "av": ["t64x64k32g22s64", "t64x64k32g42s32"]}
man = {(r["family"], r["token"]): r for r in csv.DictReader(open("space/sweep/manifest.csv"))}
t = collections.defaultdict(dict)
for r in csv.DictReader(open("raw/space/results-steady.csv")):
    if r["ok"] == "PASS" and r["us"]: t[(r["family"], r["model"])][r["token"]] = float(r["us"])
keep = collections.OrderedDict()
for (fam, model), d in sorted(t.items()):
    rank = sorted(d, key=d.get)[:10]
    for tok in rank: keep.setdefault((fam, tok), []).append(model)
    print(f"{fam} {model}: {len(d)} passed and timed, fastest {d[rank[0]]:.0f} us {rank[0]}", file=sys.stderr)
for fam, toks in ALWAYS.items():
    for tok in toks:
        if (fam, tok) in man: keep.setdefault((fam, tok), []).append("always")
w = csv.DictWriter(open("space/confirm-sdpa/top.csv", "w"), fieldnames=list(next(iter(man.values())).keys())); w.writeheader()
for k in keep: w.writerow(man[k])
print(f"{len(keep)} configurations -> space/confirm-sdpa/top.csv", file=sys.stderr)
PY
export ET_VK_SDPA_PERF_RUNS=20,8
for i in 2 3 4 5; do
  python3 $T/sweep_space.py $D space/confirm-sdpa/top.csv $O/full-r$i.csv --only qk,av > $O/full-r$i.out 2>&1
done
echo "TIMING_DONE $(date -u +%FT%TZ)" > $O/timing.done
declare -A ENVV=([qk]=ET_VK_SARC_780M_QK [av]=ET_VK_SARC_780M_AV)
[[ -f $O/passes.csv ]] || echo "family,token,pass,rc,cases_passed,cases_failed,cases_on_kernel,pairing_not_ok" > $O/passes.csv
tail -n +2 space/confirm-sdpa/top.csv | while IFS=, read -r batch fam stage token kb heads; do
  for i in $(seq 1 7); do
    grep -q "^$fam,$token,$i," $O/passes.csv && continue
    while [[ -f $D/PAUSE ]]; do sleep 20; done
    L=$O/passes/$fam-$token-r$i.log
    env ${ENVV[$fam]}=$kb $T/gl.sh bin/microbench-$batch --sdpa-correctness-only --sdpa-tier=extended > $L 2>&1 < /dev/null; rc=$?
    echo "$fam,$token,$i,$rc,$(grep -c 'PASSED' $L),$(grep -c 'FAILED' $L),$(grep 'sdpa-kernels' $L | grep -c -- "=${kb}_"),$(grep 'sdpa-kernels' $L | grep -vc 'pairing=ok')" >> $O/passes.csv
  done
done
echo "CONFIRM_SDPA_DONE $(date -u +%FT%TZ)" > $O/done
