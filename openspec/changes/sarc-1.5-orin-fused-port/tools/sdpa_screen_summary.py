#!/usr/bin/env python3
"""sdpa_screen_summary.py <raw/<screen>>: per profile and model, the median over repeats of the QK^T, softmax,
attn*V and total time (us per layer at S = 2048, GPU timestamps) from rows.csv of sdpa_screen.sh. Under a fused
profile the three kernels dispatch nothing: the total is the copy pass + the fused kernel; every repeat is listed."""
import csv, collections, statistics as st, sys
d = collections.defaultdict(list)
for r in csv.DictReader(open(sys.argv[1] + "/rows.csv")):
    if r["regime"] == "prefill" and r["variant"] == "coopmat" and r["mean_us"] not in ("", "None"):
        d[(r["profile"], r["model"], r["op"])].append(float(r["mean_us"]))
print("profile,model,qk_us,softmax_us,av_us,total_us,reps,total_us_per_rep")
for p in sorted({k[0] for k in d}, key=lambda x: (x not in ("stock", "table"), x)):
    for m in ("llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b"):
        if d.get((p, m, "total")): print(p, m, *[f'{st.median(d[(p, m, s)]):.1f}' if d.get((p, m, s)) else "" for s in ("qk", "softmax", "av", "total")], len(d[(p, m, "total")]), "/".join(f"{x:.1f}" for x in d[(p, m, "total")]), sep=",")
