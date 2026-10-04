#!/usr/bin/env python3
"""screen_sdpa_summary.py <raw/screen dir>/screen.csv: per profile, the median over rounds of the prefill
QK^T, softmax, attn*V and total kernel times (ms) per model for the coopmat arm, and base_total / total."""
import collections, csv, statistics as st, sys
d = collections.defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    if r["rc"] != "0": continue
    v = "coopmat" if r["profile"] != "base" else "tiled"
    if r["variant"] == v: d[(r["profile"], r["model"], r["op"])].append(float(r["mean_us"]) / 1000)
models = sorted({k[1] for k in d}); profiles = list(dict.fromkeys(k[0] for k in d))
med = lambda p, m, o: st.median(d[(p, m, o)]) if d[(p, m, o)] else float("nan")
print("profile," + ",".join(f"{m}_qk,{m}_softmax,{m}_av,{m}_total,{m}_x" for m in models) + ",geomean_x,n")
for p in profiles:
    row = [p]; g = 1.0
    for m in models:
        t = med(p, m, "total"); x = med("base", m, "total") / t; g *= x
        row += [f"{med(p, m, o):.2f}" for o in ("qk", "softmax", "av", "total")] + [f"{x:.3f}"]
    print(",".join(row) + f",{g ** (1 / len(models)):.3f},{len(d[(p, models[0], 'total')])}")
