#!/usr/bin/env python3
"""screen_sdpa_summary.py <raw/screen dir>/screen.csv [reference profile]: per profile, the median over rounds of
the prefill QK^T, softmax, attn*V and total kernel times (ms) per model for the coopmat arm, and
reference_total / total, also per round (min_x: the smallest of the per-round ratios, for the every-round rule).
The reference is `base` (no environment, stock kernels) unless a profile is named; for the fused kernel the
total is the fused kernel and its copy pass, and the three sub-op columns are 0."""
import collections, csv, statistics as st, sys
d = collections.defaultdict(list); REF = sys.argv[2] if len(sys.argv) > 2 else "base"
for r in csv.DictReader(open(sys.argv[1])):
    # the stock kernels (base) end with status 1: the suite reports the missing coopmat dispatch
    if r["rc"] != "0" and not (r["profile"] == "base" and r["rc"] == "1"): continue
    v = "coopmat" if r["profile"] != "base" else "tiled"
    if r["variant"] == v: d[(r["profile"], r["model"], r["op"])].append(float(r["mean_us"]) / 1000)
models = sorted({k[1] for k in d}); profiles = list(dict.fromkeys(k[0] for k in d))
med = lambda p, m, o: st.median(d[(p, m, o)]) if d[(p, m, o)] else float("nan")
print("profile," + ",".join(f"{m}_qk,{m}_softmax,{m}_av,{m}_total,{m}_x,{m}_min_x" for m in models) + ",geomean_x,n")
for p in profiles:
    row = [p]; g = 1.0
    for m in models:
        t = med(p, m, "total"); x = med(REF, m, "total") / t; g *= x
        per = [a / b for a, b in zip(d[(REF, m, "total")], d[(p, m, "total")])]   # rows are appended round by round
        row += [f"{med(p, m, o):.2f}" for o in ("qk", "softmax", "av", "total")] + [f"{x:.3f}", f"{min(per):.3f}" if per else "nan"]
    print(",".join(row) + f",{g ** (1 / len(models)):.3f},{len(d[(p, models[0], 'total')])}")
