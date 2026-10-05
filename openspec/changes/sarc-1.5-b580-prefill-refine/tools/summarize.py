#!/usr/bin/env python3
"""summarize.py <stage/<session>/raw>: per cell, the median of the first REPS (env.txt, 5 or 7) VALID timed runs of each arm
(parent, cand), cand/parent, repeat spread, clock/temperature ranges, invalid runs, next-token results; then the
geomean over the cells that have 5 valid runs on both arms. Gains inside the +-2 % noise band are marked."""
import csv, math, os, statistics as st, sys, collections
d = sys.argv[1]; rows = list(csv.DictReader(open(os.path.join(d, "runs.csv"))))
import re
REPS = int((re.search(r"reps=(\d+)", open(os.path.join(d, "env.txt")).read()) or [0, 5])[1])
timed = [r for r in rows if r["log"].startswith("logs/prefill")]
cells = collections.OrderedDict()
for r in timed: cells.setdefault((r["model"], r["scheme"]), {"parent": [], "cand": []})[r["build"]].append(r)
nt = {}
p = os.path.join(d, "nexttoken.csv")
if os.path.exists(p):
    for l in open(p):
        f = l.strip().split(",")
        if len(f) >= 4: nt[(f[0], f[1])] = "/".join(x.split(":")[-1] for x in f[2:])
print("model,scheme,parent_med_tok_s,cand_med_tok_s,ratio,gain_pct,outside_2pct_band,parent_spread_pct,cand_spread_pct,valid_parent,valid_cand,invalid,clk_med_mhz_range,temp_pre_range,next_token_2048/check/unaligned,fbusy_pct_med_max")
ratios = []
for (m, q), a in cells.items():
    v = {b: [float(r["tok_s"]) for r in a[b] if r["valid"] == "1"][:REPS] for b in a}
    inv = [f'{r["build"]}:r{r["rep"]}:{r["reason"]}' for b in a for r in a[b] if r["valid"] != "1"]
    clk = [float(r["clk_med_mhz"]) for b in a for r in a[b] if r["clk_med_mhz"]]
    tp = [int(r["temp_pre"]) for b in a for r in a[b]]
    fb = [float(r["fbusy_pct"]) for b in a for r in a[b] if r.get("fbusy_pct")] or [0.0]
    if len(v["parent"]) < REPS or len(v["cand"]) < REPS:
        print(f'{m},{q},INCOMPLETE,{len(v["parent"])},{len(v["cand"])},{";".join(inv)}'); continue
    mp, mc = st.median(v["parent"]), st.median(v["cand"]); x = mc / mp; ratios.append(x)
    sp = lambda l: (max(l) - min(l)) / st.median(l) * 100
    print(f'{m},{q},{mp:.2f},{mc:.2f},{x:.4f},{(x - 1) * 100:+.2f},{"yes" if abs(x - 1) > 0.02 else "no"},{sp(v["parent"]):.2f},{sp(v["cand"]):.2f},{len(v["parent"])},{len(v["cand"])},{";".join(inv) or "-"},{min(clk):.0f}-{max(clk):.0f},{min(tp)}-{max(tp)},{nt.get((m, q), "-")},{st.median(fb):.2f}/{max(fb):.2f}')
if ratios:
    g = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
    print(f"geomean over {len(ratios)} cells: {g:.4f} ({(g - 1) * 100:+.2f} %), min {min(ratios):.4f}, max {max(ratios):.4f}")
