#!/usr/bin/env python3
"""summarize.py <session raw dir> <verify.out> <control verify.out> <out.json>

Per cell: medians of the first 7 valid timed runs per arm, r = candidate / control, spread, invalid rows with
reasons; six-cell geomean; margins (every r >= 0.92, geomean >= 0.97, registered before the data); paired
bootstrap (i-th valid control run with the i-th valid candidate run, 20,000 resamples, seed 780, 95 % percentile
intervals); next tokens control vs candidate; pdiff, linear kernel names and the verify.sh lines (rates removed)
that differ from the control's."""
import collections, csv, json, math, random, re, statistics as st, sys

REPS, NB, SEED = 7, 20000, 780
raw, vout, cout, out = sys.argv[1:5]
rows = list(csv.DictReader(open(f"{raw}/runs.csv")))
cells = collections.OrderedDict()
for r in rows:
    if r["log"].startswith("logs/prefill"):
        cells.setdefault(f'{r["model"]}-{r["scheme"]}', {"parent": [], "cand": []})[r["build"]].append(r)
nt = {}
try:
    for line in open(f"{raw}/nexttoken.csv"):
        f = line.strip().split(",")
        nt[f"{f[0]}-{f[1]}"] = {x.rsplit(":", 1)[0]: x.rsplit(":", 1)[1] for x in f[2:]}
except FileNotFoundError:
    pass
res = {"cells": {}, "complete": True}
vals = {}
for c, a in cells.items():
    v = {b: [float(r["tok_s"]) for r in a[b] if r["valid"] == "1"][:REPS] for b in a}
    inv = [f'{r["build"]}:r{r["rep"]}:{r["reason"]}' for b in a for r in a[b] if r["valid"] != "1"]
    d = {"valid_control": len(v["parent"]), "valid_cand": len(v["cand"]), "invalid": inv,
         "next_token": nt.get(c, {})}
    if len(v["parent"]) >= REPS and len(v["cand"]) >= REPS:
        mp, mc = st.median(v["parent"]), st.median(v["cand"])
        sp = lambda l: round((max(l) - min(l)) / st.median(l) * 100, 2)
        d.update(control_med=mp, cand_med=mc, r=round(mc / mp, 5), spread_control_pct=sp(v["parent"]),
                 spread_cand_pct=sp(v["cand"]))
        vals[c] = v
    else:
        res["complete"] = False
    res["cells"][c] = d
rs = [d["r"] for d in res["cells"].values() if "r" in d]
if len(rs) == 6:
    g = math.exp(sum(math.log(x) for x in rs) / 6)
    res.update(geomean=round(g, 5), min_r=min(rs), margin_cells=min(rs) >= 0.92, margin_geomean=g >= 0.97)
    rnd = random.Random(SEED); gs = []; cr = {c: [] for c in vals}
    for _ in range(NB):
        lg = 0.0
        for c, v in vals.items():
            idx = [rnd.randrange(REPS) for _ in range(REPS)]
            x = st.median(v["cand"][i] for i in idx) / st.median(v["parent"][i] for i in idx)
            cr[c].append(x); lg += math.log(x)
        gs.append(math.exp(lg / 6))
    q = lambda l: [round(sorted(l)[int(0.025 * NB)], 5), round(sorted(l)[int(0.975 * NB) - 1], 5)]
    res["geomean_ci95"] = q(gs)
    for c in vals:
        res["cells"][c]["r_ci95"] = q(cr[c])
strip = lambda l: re.sub(r"(tok_s=)[0-9.]+", r"\1*", l.rstrip("\n"))
vl = [strip(l) for l in open(vout)] if vout != "-" else []
cl = [strip(l) for l in open(cout)] if cout != "-" else []
pd = [l for l in vl if l.startswith("pdiff")]
res["pdiff_pass"] = len(pd) == 12 and all(" rc=0 " in l and "ALL PASSED" in l for l in pd)
res["linear_kernels"] = {l.split()[1]: l.split("kernels:", 1)[1].strip() for l in vl if l.startswith("linear ")}
res["verify_lines_differing_from_control"] = [l for l in vl if l not in cl and not l.startswith("linear ")]
res["next_token_all_same"] = all(x == "SAME" for d in res["cells"].values() for x in d["next_token"].values()) \
    and len(nt) == len(cells)
res["monitoring_visibility"] = "own_user_plus_device_busy_other_users_unobservable (owner decision 2026-10-15 03:57 UTC)"
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: res[k] for k in res if k != "cells"}, indent=1))
for c, d in res["cells"].items():
    print(c, {k: d[k] for k in d if k != "invalid"}, len(d["invalid"]), "invalid")
