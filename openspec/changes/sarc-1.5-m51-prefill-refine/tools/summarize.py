#!/usr/bin/env python3
"""summarize.py <stage/<session>/raw>: per cell, the median of the first REPS VALID timed runs of each arm
(parent, cand), cand/parent, repeat spread, clock/temperature ranges, invalid runs, next-token results; then the
geomean over the cells that have REPS valid runs on both arms. Gains inside the noise band of RULES R6 are marked.
REPS = env REPS (default 5). Adapted from the 780M campaign's summarize.py for e2e_m51.sh's columns."""
import collections, csv, math, os, statistics as st, sys
d = sys.argv[1]; R = int(os.environ.get("REPS", "5"))
rows = list(csv.DictReader(open(os.path.join(d, "runs.csv"))))
cells = collections.OrderedDict()
for r in rows:
    if r["log"].startswith("logs/prefill"):
        cells.setdefault((r["model"], r["scheme"]), {"parent": [], "cand": []})[r["build"]].append(r)
nt = {}
p = os.path.join(d, "nexttoken.csv")
if os.path.exists(p):
    for l in open(p):
        f = l.strip().split(",")
        if len(f) >= 3: nt[(f[0], f[1])] = "/".join(x.split(":")[-1] for x in f[2:])
print("model,scheme,parent_med_tok_s,cand_med_tok_s,ratio,gain_pct,outside_2pct_band,parent_spread_pct,cand_spread_pct,"
      "valid_parent,valid_cand,invalid,clk_med_khz_range,temp_pre_range,load_ms_range,next_token_2048/check/unaligned")
ratios = []
for (m, q), a in cells.items():
    v = {b: [float(r["tok_s"]) for r in a[b] if r["valid"] == "1"][:R] for b in a}
    inv = [f'{r["build"]}:r{r["rep"]}:{r["reason"]}' for b in a for r in a[b] if r["valid"] != "1"]
    clk = [float(r["clk_med_khz"]) for b in a for r in a[b] if r["clk_med_khz"]]
    tp = [int(r["temp_pre"]) for b in a for r in a[b] if r["temp_pre"].isdigit()]
    lm = [float(r["load_ms"]) for b in a for r in a[b] if r["load_ms"]]
    if len(v["parent"]) < R or len(v["cand"]) < R:
        print(f'{m},{q},INCOMPLETE,{len(v["parent"])},{len(v["cand"])},{";".join(inv)}'); continue
    mp, mc = st.median(v["parent"]), st.median(v["cand"]); x = mc / mp; ratios.append(x)
    sp = lambda l: (max(l) - min(l)) / st.median(l) * 100
    rng = lambda l, f: f"{f(min(l))}-{f(max(l))}" if l else "-"
    print(f'{m},{q},{mp:.2f},{mc:.2f},{x:.4f},{(x - 1) * 100:+.2f},{"yes" if abs(x - 1) > 0.02 else "no"},'
          f'{sp(v["parent"]):.2f},{sp(v["cand"]):.2f},{len(v["parent"])},{len(v["cand"])},{";".join(inv) or "-"},'
          f'{rng(clk, int)},{rng(tp, int)},{rng(lm, int)},{nt.get((m, q), "-")}')
if ratios:
    g = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
    print(f"geomean over {len(ratios)} cells: {g:.4f} ({(g - 1) * 100:+.2f} %), min {min(ratios):.4f}, max {max(ratios):.4f}")
