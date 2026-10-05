#!/usr/bin/env python3
"""sweep_merge.py <results.csv of card 0> <results.csv of card 1> <scale.csv>

Brings the second card's half of a split cheap screen into the first card's results.csv, so that every later
step (analysis, refinement, confirmation) reads one file. The second card's file stays as it was measured.

The two cards are not assumed to be equally fast: each ran the `base` arm (the device's table kernel, no
environment) at the start and after every 50 configurations, and a card-1 time is multiplied by
median(card-0 base) / median(card-1 base) of its (model, shape, K) before it is appended. The factors go to
<scale.csv>. Appended rows carry card = 1; a (configuration, mode, repeat) already in the first file is not
appended, nor are the second card's own base and ref arms. Running it twice appends nothing new."""
import collections, csv, statistics as st, sys
main, other, scalef = sys.argv[1:4]
m = list(csv.DictReader(open(main))); cols = open(main).readline().strip().split(",")
if "card" not in cols: sys.exit(f"{main} has no card column")
o = list(csv.DictReader(open(other))); key = lambda r: (r["model"], r["shape"], r["K"])
b = [collections.defaultdict(list), collections.defaultdict(list)]
for i, rows in enumerate((m, o)):
    for r in rows:
        if r["id"] == "base" and r["mode"] == "cheap" and r["us"] and r.get("card", "0") == str(i): b[i][key(r)].append(float(r["us"]))
scale = {k: st.median(b[0][k]) / st.median(b[1][k]) for k in b[1] if k in b[0]}
with open(scalef, "w") as f:
    f.write("model,shape,K,card0_base_runs,card0_base_us,card1_base_runs,card1_base_us,factor\n")
    for k in sorted(scale): f.write(f"{k[0]},{k[1]},{k[2]},{len(b[0][k])},{st.median(b[0][k]):.3f},{len(b[1][k])},{st.median(b[1][k]):.3f},{scale[k]:.5f}\n")
have = {(r["id"], r["mode"], r["rep"]) for r in m}; add = []
for r in o:
    if r["id"] == "base" or r["id"].startswith("ref-") or (r["id"], r["mode"], r["rep"]) in have: continue
    if r["us"]:
        if key(r) not in scale: sys.exit(f"no base time on both cards for {key(r)}")
        r["us"] = f'{float(r["us"]) * scale[key(r)]:.6g}'
    r["card"] = "1"; add.append(r)
with open(main, "a", newline="") as f: csv.DictWriter(f, cols, restval="").writerows(add)
print(f"merged {len({(r['id'], r['mode'], r['rep']) for r in add})} configurations of card 1; factors {min(scale.values(), default=1):.4f} to {max(scale.values(), default=1):.4f}")
