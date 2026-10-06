#!/usr/bin/env python3
"""refine_summary.py <artifact dir> <out dir>: what refinement round 1 of the 4w family found (screening mode).

Reads raw/rand/screen.csv, raw/refine1/screen.csv and space/refine1/{centres,n1}.txt. Writes
  best.csv     the 40 fastest configurations of sample + refinement by geomean over the three screened shapes, and
               per shape, with where they came from;
  local.csv    for every centre and every single-parameter change: the neighbour's time relative to the centre
               (geomean of the three shapes), and per parameter the fastest alternative;
and prints the per-parameter summary: over the centres, the median and the best ratio of the fastest alternative
level. A ratio below 1 means the change was faster than the centre in this (noisy, about +-5 %) mode."""
import collections, csv, math, os, statistics, sys
art, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
def load(p):
    t = collections.defaultdict(dict)
    for r in csv.DictReader(open(p)):
        if r["dispatched"] == "1" and r["us"]: t[r["token"]][r["model"][-2:]] = float(r["us"])
    return {k: d for k, d in t.items() if len(d) == 3}
ra, rf = load(f"{art}/raw/rand/screen.csv"), load(f"{art}/raw/refine1/screen.csv")
g = lambda d: math.exp(sum(math.log(v) for v in d.values()) / 3)
allc = dict(ra); allc.update(rf)
w = csv.writer(open(f"{out}/best.csv", "w")); w.writerow("rank_by,rank,token,from,us_8b,us_3b,us_1b,geomean_us,vs_fastest_pct".split(","))
for key, f in (("geomean", g), ("8b", lambda d: d["8b"]), ("3b", lambda d: d["3b"]), ("1b", lambda d: d["1b"])):
    s = sorted((f(d), k) for k, d in allc.items())
    for i, (v, k) in enumerate(s[:40]):
        d = allc[k]; w.writerow([key, i + 1, k, "refine1" if k in rf else "sample", d["8b"], d["3b"], d["1b"], f"{g(d):.1f}", f"{(v / s[0][0] - 1) * 100:+.2f}"])
centres = [l.strip() for l in open(f"{art}/space/refine1/centres.txt") if l.strip()]
w = csv.writer(open(f"{out}/local.csv", "w")); w.writerow("centre,parameter,neighbour,ratio_to_centre,fastest_alternative_of_parameter".split(","))
per = collections.defaultdict(list)
byc = collections.defaultdict(lambda: collections.defaultdict(list))
for l in open(f"{art}/space/refine1/n1.txt"):
    n, c, p = l.split()
    if n in allc and c in allc: byc[c][p].append((g(allc[n]) / g(allc[c]), n))
for c in centres:
    for p, v in sorted(byc[c].items()):
        v.sort(); per[p].append(v[0][0])
        for r, n in v: w.writerow([c, p, n, f"{r:.4f}", "yes" if n == v[0][1] else ""])
print(f"sample {len(ra)} + refinement {len(rf)} configurations; fastest geomean {min(map(g, allc.values())):.0f} us; centres with data {len(byc)}")
print("parameter,centres,median_ratio_of_fastest_alternative,best_ratio,centres_where_an_alternative_is_more_than_2pct_faster")
for p, v in sorted(per.items(), key=lambda x: statistics.median(x[1])):
    print(f"{p},{len(v)},{statistics.median(v):.3f},{min(v):.3f},{sum(1 for x in v if x < 0.98)}")
