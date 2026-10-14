#!/usr/bin/env python3
"""confirm_select.py within <pct> <out manifest> <manifest,manifest,...> <screen.csv>... [--always token,token]
   confirm_select.py top <n> <out manifest> <manifest,manifest,...> <full.csv>...  [--always token,token]

Chooses the configurations that go to the next, more exact stage of a family's search (sweep_space.py rows in,
plan_space.py manifest rows out; a token keeps the batch binary it was first built in).
  within  every configuration whose kernel time is within <pct> percent of the fastest on at least one measured
          shape. Used after a screening mode, whose single values scatter by a few percent: the cut is by distance
          from the best, not by rank, so that scatter cannot drop the true best.
  top     the <n> fastest per shape (median over the given files when a token is in several).
Only rows that dispatched their own kernel count. --always adds tokens whatever they measured (the shipped tile)."""
import collections, csv, statistics, sys
a = sys.argv[1:]; always = []
if "--always" in a: i = a.index("--always"); always = a[i + 1].split(","); a = a[:i] + a[i + 2:]
mode, arg, out, manifests = a[0], float(a[1]), a[2], a[3].split(",")
rows = {}
for m in manifests:
    for r in csv.DictReader(open(m)): rows.setdefault((r["family"], r["token"]), r)
t = collections.defaultdict(lambda: collections.defaultdict(list))
for f in a[4:]:
    for r in csv.DictReader(open(f)):
        if r["dispatched"] == "1" and r["us"]: t[(r["model"], r["op"])][(r["family"], r["token"])].append(float(r["us"]))
keep = collections.OrderedDict()
for shape in sorted(t):
    med = sorted((statistics.median(v), k) for k, v in t[shape].items())
    chosen = [k for v, k in med if v <= med[0][0] * (1 + arg / 100)] if mode == "within" else [k for v, k in med[:int(arg)]]
    for k in chosen: keep.setdefault(k, []).append(f"{shape[0]}:{shape[1]}")
    print(f"{shape[0]} {shape[1]}: {len(med)} measured, fastest {med[0][0]:.0f} us {med[0][1][1]}, chosen {len(chosen)}", file=sys.stderr)
for fam in {k[0] for k in keep}:
    for tok in always:
        if (fam, tok) in rows: keep.setdefault((fam, tok), []).append("always")
w = csv.DictWriter(open(out, "w"), fieldnames=list(next(iter(rows.values())).keys())); w.writeheader()
for k in keep: w.writerow(rows[k])
print(f"{len(keep)} configurations -> {out}", file=sys.stderr)
