#!/usr/bin/env python3
"""confirm_summary.py <out csv> <full-r1.csv> <full-r2.csv> ...: the repeat stage of a family's search.

Per shape and configuration: median over the repeats of the kernel median (us), the spread of the repeats, the
time relative to the fastest configuration of that shape, and "tied" when it is within 2 percent of it. Rows that
did not dispatch their own kernel are left out. Prints the fastest five per shape."""
import collections, csv, statistics, sys
out, files = sys.argv[1], sys.argv[2:]
t = collections.defaultdict(lambda: collections.defaultdict(list)); dims = {}
for f in files:
    for r in csv.DictReader(open(f)):
        if r["dispatched"] == "1" and r["us"]:
            s = (r["family"], r["model"], r["op"]); t[s][r["token"]].append(float(r["us"])); dims[s] = (r["M"], r["N"], r["K"])
w = csv.writer(open(out, "w"))
w.writerow("family,model,op,M,N,K,rank,token,repeats,median_us,spread_pct,vs_fastest_pct,tied_within_2pct".split(","))
for s in sorted(t):
    med = sorted((statistics.median(v), k, v) for k, v in t[s].items())
    print(f"{s[0]} {s[1]} {s[2]} (N={dims[s][1]} K={dims[s][2]}): " + "; ".join(f"{k} {m:.0f}" for m, k, v in med[:5]))
    for i, (m, k, v) in enumerate(med):
        w.writerow(list(s) + list(dims[s]) + [i + 1, k, len(v), f"{m:.1f}", f"{(max(v) - min(v)) / m * 100:.2f}",
                                              f"{(m / med[0][0] - 1) * 100:+.2f}", "yes" if m <= med[0][0] * 1.02 else ""])
