#!/usr/bin/env python3
"""card_test.py <out dir> <seconds a> <seconds b> <seconds c>: verdict of card_test.sh (see there). Reads
raw/ct-{a,b,c0,c1}/results.csv beside the artifact directory of <out dir>; writes <out dir>/cardtest.csv,
report.txt and verdict."""
import csv, pathlib, statistics as st, sys
out = pathlib.Path(sys.argv[1]); ta, tb, tc = map(float, sys.argv[2:5]); RAW = out.parents[1] / "raw"
W = {"wq_wo": 2, "wk_wv": 2, "w1_w3": 2, "w2": 1}
def load(n):
    t = {}
    for r in csv.DictReader(open(RAW / f"ct-{n}" / "results.csv")):
        if r["mode"] == "cheap" and r["id"] != "base" and r["us"] and r["shape"] in W and (r["dispatched"] == "1" or r["id"].startswith("ref-")):
            t.setdefault(r["id"], {})[r["shape"]] = float(r["us"])
    for d in t.values():
        if all(k in d for k in W): d["score"] = sum(W[k] * d[k] for k in W)
    return t
def ranks(v):
    o = sorted(range(len(v)), key=v.__getitem__); r = [0.0] * len(v); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2
        i = j + 1
    return r
def spearman(x, y):
    a, b = ranks(x), ranks(y); ma, mb = st.mean(a), st.mean(b)
    return sum((p - ma) * (q - mb) for p, q in zip(a, b)) / (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** 0.5
T = {n: load(n) for n in ("a", "b", "c0", "c1")}; ok = True; lines = []
PAIRS = [("card-to-card, alone", "a", "b"), ("card 0, alone vs together", "a", "c0"), ("card 1, alone vs together", "b", "c1"), ("card-to-card, together", "c0", "c1")]
with open(out / "cardtest.csv", "w") as f:
    f.write("comparison,x,y,target,n,spearman,threshold,time_ratio_y_over_x_median,dispersion_p90_pct,dispersion_max_pct\n")
    for name, x, y in PAIRS:
        for k in list(W) + ["score"]:
            ids = [i for i in T[x] if i in T[y] and k in T[x][i] and k in T[y][i]]
            s = spearman([T[x][i][k] for i in ids], [T[y][i][k] for i in ids]); q = [T[y][i][k] / T[x][i][k] for i in ids]; m = st.median(q)
            d = sorted(abs(v / m - 1) * 100 for v in q); thr = 0.95 if k == "score" else 0.90
            gated = name != "card-to-card, together"
            if gated and (s < thr or len(ids) < 20): ok = False
            f.write(f"{name},{x},{y},{k},{len(ids)},{s:.4f},{thr if gated else ''},{m:.4f},{d[int(0.9 * (len(d) - 1))]:.2f},{d[-1]:.2f}\n")
            lines.append(f"{name}: {k} n={len(ids)} spearman={s:.4f} ratio={m:.4f} p90 dev={d[int(0.9 * (len(d) - 1))]:.2f} % max dev={d[-1]:.2f} %")
wall = tc / max(ta, tb)
if wall > 1.5: ok = False
lines.append(f"wall time: a (card 0 alone) {ta:.0f} s, b (card 1 alone) {tb:.0f} s, c (both at once) {tc:.0f} s; c / max(a, b) = {wall:.3f} (threshold 1.5), (a + b) / c = {(ta + tb) / tc:.3f}")
v = "CARD_SPLIT_OK" if ok else "CARD_SPLIT_REJECTED"
(out / "report.txt").write_text("\n".join(lines) + f"\n{v}\n"); (out / "verdict").write_text(v + "\n"); print("\n".join(lines)); print(v)
