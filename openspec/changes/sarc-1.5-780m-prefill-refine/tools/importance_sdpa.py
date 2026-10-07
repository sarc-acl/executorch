#!/usr/bin/env python3
"""importance_sdpa.py <results.csv> <qk|av> <out dir> [model,model]: parameter importance of an SDPA family (QK^T or
attn*V) from the enumeration of all its statically surviving configurations (sweep_space.py output, one row per
configuration and model, op time at S = 2048).

Response y = log(op time) averaged over the given models (default: all three). Only configurations that passed the
8 correctness cases on their own kernel and were timed on every one of those models enter; the others are counted.
Parameters: the tile geometry (WG_TILE_M/N/K, SG_GRID_X/Y, SUBGROUP_SIZE) and, for QK^T, PACKED_K (the "pk"
staging) and NO_MASK_FILL ("nf"); for attn*V, MULTI_LOAD ("ml"). Derived, not in the additive model: the
workgroup's thread count and the MMA tiles a subgroup owns.

  levels.csv      per parameter and level: n, geomean time of the level against the family geomean, and the
                  fastest configuration with that level against the fastest of the family
  importance.csv  per parameter: eta^2 (share of the variance of y between its levels, alone), unique R^2 (loss of
                  R^2 of the additive model without it), spread of the best-of-level times; "matters" = that
                  spread is at least 2 percent
  pairs.csv       per pair: R^2 gained by the pair's interaction cells over the additive model, and whether the
                  best level of each parameter changes with the level of the other
"""
import collections, csv, itertools, math, pathlib, re, sys
import numpy as np

src, fam, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
models = sys.argv[4].split(",") if len(sys.argv) > 4 else ["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b"]
GEO = ("WG_TILE_M", "WG_TILE_N", "WG_TILE_K", "SG_GRID_X", "SG_GRID_Y", "SUBGROUP_SIZE")
PARAMS = GEO + (("PACKED_K", "NO_MASK_FILL") if fam == "qk" else ("MULTI_LOAD",))
DERIVED = ("WG_THREADS", "MMAS_PER_SUBGROUP")
RE = re.compile(r"^(pk_|ml_)?t(\d+)x(\d+)k(\d+)g(\d)(\d)s(\d+)(nf)?$")

def params(token):
    m = RE.match(token); assert m, token
    px, M, N, K, X, Y, S, nf = m.groups(); M, N, K, X, Y, S = map(int, (M, N, K, X, Y, S))
    p = dict(WG_TILE_M=M, WG_TILE_N=N, WG_TILE_K=K, SG_GRID_X=X, SG_GRID_Y=Y, SUBGROUP_SIZE=S, WG_THREADS=X * Y * S,
             MMAS_PER_SUBGROUP=(M // Y // 16) * (N // X // 16))
    if fam == "qk": p.update(PACKED_K=int(bool(px)), NO_MASK_FILL=int(bool(nf)))
    else: p.update(MULTI_LOAD=int(bool(px)))
    return p

by = collections.defaultdict(dict)
for r in csv.DictReader(open(src)):
    if r["family"] == fam: by[r["token"]][r["model"]] = r
nshape = len(models); X = []; y = []; left = collections.Counter()
for t, rs in by.items():
    rows = [rs.get(m) for m in models]
    if fam == "av" and "llama-3.2-1b" in models and 64 % params(t)["WG_TILE_N"]:
        left["tile N does not divide head_dim 64 (the 1B row is the table kernel)"] += 1
    elif all(r and r["ok"] == "PASS" and r["us"] for r in rows):
        X.append(params(t)); y.append(sum(math.log(float(r["us"])) for r in rows) / nshape)
    else: left[",".join(sorted({(r["ok"] if r else "no row") for r in rows}))] += 1
y = np.array(y); n = len(y); mu = y.mean(); sst = ((y - mu) ** 2).sum(); best = y.min()

def r2(cells):
    cols = [np.ones(n)]
    for c in cells:
        keys = [tuple(x[k] for k in c) for x in X]
        for lv in sorted(set(keys))[1:]: cols.append(np.array([k == lv for k in keys], float))
    A = np.stack(cols, 1); beta = np.linalg.lstsq(A, y, rcond=None)[0]
    return 1 - ((y - A @ beta) ** 2).sum() / sst, np.linalg.matrix_rank(A)

def level_stats(k):
    g = collections.defaultdict(list)
    for x, v in zip(X, y): g[x[k]].append(v)
    return {lv: (len(v), np.mean(v), min(v)) for lv, v in g.items()}

full, _ = r2([(k,) for k in PARAMS])
with open(out / "levels.csv", "w") as f:
    w = csv.writer(f); w.writerow(["parameter", "level", "n", "geomean_vs_family_pct", "best_of_level_vs_best_pct"])
    for k in PARAMS + DERIVED:
        for lv, (c, m, b) in sorted(level_stats(k).items()):
            w.writerow([k, lv, c, f"{(math.exp(m - mu) - 1) * 100:+.1f}", f"{(math.exp(b - best) - 1) * 100:+.1f}"])
with open(out / "importance.csv", "w") as f:
    w = csv.writer(f); w.writerow(["parameter", "levels", "best_level_by_geomean", "eta2", "unique_r2", "best_of_level_spread_pct", "matters"])
    for k in PARAMS + DERIVED:
        s = level_stats(k); eta = sum(c * (m - mu) ** 2 for c, m, _ in s.values()) / sst
        uniq = full - r2([(p,) for p in PARAMS if p != k])[0] if k in PARAMS else float("nan")
        spread = (math.exp(max(b for _, _, b in s.values()) - best) - 1) * 100
        w.writerow([k, len(s), min(s, key=lambda lv: s[lv][1]), f"{eta:.3f}", f"{uniq:.3f}", f"{spread:.1f}", "yes" if spread >= 2 else "no"])
with open(out / "pairs.csv", "w") as f:
    w = csv.writer(f); w.writerow(["a", "b", "r2_gain_over_additive", "extra_terms", "best_a_changes_with_b", "best_b_changes_with_a"])
    rows = []
    for a, b in itertools.combinations(PARAMS, 2):
        r, rank = r2([(k,) for k in PARAMS] + [(a, b)]); base_rank = r2([(k,) for k in PARAMS])[1]
        def flips(p, q):
            g = collections.defaultdict(lambda: collections.defaultdict(lambda: math.inf))
            for x, v in zip(X, y): g[x[q]][x[p]] = min(g[x[q]][x[p]], v)
            return len({min(d, key=d.get) for d in g.values()}) > 1
        rows.append([a, b, r - full, rank - base_rank, "yes" if flips(a, b) else "no", "yes" if flips(b, a) else "no"])
    for r in sorted(rows, key=lambda r: -r[2]): w.writerow(r[:2] + [f"{r[2]:.4f}"] + r[3:])
print(f"{fam}: {len(by)} configurations in the file, not entered: {dict(left)}")
print(f"n={n} configurations, {nshape} models each; additive model R^2 = {full:.3f}; fastest/slowest = {math.exp(best - y.max()):.3f}")
