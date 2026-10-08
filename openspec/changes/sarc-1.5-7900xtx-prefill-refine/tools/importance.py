#!/usr/bin/env python3
"""importance.py <screen results.csv> <out dir> [--shipped <token>]: parameter importance of the 4w family from the
uniform random sample (sweep_space.py output, one row per configuration and shape).

Response y = log(kernel time), averaged over the screened shapes (so exp(y) is the geomean time). Only
configurations that dispatched their own kernel on every screened shape enter; the others are counted per parameter
level in reach.csv (the selector refused them, or the process failed).

  levels.csv        per parameter and level: n, geomean time relative to the sample geomean and to the best level;
                    the fastest and the 5th-percentile configuration with that level relative to the fastest
                    configuration of the sample; the level's share of the fastest 5 percent against its share of
                    the sample
  importance.csv    per parameter: best/worst level by geomean, range of the level geomeans (percent), eta^2 (share
                    of the variance of y between its levels, alone), unique R^2 (loss of R^2 of the additive model
                    when the parameter is dropped), and the spread of the best-of-level times (how much the worst
                    level costs when everything else is chosen well); "matters" = that spread is at least 2 percent.
                    The derived quantities (per-subgroup tile, threads per workgroup) are listed after the yaml
                    parameters; they are not part of the additive model.
  pairs.csv         per pair: R^2 gained by adding the pair's interaction cells to the additive model, with the
                    gain expected from noise for that many extra terms (df * (1 - R^2) / (n - p))
  independence.txt  the staged plan's assumption (tile geometry and options act independently): R^2 of the additive
                    model, with geometry x geometry, option x option and geometry x option interaction blocks, and
                    for every option whether its best level is the same in every geometry stratum
"""
import csv, collections, itertools, math, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gen_space_names as gn

src, out = sys.argv[1], pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
GEO = ("WG_TILE_M", "WG_TILE_N", "WG_TILE_K", "SG_GRID_X", "SG_GRID_Y", "SUBGROUP_SIZE")
OPT = ("ACC", "CSH", "FRAG_LAYOUT", "IMG_A", "IMG_W", "B_COLMAJOR", "SH_F16V4", "TEXEL_STAGING")
PARAMS = GEO + OPT
DERIVED = ("SG_TILE_M", "SG_TILE_N", "MMAS_PER_SUBGROUP", "WG_THREADS")

def params(token):
    c = gn.parse_4w(token); f = c["f"]; p = dict(zip(GEO, c["g"]))
    p["ACC"] = "fp32" if f["ACC_FP32"] else "group_fp32" if f["ACC_GROUP_FP32"] else "group_fp32_reg" if f["ACC_GROUP_FP32_REG"] else "fp16"
    p["CSH"] = ("full_pool" if f["CSH_POOL"] else "full") if f["CSH_FULL"] else "in_ash" if f["CSH_IN_ASH"] else "band" if f["CSH_BAND"] else "default"
    for k in ("FRAG_LAYOUT", "IMG_A", "IMG_W", "B_COLMAJOR", "SH_F16V4"): p[k] = int(f[k])
    p["TEXEL_STAGING"] = int(c["bx"])
    M, N, K, gx, gy, sg = c["g"]
    p.update(SG_TILE_M=M // gy, SG_TILE_N=N // gx, MMAS_PER_SUBGROUP=(M // gy // 16) * (N // gx // 16), WG_THREADS=gx * gy * sg)
    return p

by = collections.defaultdict(list)
for r in csv.DictReader(open(src)): by[r["token"]].append(r)
nshape = max(len(v) for v in by.values())
X = []; y = []; toks = []; reach = collections.Counter(); tot = collections.Counter()
for t, rs in by.items():
    p = params(t); ok = len(rs) == nshape and all(r["dispatched"] == "1" and r["us"] for r in rs)
    for k in PARAMS: tot[(k, p[k])] += 1; reach[(k, p[k])] += ok
    if ok: X.append(p); y.append(sum(math.log(float(r["us"])) for r in rs) / nshape); toks.append(t)
y = np.array(y); n = len(y); mu = y.mean(); sst = ((y - mu) ** 2).sum()
with open(out / "reach.csv", "w") as f:
    w = csv.writer(f); w.writerow(["parameter", "level", "sampled", "timed", "not_timed"])
    for (k, v), c in sorted(tot.items(), key=lambda kv: (PARAMS.index(kv[0][0]), str(kv[0][1]))): w.writerow([k, v, c, reach[(k, v)], c - reach[(k, v)]])

def onehot(cols):
    """Design matrix with an intercept and treatment-coded cells of the given parameter tuples."""
    M = [np.ones(n)]
    for cset in cols:
        cells = sorted({tuple(x[c] for c in cset) for x in X}, key=str)[1:]
        for cell in cells: M.append(np.array([float(tuple(x[c] for c in cset) == cell) for x in X]))
    return np.column_stack(M)
def r2(cols):
    A = onehot(cols); beta, *_ = np.linalg.lstsq(A, y, rcond=None); res = y - A @ beta
    return 1 - (res ** 2).sum() / sst, np.linalg.matrix_rank(A)

main = [(p,) for p in PARAMS]; r_add, p_add = r2(main)
lev = {}
with open(out / "levels.csv", "w") as f:
    w = csv.writer(f); w.writerow(["parameter", "level", "n", "geomean_vs_sample", "geomean_vs_best_level", "fastest_vs_sample_fastest",
                                   "p05_vs_sample_fastest", "share_of_sample", "share_of_fastest_5pct"])
    top = set(np.argsort(y)[:max(1, n // 20)].tolist()); bol = {}
    for k in PARAMS + DERIVED:
        g = collections.defaultdict(list); gt = collections.Counter()
        for i, (x, v) in enumerate(zip(X, y)): g[x[k]].append(v); gt[x[k]] += i in top
        m = {l: float(np.mean(v)) for l, v in g.items()}; best = min(m.values()); lev[k] = (m, {l: len(v) for l, v in g.items()})
        bol[k] = {l: min(v) - y.min() for l, v in g.items()}
        for l in sorted(m, key=lambda l: (isinstance(l, str), l)):
            w.writerow([k, l, len(g[l]), f"{math.exp(m[l] - mu):.4f}", f"{math.exp(m[l] - best):.4f}", f"{math.exp(bol[k][l]):.4f}",
                        f"{math.exp(np.percentile(g[l], 5) - y.min()):.4f}", f"{len(g[l]) / n:.3f}", f"{gt[l] / len(top):.3f}"])
imp = []
for k in PARAMS + DERIVED:
    m, cnt = lev[k]; best = min(m, key=m.get); worst = max(m, key=m.get)
    eta = sum(cnt[l] * (m[l] - mu) ** 2 for l in m) / sst
    uniq = r_add - r2([c for c in main if c != (k,)])[0] if k in PARAMS else float("nan")
    rng = (math.exp(m[worst] - m[best]) - 1) * 100
    spread = (math.exp(max(bol[k].values())) - 1) * 100
    imp.append((k, best, worst, rng, eta, uniq, min(bol[k], key=bol[k].get), max(bol[k], key=bol[k].get), spread, "yes" if spread >= 2 else "no"))
with open(out / "importance.csv", "w") as f:
    w = csv.writer(f); w.writerow(["parameter", "kind", "best_level_by_geomean", "worst_level_by_geomean", "range_of_level_geomeans_pct", "eta2_alone",
                                   "unique_r2_in_additive_model", "level_of_sample_fastest", "level_with_slowest_best", "best_of_level_spread_pct", "matters"])
    for r in sorted(imp, key=lambda r: (r[0] in DERIVED, -r[4])):
        w.writerow([r[0], "derived" if r[0] in DERIVED else "geometry" if r[0] in GEO else "option", r[1], r[2], f"{r[3]:.1f}", f"{r[4]:.4f}",
                    f"{r[5]:.4f}", r[6], r[7], f"{r[8]:.1f}", r[9]])
pairs = []
for a, b in itertools.combinations(PARAMS, 2):
    rr, pp = r2(main + [(a, b)]); df = pp - p_add
    pairs.append((a, b, rr - r_add, df, df * (1 - rr) / max(n - pp, 1)))
with open(out / "pairs.csv", "w") as f:
    w = csv.writer(f); w.writerow(["parameter_a", "parameter_b", "r2_gain_over_additive", "extra_terms", "gain_expected_from_noise", "kind"])
    for a, b, d, df, e in sorted(pairs, key=lambda r: -r[2]):
        kind = "geometry x geometry" if a in GEO and b in GEO else "option x option" if a in OPT and b in OPT else "geometry x option"
        w.writerow([a, b, f"{d:.4f}", df, f"{e:.4f}", kind])
def block(sel): return r2(main + [(a, b) for a, b in itertools.combinations(PARAMS, 2) if sel(a, b)])
gg = block(lambda a, b: a in GEO and b in GEO); oo = block(lambda a, b: a in OPT and b in OPT)
go = block(lambda a, b: (a in GEO) != (b in GEO)); al = block(lambda a, b: True)
with open(out / "independence.txt", "w") as f:
    f.write(f"configurations timed: {n} of {len(by)} sampled; shapes per configuration: {nshape}; sd of log time: {y.std():.3f}\n")
    f.write(f"R^2 additive (main effects only): {r_add:.4f} ({p_add} terms)\n")
    for name, (rr, pp) in (("+ geometry x geometry pairs", gg), ("+ option x option pairs", oo), ("+ geometry x option pairs", go), ("+ all pairs", al)):
        f.write(f"R^2 {name}: {rr:.4f} ({pp} terms), gain {rr - r_add:+.4f}\n")
    f.write("\nbest level of each option within each level of each geometry parameter (strata with at least 8 per level):\n")
    for o in OPT:
        overall = min(lev[o][0], key=lev[o][0].get); flips = []; strata = 0
        for gk in GEO:
            for gl in sorted({x[gk] for x in X}):
                g = collections.defaultdict(list)
                for x, v in zip(X, y):
                    if x[gk] == gl: g[x[o]].append(v)
                g = {l: np.mean(v) for l, v in g.items() if len(v) >= 8}
                if len(g) < 2: continue
                strata += 1; b = min(g, key=g.get)
                if b != overall and overall in g: flips.append(f"{gk}={gl}: {b} ({(math.exp(g[overall] - g[b]) - 1) * 100:.1f} % faster than {overall})")
        f.write(f"  {o}: best overall {overall}; different best level in {len(flips)} of {strata} strata" + ("".join("\n      " + x for x in flips) if flips else "") + "\n")
with open(out / "ranked.csv", "w") as f:
    w = csv.writer(f); w.writerow(["rank", "token", "geomean_us", "vs_fastest"] + list(PARAMS + DERIVED))
    for i, j in enumerate(np.argsort(y)): w.writerow([i + 1, toks[j], f"{math.exp(y[j]):.1f}", f"{math.exp(y[j] - y.min()):.4f}"] + [X[j][k] for k in PARAMS + DERIVED])
print(open(out / "independence.txt").read()); print(open(out / "importance.csv").read())
