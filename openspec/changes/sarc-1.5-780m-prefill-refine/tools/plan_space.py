#!/usr/bin/env python3
"""plan_space.py <out dir> [--stage-b <4w token list>]: the sweep plan over enum_space.py's survivors, cut into
builds ("batches": one microbench binary each, at most BATCH variants, so spv.cpp stays compilable).

  8da4w, qk, av   every survivor.
  4w              125,712 survivors is 22 days at 15 s a configuration, so the family is cut in stages:
     A  every tile geometry (M, N, K, grid, subgroup size) once, with the surviving flag set closest to the
        shipped one (ACC_FP32 + CSH_IN_ASH; then ACC_FP32 alone, then with B_COLMAJOR or the texel-wise twin)
     B  every flag combination (and the texel-wise staging twin) on a few geometries: the shipped tile, the
        refine3 wide tile and their 2x2 / 2x4 grids here; the measured top geometries of stage A with --stage-b
Writes <out dir>/bNN/{4w,8da4w,qk,av}.txt (gen_space.py selections) and <out dir>/manifest.csv
(batch,family,stage,token,kernel_base,heads)."""
import csv, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import enum_space as es, gen_space_names as gn

BATCH = 560
out = pathlib.Path(sys.argv[1]); a = sys.argv[2:]
stage_b = [l.split()[0] for l in open(a[a.index("--stage-b") + 1]) if l.strip()] if "--stage-b" in a else None
B_GEOS = {(128, 128, 32, 4, 2, 32), (128, 256, 32, 4, 2, 32), (128, 128, 32, 2, 2, 32), (128, 128, 32, 2, 4, 32)}
SHIP = lambda f: {k for k, v in f.items() if v}
items = {}  # family -> [(stage, token, heads)]
if stage_b is None:
    for fam in ("8da4w", "qk", "av"):
        items[fam] = [("all", *(es.token(fam, c).split("/hd") + [""])[:2]) for c, r in es.space(fam) if r is None]
    geo = {}; b = []
    for c, r in es.space("4w"):
        if r is not None: continue
        if c["g"] in B_GEOS: b.append(("B", es.token("4w", c), ""))
        s = SHIP(c["f"])   # the survivor closest to the shipped flag set represents the geometry
        rank = ("ACC_FP32" not in s, c["bx"], len(s - {"ACC_FP32", "CSH_IN_ASH"}), "CSH_IN_ASH" not in s)
        if c["g"] not in geo or rank < geo[c["g"]][0]: geo[c["g"]] = (rank, es.token("4w", c))
    geo = {k: v[1] for k, v in geo.items()}
    seen = {t for _, t, _ in b}
    items["4w"] = [("A", t, "") for t in geo.values() if t not in seen] + b
else:
    geos = {gn.geometry(t) for t in stage_b}
    items["4w"] = [("B", es.token("4w", c), "") for c, r in es.space("4w") if r is None and c["g"] in geos]
start = int(a[a.index("--first-batch") + 1]) if "--first-batch" in a else 0
flat = [(f, *x) for f in ("qk", "8da4w", "4w") for x in items.get(f, [])]
av = items.get("av", []); rows = []; nb = start
for i in range(0, len(flat), BATCH):
    chunk = flat[i:i + BATCH]
    if av and chunk[0][0] == "qk":   # the av tokens ride along with the qk runs, spread over the qk batches
        nqk = -(-len(items["qk"]) // BATCH); k = (i // BATCH); per = -(-len(av) // nqk)
        chunk = chunk + [("av", *x) for x in av[k * per:(k + 1) * per]]
    d = out / f"b{nb:02d}"; d.mkdir(parents=True, exist_ok=True)
    for fam in ("4w", "8da4w", "qk", "av"):
        toks = [t for f, _, t, _ in chunk if f == fam]
        if toks: (d / f"{fam}.txt").write_text("\n".join(toks) + "\n")
    rows += [(f"b{nb:02d}", f, s, t, gn.kernel_base(f, t), h) for f, s, t, h in chunk]; nb += 1
mf = out / "manifest.csv"; new = not mf.exists()
with open(mf, "a") as f:
    w = csv.writer(f)
    if new: w.writerow("batch,family,stage,token,kernel_base,heads".split(","))
    w.writerows(rows)
import collections
print(collections.Counter((r[1], r[2]) for r in rows), "batches", start, "to", nb - 1)
