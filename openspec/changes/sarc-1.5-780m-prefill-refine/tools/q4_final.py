#!/usr/bin/env python3
"""q4_final.py <results/780m/space dir> <out dir> [summary dir = out dir]: the final 4w result over ALL full measurements of the round
(twelve real shapes, 3 warm-up + 5 timed runs each): confirm-4w/full-r1..r5, refine2/full-r1..r2, refine3/full-r1,
geo-cbt/full-r1, confirm2-4w/full-r1..r4, confirm3-4w/full-r1..r3.

  summary.csv, summary.txt   confirm_summary.py over that pool (median per shape and configuration, repeats, spread,
                             distance from the fastest, tied within 2 %)
  coverage.csv               every configuration in the ten fastest of some shape: its shapes and ranks, the fewest
                             full measurements it has on a shape, and its production-diff passes per model
                             (confirm-4w, confirm2-4w, confirm3-4w pdiff.csv; a pass counts when rc = 0 and ALL PASSED)
  per-shape.csv              per shape: the fastest configuration, how many are tied within 2 %, and the kernels of
                             profile refine10 (final), refine9, 780m-refine3 and the release table against it
  per-layer.txt              2 wq_wo + 2 wk_wv + 2 w1_w3 + w2 per model for each of those and for the best per shape
Exit status 1 if a top-ten configuration has fewer than 5 full measurements or fewer than 12 passes on a model."""
import collections, csv, glob, os, statistics as st, subprocess, sys
root, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); sdir = sys.argv[3] if len(sys.argv) > 3 else out
POOL = [("confirm-4w", 5), ("refine2", 2), ("refine3", 1), ("geo-cbt", 1), ("confirm2-4w", 4), ("confirm3-4w", 3)]
files = [f"{root}/{d}/full-r{i}.csv" for d, n in POOL for i in range(1, n + 1)]
here = os.path.dirname(os.path.realpath(__file__))
txt = subprocess.run([sys.executable, f"{here}/confirm_summary.py", f"{sdir}/summary.csv"] + files, capture_output=True, text=True, check=True).stdout
open(f"{sdir}/summary.txt", "w").write(txt)
t = collections.defaultdict(lambda: collections.defaultdict(list)); dims = {}
for f in files:
    for r in csv.DictReader(open(f)):
        if r["family"] == "4w" and r["dispatched"] == "1" and r["us"]:
            s = (r["model"], r["op"]); t[s][r["token"]].append(float(r["us"])); dims[s] = (int(r["N"]), int(r["K"]))
MODELS = ("llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b"); pd = collections.Counter(); bad = collections.Counter()
for d in ("confirm-4w", "confirm2-4w", "confirm3-4w"):
    for r in csv.DictReader(open(f"{root}/{d}/pdiff.csv")):
        if r["rc"] == "0" and r["verdict"] == "ALL PASSED": pd[(r["token"], r["model"])] += 1
        else: bad[(r["token"], r["model"])] += 1
med = {s: {k: st.median(v) for k, v in d.items()} for s, d in t.items()}
top = collections.defaultdict(list)
for s in sorted(med):
    for i, k in enumerate(sorted(med[s], key=med[s].get)[:10]): top[k].append(f"{s[0][6:]}:{s[1]}#{i + 1}")
short = 0
with open(f"{out}/coverage.csv", "w") as f:
    w = csv.writer(f); w.writerow(["token", "top10_shapes", "min_full_measurements"] + [f"pdiff_passes_{m}" for m in MODELS] + ["pdiff_failed", "covered"])
    for k in sorted(top):
        reps = min(len(t[s][k]) for s in t if k in t[s]); p = [pd[(k, m)] for m in MODELS]; ok = reps >= 5 and min(p) >= 12
        short += not ok; w.writerow([k, " ".join(top[k]), reps] + p + [sum(bad[(k, m)] for m in MODELS), "yes" if ok else "NO"])
def refine10(N, K):
    if N >= 1024 and K >= 4096: return "t256x128k32g18s32f32cbt"
    if N >= 2048 and 3072 <= K < 4096: return "t256x128k32g24s32f32cbt"
    if N >= 8192 and K < 3072: return "t256x128k32g28s32f32cbt"
    if K < 3072: return "t128x128k32g24s32f32cbt"
    return "t128x256k32g42s32f32cbt" if N >= 1024 else "t128x128k32g42s32f32c"
def refine9(N, K):
    if N >= 2048 and K >= 4096: return "t256x128k32g18s32f32bbt"
    if N >= 8192 and K < 3072: return "t256x128k32g28s32f32bbt"
    return "t128x256k32g42s32f32cbt" if N >= 1024 else "t128x128k32g42s32f32c"
refine3 = lambda N, K: "t128x256k32g42s32f32c" if N >= 1024 else "t128x128k32g42s32f32c"
table = lambda N, K: "t128x128k32g42s32f32c"
PROF = (("refine10", refine10), ("refine9", refine9), ("780m-refine3", refine3), ("table", table))
W = {"wq_wo": 2, "wk_wv": 2, "w1_w3": 2, "w2": 1}; layer = collections.defaultdict(lambda: collections.Counter())
with open(f"{out}/per-shape.csv", "w") as f:
    w = csv.writer(f); w.writerow(["model", "op", "N", "K", "fastest", "fastest_us", "repeats", "tied_within_2pct"] +
                                  [x for n, _ in PROF for x in (f"{n}_kernel", f"{n}_us", f"{n}_vs_fastest_pct")] + ["refine10_vs_refine3_pct", "refine10_tied"])
    for s in sorted(med, key=lambda s: (MODELS.index(s[0]), s[1])):
        N, K = dims[s]; b = min(med[s], key=med[s].get); row = [s[0], s[1], N, K, b, f"{med[s][b]:.1f}", len(t[s][b]), sum(v <= med[s][b] * 1.02 for v in med[s].values())]
        layer[s[0]]["best per shape"] += W[s[1]] * med[s][b]
        for n, fn in PROF:
            k = fn(N, K); row += [k, f"{med[s][k]:.1f}", f"{(med[s][k] / med[s][b] - 1) * 100:+.2f}"]; layer[s[0]][n] += W[s[1]] * med[s][k]
        r10, r3 = med[s][refine10(N, K)], med[s][refine3(N, K)]
        w.writerow(row + [f"{(r10 / r3 - 1) * 100:+.2f}", "yes" if r10 <= med[s][b] * 1.02 else "no"])
with open(f"{out}/per-layer.txt", "w") as f:
    for m in MODELS:
        L = layer[m]; f.write(f"{m}: 780m-refine3 {L['780m-refine3']:.0f} us per layer; " + "; ".join(
            f"{n} {L[n]:.0f} ({(L[n] / L['780m-refine3'] - 1) * 100:+.2f} %)" for n in ("best per shape", "refine10", "refine9", "table")) + "\n")
print(open(f"{out}/per-layer.txt").read(), end="")
print(f"{len(files)} files, {len({k for d in t.values() for k in d})} configurations measured in full, {len(top)} in a top ten, {short} not covered")
sys.exit(1 if short else 0)
