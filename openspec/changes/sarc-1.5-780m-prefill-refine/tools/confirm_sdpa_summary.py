#!/usr/bin/env python3
"""confirm_sdpa_summary.py <out csv> <enumeration results-steady.csv> <full-r2.csv> ...: the repeat stage of the
QK^T / attn*V search. Per family, model and configuration of the repeat files: median over the repeats of the op
time at S = 2048 (us; repeat 1 is the enumeration row), the spread of the repeats, the time relative to the fastest
configuration of that model, and "tied" when it is within 2 percent of it. Only rows that passed the correctness
cases on the configuration's own kernel count; attn*V rows of the 1B model are dropped when the tile N does not
divide head_dim 64 (that row is the table kernel). Prints the fastest five and the reference kernels per model."""
import collections, csv, re, statistics, sys
out, enum, files = sys.argv[1], sys.argv[2], sys.argv[3:]
REF = {"qk": ("t128x64k32g22s64", "t128x64k32g22s64nf"), "av": ("t64x64k32g22s64", "t64x64k32g42s32")}
want = {(r["family"], r["token"]) for f in files for r in csv.DictReader(open(f))}
t = collections.defaultdict(lambda: collections.defaultdict(list))
for f in [enum] + files:
    for r in csv.DictReader(open(f)):
        if (r["family"], r["token"]) not in want or r["ok"] != "PASS" or not r["us"]: continue
        n = int(re.search(r"t\d+x(\d+)k", r["token"]).group(1))
        if r["family"] == "av" and r["model"] == "llama-3.2-1b" and 64 % n: continue
        t[(r["family"], r["model"])][r["token"]].append(float(r["us"]))
w = csv.writer(open(out, "w")); w.writerow("family,model,rank,token,repeats,median_us,spread_pct,vs_fastest_pct,tied_within_2pct".split(","))
for (fam, model), d in sorted(t.items()):
    med = sorted((statistics.median(v), k) for k, v in d.items()); best = med[0][0]
    for i, (m, k) in enumerate(med, 1):
        v = d[k]; w.writerow([fam, model, i, k, len(v), f"{m:.1f}", f"{(max(v) - min(v)) / m * 100:.2f}", f"{(m / best - 1) * 100:+.2f}", "yes" if m <= best * 1.02 else "no"])
    tied = sum(m <= best * 1.02 for m, _ in med)
    print(f"{fam} {model}: " + "; ".join(f"{k} {m:.0f}" for m, k in med[:5]) + f" | tied within 2 %: {tied} | " +
          "; ".join(f"{k} {statistics.median(d[k]):.0f} ({(statistics.median(d[k]) / best - 1) * 100:+.1f} %)" for k in REF[fam] if k in d))
