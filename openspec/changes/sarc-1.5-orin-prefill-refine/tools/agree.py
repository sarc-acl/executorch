#!/usr/bin/env python3
"""agree.py <primary screen dir> <second-device screen dir>: agreement of the two Orins on one identical batch of
kernel-screen configurations (owner offer 2026-10-05). Per configuration (token): the geomean over the model
shapes of the median kernel time (lin_summary.py's reading of the microbench JSON). Prints the per-configuration
ratio second/primary, the Spearman rank correlation and the verdict. Threshold, fixed before the second device
was measured: rank correlation >= 0.95 and every ratio within 0.95 .. 1.05."""
import collections, glob, json, math, os, statistics as st, sys
def read(d):
    r = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in sorted(glob.glob(d + "/*.json")):
        tok = os.path.basename(f)[:-5].split("-", 1)[1].rsplit("-r", 1)[0]
        try: cs = json.load(open(f))["cases"]
        except Exception: continue
        for c in cs:
            if str(c.get("storage", "")).lower() == "texture3d": r[tok][(c["model"], c["op"])].append(c["kernel_median_us"])
    return {t: math.exp(sum(math.log(st.median(v)) for v in s.values()) / len(s)) for t, s in r.items() if s}
a, b = read(sys.argv[1]), read(sys.argv[2]); toks = sorted(set(a) & set(b))
def rank(x): o = sorted(range(len(x)), key=lambda i: x[i]); r = [0] * len(x); [r.__setitem__(i, k) for k, i in enumerate(o)]; return r
ra, rb = rank([a[t] for t in toks]), rank([b[t] for t in toks]); n = len(toks)
rho = 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1)) if n > 2 else float("nan")
ratios = {t: b[t] / a[t] for t in toks}
print("configuration,primary_geomean_us,second_geomean_us,ratio")
for t in toks: print(f"{t},{a[t]:.1f},{b[t]:.1f},{ratios[t]:.4f}")
ok = n >= 20 and rho >= 0.95 and all(0.95 <= x <= 1.05 for x in ratios.values())
print(f"configurations {n}, Spearman rank correlation {rho:.4f}, ratio second/primary median {st.median(ratios.values()):.4f} min {min(ratios.values()):.4f} max {max(ratios.values()):.4f}")
print("verdict:", "AGREE (threshold: rho >= 0.95, every ratio within 0.95 .. 1.05, at least 20 configurations)" if ok else "DISAGREE: the second device is not used")
sys.exit(0 if ok else 1)
