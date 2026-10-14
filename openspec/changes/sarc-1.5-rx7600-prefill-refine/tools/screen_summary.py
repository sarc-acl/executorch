#!/usr/bin/env python3
"""screen_summary.py <raw/screen dir> <scheme>: per token, per model, the per-layer-weighted linear time
(2*wq_wo + 2*wk_wv + 2*w1_w3 + w2, median of repeats per shape) and its ratio base/token (>1 = faster)."""
import collections, glob, json, os, re, statistics as st, sys
d, q = sys.argv[1], sys.argv[2]
W = {"wq_wo": 2, "wk_wv": 2, "w1_w3": 2, "w2": 1}
data = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(os.path.join(d, f"{q}-*-r*.json"))):
    tok = re.match(rf"{q}-(.*)-r\d+\.json", os.path.basename(f)).group(1)
    try: cs = json.load(open(f))["cases"]
    except Exception as e: print("bad", f, e); continue
    for c in cs:
        if c.get("suite", "linear") == "linear" and c["storage"].lower().startswith("texture"):
            if c.get("kernel_median_us"): data[tok][(c["model"], c["op"])].append(c["kernel_median_us"])
models = sorted({m for t in data.values() for m, _ in t})
def wsum(tok, m):
    try: return sum(W[o] * st.median(data[tok][(m, o)]) for o in W)
    except Exception: return float("nan")
base = {m: wsum("base", m) for m in models}
print("token," + ",".join(f"{m}_us,{m}_x" for m in models) + ",geomean_x,max_repeat_spread_pct")
for tok in data:
    xs = []; row = [tok]
    for m in models:
        s = wsum(tok, m); x = base[m] / s if s == s and s > 0 else float("nan"); xs.append(x); row += [f"{s:.0f}", f"{x:.3f}"]
    g = 1.0
    for x in xs: g *= x
    sp = max((max(v) - min(v)) / st.median(v) * 100 for v in data[tok].values())
    print(",".join(row) + f",{g ** (1 / len(xs)):.3f},{sp:.1f}")
