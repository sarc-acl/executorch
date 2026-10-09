#!/usr/bin/env python3
"""trace_attention.py <stage/<session>>: the attention kernels of the session's warm ETDumps by kernel name
(the family table of trace_analysis.py puts the fused kernel and its copy pass under copy/view/other).
Per cell and arm, ms per prefill: QK^T, softmax, attn*V, fused kernel, K/V copy pass, their sum, the number of
dispatches of each, and the total dispatch time. Writes <session>/trace/attention.csv and prints it."""
import csv, json, sys, warnings
from pathlib import Path
from executorch.devtools import Inspector
warnings.filterwarnings("ignore")
D = Path(sys.argv[1]); T = D / "trace" / "raw" / "xe2" / "trace2"
def kind(k):
    if "sdpa_fused" in k: return "fused"
    if "sdpa_kvt" in k: return "kv_copy"
    if "softmax" in k: return "softmax"
    if "sdpa_compute_attn_weights" in k or "sdpa_qk" in k: return "qk"
    if "sdpa_compute_out" in k or "sdpa_av" in k: return "av"
    return None
K = ["qk", "softmax", "av", "fused", "kv_copy"]; rows = []
for f in sorted(T.glob("*.etdp")):
    m, q, b = f.stem.split("-"); ms = dict.fromkeys(K, 0.0); n = dict.fromkeys(K, 0); total = 0.0
    for _, r in Inspector(etdump_path=str(f)).to_dataframe().iterrows():
        name = str(r.event_name)
        if not name.startswith("{"): continue
        try: k = json.loads(name).get("kernel_name", "")
        except Exception: continue
        if not k: continue
        raw = r.raw if isinstance(r.raw, list) else [r.raw]; t = float(raw[-1]); total += t
        c = kind(k)
        if c and t > 0: ms[c] += t; n[c] += 1
    rows.append([m, q, b] + [f"{ms[k]:.2f}" for k in K] + [f"{sum(ms.values()):.2f}", f"{total:.2f}"] + [n[k] for k in K])
hdr = ["model", "scheme", "arm"] + [k + "_ms" for k in K] + ["attention_ms", "dispatch_ms"] + [k + "_n" for k in K]
with open(D / "trace" / "attention.csv", "w", newline="") as o: csv.writer(o).writerows([hdr] + rows)
for r in [hdr] + rows: print(",".join(str(x) for x in r))
