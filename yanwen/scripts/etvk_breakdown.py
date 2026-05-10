#!/usr/bin/env python3
"""Extract per-shader GPU breakdown from events.tsv."""
import csv
import json
import sys
from collections import defaultdict

path = sys.argv[1]
N_ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8

# kernel_name -> (count, sum_per_iter_ms_iter0, sum_per_iter_ms_steady)
agg = defaultdict(
    lambda: [0, 0.0, 0.0]
)  # count, total_iter0, total_steady (avg of iter 1..N-1)
total_iter0 = 0.0
total_steady = 0.0

with open(path) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        name = row.get("event_name", "")
        if "kernel_name" not in name:
            continue
        # event_name is a quoted JSON string with doubled quotes
        s = name.strip().strip('"').replace('""', '"')
        try:
            j = json.loads(s)
        except Exception:
            continue
        kernel = j.get("kernel_name") or j.get("operator", {}).get("name")
        if not kernel:
            continue
        # raw is a Python list literal as a string
        raw_str = row.get("raw", "").strip()
        if not raw_str.startswith("["):
            continue
        try:
            raw = json.loads(raw_str)
        except Exception:
            continue
        if len(raw) != N_ITERS:
            continue
        iter0 = raw[0]
        steady = sum(raw[1:]) / (N_ITERS - 1)
        a = agg[kernel]
        a[0] += 1
        a[1] += iter0
        a[2] += steady
        total_iter0 += iter0
        total_steady += steady

rows = sorted(agg.items(), key=lambda kv: kv[1][2], reverse=True)
print(
    f"\n=== ETVK kernel breakdown ({sum(v[0] for v in agg.values())} dispatches, {len(agg)} unique kernels) ==="
)
print(f"Iter 0 total GPU time: {total_iter0:.1f} ms")
print(f"Steady (avg iter 1..{N_ITERS-1}) total GPU time: {total_steady:.1f} ms")
print()
print(
    f"{'rank':>4}  {'kernel':<55}  {'count':>6}  {'iter0_ms':>10}  {'steady_ms':>10}  {'%steady':>8}"
)
for i, (k, (c, i0, s)) in enumerate(rows[:30]):
    pct = 100 * s / total_steady if total_steady else 0
    kshort = k if len(k) <= 55 else k[:52] + "..."
    print(f"{i+1:>4}  {kshort:<55}  {c:>6}  {i0:>10.2f}  {s:>10.2f}  {pct:>7.1f}%")

# Also bucket by op family
buckets = defaultdict(lambda: [0, 0.0, 0.0])
for k, (c, i0, s) in agg.items():
    if "mm_optim" in k or "matmul" in k or "linear" in k:
        b = "matmul/linear"
    elif "softmax" in k:
        b = "softmax"
    elif "rms_norm" in k or "layer_norm" in k or "norm" in k:
        b = "norm"
    elif "rope" in k or "rotary" in k:
        b = "rope"
    elif "binary" in k:
        b = "binary (add/mul/etc)"
    elif "buffer" in k or "copy" in k or "view" in k or "slice" in k:
        b = "layout/copy/slice"
    elif "sdpa" in k or "attention" in k or "bmm" in k:
        b = "attention"
    elif "silu" in k or "sigmoid" in k or "gelu" in k or "swiglu" in k:
        b = "activation (silu/sigmoid/etc)"
    else:
        b = f"other ({k[:30]})"
    buckets[b][0] += c
    buckets[b][1] += i0
    buckets[b][2] += s

print("\n=== Bucketed by op family ===")
print(
    f"{'family':<35}  {'count':>6}  {'iter0_ms':>10}  {'steady_ms':>10}  {'%steady':>8}"
)
for b, (c, i0, s) in sorted(buckets.items(), key=lambda kv: kv[1][2], reverse=True):
    pct = 100 * s / total_steady if total_steady else 0
    print(f"{b:<35}  {c:>6}  {i0:>10.2f}  {s:>10.2f}  {pct:>7.1f}%")
