#!/usr/bin/env python3
"""Compare test_llama_microbench --json-out results: dispatched kernel and
kernel median time per (model, scheme, op, storage).

usage: compare_kernels.py <new.json> <reference.json | dir of *-r*.json> [--scheme 4w] [--tol 0.03]

The reference is typically an earlier confirmation (several repeats: the
median over repeats is used). Prints one line per case and a summary; exits 1
if any case is outside +-tol or missing. Kernel names are shown for both sides
(they differ by design when comparing release 1.4 names with SARC names).
"""
import argparse, glob, json, os, re, statistics as st, sys

ap = argparse.ArgumentParser()
ap.add_argument("new"); ap.add_argument("ref")
ap.add_argument("--scheme", default=None); ap.add_argument("--tol", type=float, default=0.03)
ap.add_argument("--suite", default="linear", help="reference file prefix when ref is a dir")
a = ap.parse_args()

def cases(path):
    cs = json.load(open(path))["cases"]
    for c in cs:  # some drivers (NVIDIA Tegra) report '"kernel_name": "<name>", ...'
        m = re.search(r'"kernel_name": "([^"]+)"', c.get("kernel", ""))
        if m:
            c["kernel"] = m.group(1)
    return cs

key = lambda c: (c["model"], c["scheme"], c["op"], c["storage"])
new = {key(c): c for c in cases(a.new) if c.get("regime", "prefill") == "prefill"}
refs = {}
files = sorted(glob.glob(os.path.join(a.ref, f"{a.suite}-r*.json"))) if os.path.isdir(a.ref) else [a.ref]
for f in files:
    for c in cases(f):
        refs.setdefault(key(c), []).append(c)

bad = 0; ratios = []
for k in sorted(new):
    if a.scheme and k[1] != a.scheme:
        continue
    c = new[k]; r = refs.get(k)
    if not r:
        print(f"MISSING ref {k}"); bad += 1; continue
    m = st.median(x["kernel_median_us"] for x in r)
    ratio = c["kernel_median_us"] / m; ratios.append(ratio)
    flag = "OK " if abs(ratio - 1) <= a.tol else "OUT"
    bad += flag == "OUT"
    print(f"{flag} {k[0]:13s} {k[1]:6s} {k[2]:6s} {k[3]:9s} {m:9.1f} -> {c['kernel_median_us']:9.1f} us "
          f"({ratio:.3f})  {r[0]['kernel']}  ->  {c['kernel']}")
if ratios:
    print(f"cases {len(ratios)}  ratio median {st.median(ratios):.3f}  min {min(ratios):.3f}  "
          f"max {max(ratios):.3f}  outside +-{a.tol:.0%}: {bad}")
sys.exit(1 if bad else 0)
