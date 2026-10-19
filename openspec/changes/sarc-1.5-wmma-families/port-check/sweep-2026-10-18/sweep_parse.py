#!/usr/bin/env python3
"""Compare etvk_bench sweep logs: per case, median op time of each arm over rounds.

usage: sweep_parse.py DIR PREFIX ARM1 ARM2 [ARM3...]
  logs are DIR/<PREFIX><arm>_r<N>.log; the first arm is the baseline.
"""
import glob
import re
import statistics
import sys

d, prefix, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
row = re.compile(r"^(\S+)\s.*?\s(linear_\S+_Texture3D)\s+\[\S+\]\s+([\d.]+) μs")
kern = re.compile(r"^(linear_\S+|q4gsw_\S+)\.\.")


def load(arm):
    times = {}
    kernels = {}
    for f in sorted(glob.glob(f"{d}/{prefix}{arm}_r*.log")):
        last_k = None
        for line in open(f, encoding="utf-8", errors="replace"):
            mk = kern.match(line)
            if mk and "quantize" not in line and "transpose" not in line:
                last_k = mk.group(1)
            m = row.match(line)
            if m:
                name = m.group(1)
                case = m.group(2)
                k = name if not name.startswith("et_vk") else (last_k or name)
                times.setdefault(case, []).append(float(m.group(3)))
                kernels[case] = k
                last_k = None
    return {c: statistics.median(v) for c, v in times.items()}, kernels


data = {a: load(a) for a in arms}
base_t, _ = data[arms[0]]
print("case".ljust(44) + "".join(f"{a:>10}" for a in arms) + "".join(f"{a + '/' + arms[0]:>12}" for a in arms[1:]) + "  kernels")
for case in base_t:
    ts = [data[a][0].get(case) for a in arms]
    if None in ts:
        continue
    ratios = [ts[0] / t for t in ts[1:]]  # speedup vs baseline
    ks = " | ".join(data[a][1].get(case, "?")[:34] for a in arms[1:])
    short = case.replace("_Texture3D", "")
    print(short.ljust(44) + "".join(f"{t:10.1f}" for t in ts) + "".join(f"{r:12.3f}" for r in ratios) + "  " + ks)
