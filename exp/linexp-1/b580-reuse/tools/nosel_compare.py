#!/usr/bin/env python3
"""nosel_compare.py <data/nosel>: per configuration (none, fused1) compare the kernel dispatched for every case
(model, scheme, op) between the parent build and the experiment build. Prints the differences and a verdict."""
import json, sys
D = sys.argv[1]; ok = True
for c in ("none", "fused1"):
    k = {b: {(x["model"], x["scheme"], x["op"]): x["kernel"] for x in json.load(open(f"{D}/{b}-{c}.json"))["cases"]} for b in ("parent", "exp1")}
    diff = [(key, k["parent"].get(key), k["exp1"].get(key)) for key in sorted(set(k["parent"]) | set(k["exp1"])) if k["parent"].get(key) != k["exp1"].get(key)]
    exp = sorted({v for v in k["exp1"].values() if "sarc_exp_" in v})
    print(f"{c}: {len(k['parent'])} cases parent, {len(k['exp1'])} cases exp1, {len(diff)} differ, experiment kernels dispatched: {exp or 'none'}; kernels: {sorted(set(k['exp1'].values()))}")
    for d in diff: print("  DIFF", d)
    ok = ok and not diff and not exp and len(k["parent"]) == len(k["exp1"]) > 0
print("NOTHING_SELECTED_OK" if ok else "NOTHING_SELECTED_FAILED")
