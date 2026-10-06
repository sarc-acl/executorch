#!/usr/bin/env python3
"""spv_compare.py <reference build dir> <build dir>: compares the SPIR-V of two builds made with the same toolchain.
Shipped variants (every NAME in glsl/sarc/*.yaml, as sarc/tools/spirv_golden.py enumerates them) must be
byte-identical; also reports how many other .spv files are identical, changed, added or removed."""
import glob, hashlib, itertools, os, subprocess, sys

import yaml

ref, new = (os.path.join(d, "vulkan_compute_shaders") for d in sys.argv[1:3])
root = subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--show-toplevel"], text=True).strip()
h = lambda d: {f[:-4]: hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest() for f in os.listdir(d) if f.endswith(".spv")}
a, b = h(ref), h(new)
names = []
for y in sorted(glob.glob(f"{root}/backends/vulkan/runtime/graph/ops/glsl/sarc/*.yaml")):
    for t in yaml.safe_load(open(y)).values():
        forall = t.get("generate_variant_forall") or {}
        axes = [[c.get("suffix") or "_".join(map(str, c["parameter_values"])) for c in v["combos"]] if k == "combination" else [x["VALUE"] for x in v] for k, v in forall.items()]
        sfx = ["".join("_" + x for x in c) for c in itertools.product(*axes)] or [""]
        names += [v["NAME"] + s for v in t.get("shader_variants", []) for s in sfx]
bad = [n for n in names if a.get(n) is None or a.get(n) != b.get(n)]
print(f"shipped variants: {len(names)}, byte-identical: {len(names) - len(bad)}, differ or missing: {len(bad)} {bad[:5]}")
common = set(a) & set(b)
print(f"all spv: identical {sum(a[k] == b[k] for k in common)}, changed {sum(a[k] != b[k] for k in common)}, added {len(set(b) - set(a))}, removed {len(set(a) - set(b))}")
for k in sorted(k for k in common if a[k] != b[k])[:20]: print("  changed", k)
print("SPV_COMPARE", "PASS" if not bad else "FAIL")
