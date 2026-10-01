#!/usr/bin/env python3
"""Compare (or update) the SHA-256 of every shipped SARC SPIR-V variant.

usage: spirv_golden.py <vulkan_compute_shaders dir> <golden.json> [--update --owner <device> --prefix <kernel_base>]

A shipped variant is any NAME in glsl/sarc/*.yaml. golden.json maps
NAME -> {"sha256", "owner"}; a mismatch names the owning device, whose
verification must precede updating the entry. --update rewrites the entries
whose NAME starts with --prefix (all when omitted) and records --owner for new
ones. The glslc version used is stored alongside (hashes differ across glslc
versions; always build with sarc/tools/build.sh, which pins it).
"""
import argparse, glob, hashlib, json, os, subprocess, sys

import yaml

root = subprocess.check_output(["git", "-C", os.path.dirname(__file__), "rev-parse", "--show-toplevel"], text=True).strip()
ap = argparse.ArgumentParser()
ap.add_argument("spv_dir"); ap.add_argument("golden")
ap.add_argument("--update", action="store_true"); ap.add_argument("--owner", default="")
ap.add_argument("--prefix", default=""); ap.add_argument("--glslc", default="")
a = ap.parse_args()

import itertools
names = []
for y in sorted(glob.glob(f"{root}/backends/vulkan/runtime/graph/ops/glsl/sarc/*.yaml")):
    for tmpl in yaml.safe_load(open(y)).values():
        # generate_variant_forall appends _<VALUE> per key, in key order.
        forall = tmpl.get("generate_variant_forall") or {}
        axes = []
        for key, vals in forall.items():
            if key == "combination":  # {parameter_names, combos: [{parameter_values, suffix?}]}
                axes.append([c.get("suffix") or "_".join(map(str, c["parameter_values"]))
                             for c in vals["combos"]])
            else:
                axes.append([x["VALUE"] for x in vals])
        suffixes = ["".join("_" + x for x in combo) for combo in itertools.product(*axes)] or [""]
        names += [v["NAME"] + sfx for v in tmpl.get("shader_variants", []) for sfx in suffixes]
gold = json.load(open(a.golden)) if os.path.exists(a.golden) else {"glslc": "", "variants": {}}
bad = 0
for n in names:
    p = os.path.join(a.spv_dir, n + ".spv")
    if not os.path.exists(p):
        print(f"MISSING spv {n}"); bad += 1; continue
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    g = gold["variants"].get(n)
    if a.update and n.startswith(a.prefix):
        gold["variants"][n] = {"sha256": h, "owner": (g or {}).get("owner") or a.owner}
        continue
    if g is None:
        print(f"NEW   {n} (no golden entry; add it with --update after verifying on its device)"); bad += 1
    elif g["sha256"] != h:
        print(f"DIFF  {n} (owner {g['owner']}: re-verify on that device, then --update)"); bad += 1
for n in list(gold["variants"]):
    if n not in names:
        print(f"GONE  {n} (in golden, not shipped)"); bad += 1
if a.update:
    if a.glslc: gold["glslc"] = a.glslc
    os.makedirs(os.path.dirname(os.path.abspath(a.golden)), exist_ok=True)
    with open(a.golden, "w") as f:
        f.write(json.dumps(gold, indent=2, sort_keys=True) + "\n")
    print(f"updated {a.golden}"); sys.exit(0)
print(f"spirv golden: {'PASS' if bad == 0 else 'FAIL'} ({len(names)} shipped variants)")
sys.exit(1 if bad else 0)
