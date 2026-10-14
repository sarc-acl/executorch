#!/usr/bin/env python3
"""shipped.py <artifact build dir> <parent tag> <candidate tag>: workstation side. Shipped SPIR-V of a cross build.

The goldens (sarc/golden/spirv.json) were made with the glslc of the x86 build image (shaderc v2023.8); the cross
image for the Orin has shaderc v2026.1, with which some shipped variants compile to different bytes even on the
pristine parent. So two comparisons are made from the per-build hash lists (build/<tag>.spv.sha256):
  1. every shipped variant (the names of the golden) is byte-identical between the parent build and the
     candidate build: the candidate changed no shipped shader;
  2. against the golden, the candidate has exactly the parent's set of differing variants (the compiler's), and
     none of them is a variant the Orin rows dispatch.
Exit 0 only when both hold."""
import json, os, re, sys
bd, P, C = sys.argv[1:4]; here = os.path.dirname(os.path.abspath(__file__)); et = os.path.normpath(os.path.join(here, "../../../.."))
gold = json.load(open(os.path.join(et, "sarc/golden/spirv.json")))["variants"]
def hashes(tag): return {l.split()[1].removesuffix(".spv"): l.split()[0] for l in open(os.path.join(bd, tag + ".spv.sha256"))}
p, c = hashes(P), hashes(C); bad = 0
missing = [n for n in gold if n not in p or n not in c]
changed = [n for n in gold if n in p and n in c and p[n] != c[n]]
dp = sorted(n for n in gold if p.get(n) != gold[n]["sha256"]); dc = sorted(n for n in gold if c.get(n) != gold[n]["sha256"])
orin = set(re.findall(r'"tegra orin"[^"]*"(sarc_[a-z0-9_]+)"', open(os.path.join(et, "backends/vulkan/runtime/graph/ops/impl/sarc/table_nvidia.cpp")).read()))
orin_diff = [n for n in dc if any(n.startswith(k + "_") for k in orin)]
print(f"shipped variants in the golden: {len(gold)}; missing in a build: {len(missing)}; parent vs candidate different: {len(changed)} {changed}")
print(f"differ from the golden (cross glslc): parent {len(dp)}, candidate {len(dc)}, same set: {dp == dc}; owners: {sorted({gold[n]['owner'] for n in dc})}")
print(f"Orin kernels ({len(orin)} bases): variants differing from the golden: {len(orin_diff)} {orin_diff}")
ok = not missing and not changed and dp == dc and not orin_diff
print("shipped SPIR-V:", "UNCHANGED" if ok else "CHANGED"); sys.exit(0 if ok else 1)
