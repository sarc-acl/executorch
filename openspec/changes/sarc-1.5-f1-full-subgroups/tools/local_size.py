#!/usr/bin/env python3
"""local_size.py <generated shader dir> <rows.csv of tools/dump_rows> > data/local-size.csv

Static check of VUID-VkPipelineShaderStageCreateInfo-pNext-02757 for every generated SARC shader variant that
requests a required subgroup size (the line "// REQUIRED_SUBGROUP_SIZE = N" that gen_vulkan_spv.py writes from
the yaml's SUBGROUP_SIZE): with the full-subgroups flag the local size in X must be a multiple of N.

The local size is not in the shader (local_size_x_id = 0, a specialization constant): the op sets it at run time.
Where it comes from, per variant (column source):
  row        the TileDims of the selection row(s) / sweep candidate(s) naming the kernel: sg_grid_x * sg_grid_y *
             subgroup_size (Select.h wg_size(); Q4gswCoopmat.cpp, Dq8caCoopmat.cpp, SdpaCoopmat.cpp)
  fused      sarc_dev_b580_sdpa_fused_d<D>_t<M>x<N>s<S>m8[g<G>]...: S * G (impl/sarc_dev/b580/SdpaB580Fused.cpp)
  constants  no row names the variant (it cannot be dispatched): the shader's own SG_GRID_X * SG_GRID_Y *
             SUBGROUP_SIZE, which is what a row for it would have to carry
One line per (variant, distinct local size). selected_on lists the devices whose table rows select the kernel
(verified/unverified); candidates are sweep rows selected by name only.
The upstream (non-SARC) shaders with a required subgroup size are listed last with source "upstream": their
local size is set in upstream op code and is not derived here."""
import csv, glob, os, re, sys, collections
gen, rows_csv = sys.argv[1:3]
rows = collections.defaultdict(list)
for r in csv.DictReader(open(rows_csv)): rows[r["kernel_base"]].append(r)
bases = sorted(rows, key=len, reverse=True)
out = csv.writer(sys.stdout, lineterminator="\n")
out.writerow(["variant", "zone", "required_subgroup_size", "local_size_x", "multiple", "row_subgroup_size_equal", "source", "selected_on"])
n = bad = 0; upstream = []
for p in sorted(glob.glob(os.path.join(gen, "*.glsl"))):
    t = open(p, errors="replace").read()
    m = re.match(r"// REQUIRED_SUBGROUP_SIZE = (\d+)", t)
    if not m: continue
    name = os.path.basename(p)[:-5]; req = int(m[1])
    if not name.startswith("sarc"): upstream.append((name, req)); continue
    zone = "dev" if re.search(r"^sarc_dev_|_(sweep|ml|pk|xe2|xe2c)_", name) else "release"
    base = next((b for b in bases if name.startswith(b + "_")), None)
    cases = []
    if base:
        by = collections.defaultdict(list)
        for r in rows[base]: by[(int(r["local_size_x"]), int(r["subgroup_size"]))].append(f'{r["device"]}:{r["status"]}' if r["list"] == "row" else "candidate")
        cases = [(x, sg == req, "row", ";".join(sorted(set(d)))) for (x, sg), d in sorted(by.items())]
    elif f := re.match(r"sarc_dev_b580_sdpa_fused_d\d+_t\d+x\d+s(\d+)m8(?:g(\d+))?", name):
        cases = [(int(f[1]) * int(f[2] or 1), int(f[1]) == req, "fused", "bmg g21:dev profile (b580-fused1/2) or by name")]
    else:
        c = {k: int(v) for k, v in re.findall(r"^const uint (SG_GRID_X|SG_GRID_Y|SUBGROUP_SIZE) = (\d+);", t, re.M)}
        cases = [(c["SG_GRID_X"] * c["SG_GRID_Y"] * c["SUBGROUP_SIZE"], c["SUBGROUP_SIZE"] == req, "constants", "no row")] if len(c) == 3 else [("", "", "unknown", "no row")]
    for x, eq, src, sel in cases:
        ok = x != "" and x % req == 0; n += 1; bad += not (ok and eq)
        out.writerow([name, zone, req, x, "yes" if ok else "NO", "yes" if eq else "NO", src, sel])
for name, req in upstream: out.writerow([name, "upstream", req, "", "", "", "upstream", ""])
print(f"local_size.py: {n} lines, {bad} not a multiple or with a row subgroup size different from the shader's", file=sys.stderr)
