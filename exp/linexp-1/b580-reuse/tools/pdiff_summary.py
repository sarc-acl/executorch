#!/usr/bin/env python3
"""pdiff_summary.py <data/pdiff> <out.csv>: one row per production-diff process: token, model, zero-point mode, exit
status, shapes checked, shapes PASSED, shapes with the cooperative-matrix kernel dispatched, the kernel name(s)
printed by the run, and whether every shape ran the token's own kernel (`base`: the incumbent)."""
import csv, glob, os, re, sys
INC = "sarc_dev_linear_dq8ca_coopmat_zpg_xe2bt_t128x128k64g84s16m8"
rows = [["token", "model", "zero_points", "rc", "shapes", "passed", "coopmat_dispatched", "kernels", "own_kernel_on_all_shapes", "all_passed", "shapes_failed_validation"]]
for f in sorted(glob.glob(sys.argv[1] + "/*.log")):
    m = re.match(r"(.*)-(llama-[\d.]+-\db)-(zp|nzp)\.log$", os.path.basename(f)); t = open(f, errors="replace").read()
    res = re.findall(r"\[production-diff\] \S+ -> (\S+) \((coopmat dispatched|NOT coopmat)[^,]*, correctness=(\w+)", t)
    want = INC if m[1] == "base" else "sarc_" + m[1]
    rows.append([m[1], m[2], "zero" if m[3] == "zp" else "nonzero", open(f[:-4] + ".rc").read().strip() if os.path.exists(f[:-4] + ".rc") else "", len(res),
                 sum(r[2] == "PASSED" for r in res), sum(r[1] == "coopmat dispatched" for r in res), ";".join(sorted({r[0] for r in res})),
                 len(res) == 4 and all(r[0].startswith(want + "_") for r in res), "ALL PASSED" in t, t.count("threw: Correctness validation failed")])
csv.writer(open(sys.argv[2], "w", newline="")).writerows(rows)
for r in rows: print(*[str(x).replace("exp_linear_dq8ca_coopmat_zpg_", "").replace("_texture3d_texture2d_half", "") for x in r], sep="\t")
