#!/usr/bin/env python3
"""roof_util.py <trace gemm.csv> [...]: time-weighted prefill GEMM rate per cell and arm from the warm ETDump
(sum of 2*M*N*K over sum of dispatch time) and its share of the matching confirmed 780M roof
(openspec/changes/sarc-1.5-e2e-benchmark/evidence/roofline.md section 2: fp16->fp32 matrix 14.772 TFLOP/s for
4w, int8 matrix 14.393 TOP/s for 8da4w; fast plan, short run, clocks not pinned)."""
import csv, collections, sys
ROOF = {"4w": 14.772, "8da4w": 14.393}
for f in sys.argv[1:]:
    acc = collections.defaultdict(lambda: [0.0, 0.0, set()])
    for r in csv.DictReader(open(f)):
        k = (r["model"], r["scheme"], r["build"]); ms = float(r["ms"])
        acc[k][0] += 2.0 * int(r["M"]) * int(r["N"]) * int(r["K"]); acc[k][1] += ms; acc[k][2].add(r["kernel"].replace("sarc_linear_", "").replace("_texture3d_texture2d_half", ""))
    print(f"# {f}\nmodel,scheme,arm,gemm_ms,rate_T_per_s,pct_of_matrix_roof,kernels")
    for (m, q, b), (ops, ms, ks) in sorted(acc.items()):
        rate = ops / (ms * 1e-3) / 1e12
        print(f"{m},{q},{b},{ms:.1f},{rate:.3f},{rate / ROOF[q] * 100:.1f},{'+'.join(sorted(ks))}")
