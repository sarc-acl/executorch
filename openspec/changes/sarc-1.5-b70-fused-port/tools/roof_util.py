#!/usr/bin/env python3
"""roof_util.py [--roof 4w=<TFLOP/s>] [--roof 8da4w=<TOP/s>] [--source <text>] <trace gemm.csv> [...]

Time-weighted prefill GEMM rate per cell and arm from the warm ETDump (sum of 2*M*N*K over the sum of dispatch
time). A percent-of-roof column is printed only for a scheme whose roof is given on the command line, and only
together with --source naming the igpu-roofline run it was measured in (fast plan, this card, this driver).
No roof is built in: the values in the older evidence are not to be reused, and without a fresh measurement
the rates are reported without percent-of-roof."""
import csv, collections, sys
ROOF, SOURCE, files = {}, "", []
it = iter(sys.argv[1:])
for x in it:
    if x == "--roof": k, v = next(it).split("="); ROOF[k] = float(v)
    elif x == "--source": SOURCE = next(it)
    else: files.append(x)
if ROOF and not SOURCE: sys.exit("--roof needs --source <igpu-roofline run it was measured in>")
print(f"# roofs: {ROOF if ROOF else 'none given, rates only'}; source: {SOURCE or '-'}")
for f in files:
    acc = collections.defaultdict(lambda: [0.0, 0.0, set()])
    for r in csv.DictReader(open(f)):
        k = (r["model"], r["scheme"], r["build"]); ms = float(r["ms"])
        acc[k][0] += 2.0 * int(r["M"]) * int(r["N"]) * int(r["K"]); acc[k][1] += ms; acc[k][2].add(r["kernel"].replace("sarc_linear_", "").replace("_texture3d_texture2d_half", ""))
    print(f"# {f}\nmodel,scheme,arm,gemm_ms,rate_T_per_s,pct_of_matrix_roof,kernels")
    for (m, q, b), (ops, ms, ks) in sorted(acc.items()):
        rate = ops / (ms * 1e-3) / 1e12
        print(f"{m},{q},{b},{ms:.1f},{rate:.3f},{f"{rate / ROOF[q] * 100:.1f}" if q in ROOF else "n/a"},{'+'.join(sorted(ks))}")
