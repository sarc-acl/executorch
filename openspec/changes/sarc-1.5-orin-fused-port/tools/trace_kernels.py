#!/usr/bin/env python3
"""trace_kernels.py <trace dir of a session>: workstation side, after trace_analyze.sh. The kit's trace_analysis.py
counts the fused attention kernel and its copy pass under "copy/view/other". This reads the same warm ETDumps
(<trace dir>/raw/orin/trace2/<model>-<scheme>-<build>.etdp) with the kit's own loader and writes, beside the kit's
files (<trace dir>/report/evidence/trace/):
  attention.csv  model, scheme, build, qk_ms, softmax_ms, av_ms, fused_ms, kv_copy_ms, attention_ms, other_copy_ms,
                 gemm_ms, dispatch_ms       (other_copy_ms = the kit's copy/view/other without the fused node)
  kernels.csv    model, scheme, build, kernel, dispatches, ms     (every attention kernel by name)"""
import csv, importlib.util, pathlib, sys, collections
C = pathlib.Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("trace_analysis", C / "tools" / "trace_analysis.py"); ta = importlib.util.module_from_spec(spec); spec.loader.exec_module(ta)
def fam(k, m_rows, opn):
    if "_sdpa_fused" in k: return "fused"
    if "_sdpa_kvt" in k or "_sdpa_vt" in k: return "kv_copy"
    return {"attention: QK^T": "qk", "attention: softmax": "softmax", "attention: AV": "av", "prefill GEMM (linear layers)": "gemm",
            "copy/view/other": "other_copy"}.get(ta.family(k, m_rows, opn), "rest")
att, ker = [], []
for p in sorted((C / "raw" / "orin" / "trace2").glob("*.etdp")):
    m, q, b = p.stem.split("-"); rows, _ = ta.load(p); f = collections.defaultdict(float); kk = collections.defaultdict(lambda: [0, 0.0])
    for k, ms, mnk, _, opn in rows:
        x = fam(k, mnk[0] if mnk else 0, opn); f[x] += ms
        if x in ("qk", "softmax", "av", "fused", "kv_copy"): kk[k][0] += 1; kk[k][1] += ms
    a = sum(f[x] for x in ("qk", "softmax", "av", "fused", "kv_copy"))
    att.append([m, q, b] + [round(f[x], 2) for x in ("qk", "softmax", "av", "fused", "kv_copy")] + [round(a, 2), round(f["other_copy"], 2), round(f["gemm"], 2), round(sum(f.values()), 2)])
    ker += [[m, q, b, k, n, round(ms, 3)] for k, (n, ms) in sorted(kk.items())]
O = C / "report" / "evidence" / "trace"; O.mkdir(parents=True, exist_ok=True)
for name, head, rows in (("attention.csv", "model,scheme,build,qk_ms,softmax_ms,av_ms,fused_ms,kv_copy_ms,attention_ms,other_copy_ms,gemm_ms,dispatch_ms", att),
                         ("kernels.csv", "model,scheme,build,kernel,dispatches,ms", ker)):
    with open(O / name, "w", newline="") as fh: w = csv.writer(fh, lineterminator="\n"); w.writerow(head.split(",")); w.writerows(rows)
for r in att: print(*r, sep=",")
