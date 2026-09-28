import csv, importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("ta", "/home/doremy/Desktop/sarc-acl/.artifacts/e2e-1.5-2026-09-28/report/trace_analysis.py")
ta = importlib.util.module_from_spec(spec); spec.loader.exec_module(ta)
d = Path("/home/doremy/Desktop/sarc-acl/.artifacts/e2e-1.5-2026-09-28/raw/orin/trace")
out = []
for p in sorted(d.glob("*.etdp")):
    m, q, b = p.stem.split("-")
    rows, graph = ta.load(p)
    execs = max((r[3] for r in rows), default=0)
    n=0
    for k, ms, mnk, _ in rows:
        f = ta.family(k, mnk[0] if mnk else 0)
        if f == "prefill GEMM (linear layers)" and mnk:
            M, N, K = mnk; n+=1
            out.append({"gpu": "orin", "model": m, "scheme": q, "build": b, "kernel": k, "M": M, "N": N, "K": K,
                        "ms": round(ms, 4), "tflops": round(2*M*N*K/(ms*1e-3)/1e12, 3), "executions": execs})
    tot=sum(r[1] for r in rows); gm=sum(r[1] for r in rows if ta.family(r[0], r[2][0] if r[2] else 0)=="prefill GEMM (linear layers)")
    print(p.name, "total_dispatch_ms", round(tot,1), "gemm_ms", round(gm,1), "gemm dispatches", n, "executions", execs, "graph_ms", graph, file=sys.stderr)
w = csv.DictWriter(open(sys.argv[1], "w", newline=""), fieldnames=list(out[0].keys()))
w.writeheader(); w.writerows(out)
