"""Per-kernel GPU time from the warmup ETDumps: raw/<gpu>/trace2/<model>-<scheme>-<build>.etdp.

Each ETDump holds two executions of the prefill (--warmup, then the measured one); every
event's `raw` list has one duration per execution and we use the last (the warm one).

Families are by kernel name, except that stock 4w's input-transpose dispatch (issued by the linear
operator) is its own family. totals.csv also has linear_op_ms: every dispatch issued by a linear
operator (GEMM + 8-bit activation quantize + input transpose), i.e. the operator-level time.

Outputs (report/evidence/trace/):
  families.csv  gpu, model, scheme, build, family, ms, share   (per-family GPU time)
  gemm.csv      gpu, model, scheme, build, kernel, M, N, K, ms, tflops   (prefill GEMM dispatches)
  totals.csv    gpu, model, scheme, build, dispatch_ms, graph_ms, n_dispatch

Run with the dev/1.5 venv:
  ~/Desktop/sarc-acl/dev/1.5/executorch/.venv/bin/python report/trace_analysis.py
"""

import csv
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

from executorch.devtools import Inspector

warnings.filterwarnings("ignore")
C = Path(__file__).resolve().parent.parent
OUT = C / "report" / "evidence" / "trace"
GPUS = ["780m", "b580", "b70", "4070ti", "orin"]
SUB = sys.argv[1] if len(sys.argv) > 1 else "trace2"


def family(k, m_rows, opname=""):
    if "transpose" in k and ("linear" in opname or "q4gsw" in opname):
        return "linear-op input transpose"
    if "sdpa_compute_attn_weights" in k or "sdpa_qk" in k:
        return "attention: QK^T"
    if "softmax" in k:
        return "attention: softmax"
    if "sdpa_compute_out" in k or "sdpa_av" in k:
        return "attention: AV"
    if "kv_cache" in k:
        return "attention: KV update"
    if "quantize_and_pack" in k or "choose_qparams" in k or "quantize" in k:
        return "8-bit activation quantize"
    if "rotary" in k or "rope" in k:
        return "RoPE"
    if "embedding" in k:
        return "embedding"
    if "gemv" in k or (("linear" in k or "q4gsw" in k or "dq8ca" in k) and m_rows == 1):
        return "LM head (1 token)"
    if "linear" in k or "q4gsw" in k or "dq8ca" in k:
        return "prefill GEMM (linear layers)"
    if "rms_norm" in k:
        return "RMSNorm"
    if "rotary" in k or "rope" in k:
        return "RoPE"
    if "binary" in k or "unary" in k or "silu" in k or "sigmoid" in k or "mul" in k or "add" in k:
        return "elementwise"
    if "nchw" in k or "staging" in k:
        return "staging"
    return "copy/view/other"


def shapes(op):
    """(M, N, K) of a linear op from its operator args: input [.., M, K], output [.., M, N]."""
    tens = [a for a in op.get("args", []) if a.get("type") == "TENSOR"]
    if len(tens) < 2:
        return None
    inp, out = tens[0]["sizes"], tens[-1]["sizes"]
    M, K, N = inp[-2] if len(inp) >= 2 else 1, inp[-1], out[-1]
    return int(M), int(N), int(K)


def load(path):
    df = Inspector(etdump_path=str(path)).to_dataframe()
    graph = None
    rows = []
    for _, r in df.iterrows():
        n = str(r.event_name)
        raw = r.raw if isinstance(r.raw, list) else [r.raw]
        if n == "ETVK_COMPUTE_GRAPH_EXECUTE":
            graph = float(raw[-1])
            continue
        if not n.startswith("{"):
            continue
        try:
            j = json.loads(n)
        except Exception:
            continue
        k = j.get("kernel_name", "")
        if not k:
            continue
        op = j.get("operator") or {}
        mnk = shapes(op) if any(s in (op.get("name") or "") for s in ("linear", "q4gsw", "dq8ca")) else None
        rows.append((k, float(raw[-1]), mnk, len(raw), op.get("name") or ""))
    return rows, graph


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fam_rows, gemm_rows, tot_rows = [], [], []
    for g in GPUS:
        d = C / "raw" / g / SUB
        for p in sorted(d.glob("*.etdp")):
            m, q, b = p.stem.split("-")
            rows, graph = load(p)
            execs = max((r[3] for r in rows), default=0)
            lin_op = sum(r[1] for r in rows if any(t in r[4] for t in ("linear_q4gsw", "linear_dq8ca")))
            fam = defaultdict(float)
            for k, ms, mnk, _, opn in rows:
                f = family(k, mnk[0] if mnk else 0, opn)
                fam[f] += ms
                if f == "prefill GEMM (linear layers)" and mnk:
                    M, N, K = mnk
                    gemm_rows.append({"gpu": g, "model": m, "scheme": q, "build": b, "kernel": k,
                                      "M": M, "N": N, "K": K, "ms": round(ms, 4),
                                      "tflops": round(2 * M * N * K / (ms * 1e-3) / 1e12, 3)})
            tot = sum(fam.values())
            for f, ms in sorted(fam.items(), key=lambda x: -x[1]):
                fam_rows.append({"gpu": g, "model": m, "scheme": q, "build": b, "family": f,
                                 "ms": round(ms, 3), "share": round(ms / tot, 4)})
            tot_rows.append({"gpu": g, "model": m, "scheme": q, "build": b, "dispatch_ms": round(tot, 3),
                             "graph_ms": round(graph or 0, 3), "n_dispatch": len(rows), "executions": execs,
                             "linear_op_ms": round(lin_op, 3)})
            print(f"{g} {m} {q} {b}: {tot:.1f} ms in {len(rows)} dispatches (graph {graph:.1f} ms, {execs} exec)")
    for name, rows in [("families.csv", fam_rows), ("gemm.csv", gemm_rows), ("totals.csv", tot_rows)]:
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
