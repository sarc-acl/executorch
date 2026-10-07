#!/usr/bin/python3
"""etdump_families.py <trace dir>: per-kernel GPU time from the warm-up ETDumps <m>-<q>-<b>.etdp of trace.sh.

The kit's trace_analysis.py needs executorch.devtools.Inspector (torch); on this host that import takes minutes
from NFS, so the ETDump is decoded here with the build's own flatc (--json --size-prefixed, the same call the
Inspector makes) and the same reading as trace_analysis.py: Vulkan dispatch events are named by a JSON string with
kernel_name (in delegate_debug_id_str; no operator metadata in these dumps, so the linear shapes come from the
order of the dispatches only where stated); the last of the two executions (--warmup) is the warm one; the kernel families are
trace_analysis.py's, plus the fused attention node of this branch.

Outputs in <trace dir>: families.csv (model, scheme, build, family, ms, share), kernels.csv (per kernel name and
linear shape: count, ms), totals.csv (dispatch_ms, graph_ms, n_dispatch per run)."""
import csv, glob, json, os, subprocess, sys, tempfile
from collections import defaultdict

ART = "<campaign-root>/.artifacts"
FLATC = f"{ART}/build/rx7600/parent-traced/llama/third-party/flatc_ep/bin/flatc"
SCHEMA = f"{ART}/src/rx7600/parent/executorch/devtools/etdump"


def family(k, m_rows, opname=""):  # kit/analysis/trace_analysis.py, plus the fused attention node
    if "sdpa_fused" in k or "sdpa_kvt" in k or "sdpa_vt" in k:
        return "attention: fused (incl. K/V copy pass)"
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
    if "binary" in k or "unary" in k or "silu" in k or "sigmoid" in k or "mul" in k or "add" in k:
        return "elementwise"
    if "nchw" in k or "staging" in k:
        return "staging"
    return "copy/view/other"


def shapes(op):
    tens = [a for a in op.get("args", []) if a.get("type") == "TENSOR"]
    if len(tens) < 2:
        return None
    inp, out = tens[0]["sizes"], tens[-1]["sizes"]
    return int(inp[-2] if len(inp) >= 2 else 1), int(out[-1]), int(inp[-1])


def decode(path):
    with tempfile.TemporaryDirectory(dir=f"{ART}/tmp") as d:
        subprocess.run([FLATC, "--json", "--strict-json", "--raw-binary", "--size-prefixed", "--defaults-json",
                        "-I", SCHEMA, "-o", d, f"{SCHEMA}/etdump_schema_flatcc.fbs", "--", path],
                       check=True, capture_output=True)
        return json.load(open(glob.glob(f"{d}/*.json")[0]))


def load(path):
    runs = decode(path).get("run_data", [])
    out = []
    for rd in runs:
        rows, graph = [], None
        for e in rd.get("events", []):
            pe = e.get("profile_event")
            n = (pe or {}).get("name") or (pe or {}).get("delegate_debug_id_str")
            if not n:
                continue
            ms = (int(pe.get("end_time", 0)) - int(pe.get("start_time", 0))) / 1e6
            if n == "ETVK_COMPUTE_GRAPH_EXECUTE":
                graph = ms
                continue
            if not n.startswith("{"):
                continue
            try:
                j = json.loads(n)
            except ValueError:
                continue
            k = j.get("kernel_name", "")
            if not k:
                continue
            op = j.get("operator") or {}
            opn = op.get("name") or ""
            mnk = shapes(op) if any(s in opn for s in ("linear", "q4gsw", "dq8ca")) else None
            rows.append((k, ms, mnk, opn))
        out.append((rows, graph))
    return out


def main():
    d = sys.argv[1]
    fam_rows, ker_rows, tot_rows = [], [], []
    for p in sorted(glob.glob(f"{d}/*.etdp")):
        m, q, b = os.path.basename(p)[:-5].split("-")
        execs = load(p)
        rows, graph = [x for x in execs if x[0]][-1]
        fam, ker = defaultdict(float), defaultdict(lambda: [0, 0.0])
        for k, ms, mnk, opn in rows:
            fam[family(k, mnk[0] if mnk else 0, opn)] += ms
            key = (k, "x".join(map(str, mnk)) if mnk else "")
            ker[key][0] += 1
            ker[key][1] += ms
        tot = sum(fam.values())
        for f, ms in sorted(fam.items(), key=lambda x: -x[1]):
            fam_rows.append({"model": m, "scheme": q, "build": b, "family": f, "ms": round(ms, 3), "share": round(ms / tot, 4)})
        for (k, s), (n, ms) in sorted(ker.items(), key=lambda x: -x[1][1]):
            ker_rows.append({"model": m, "scheme": q, "build": b, "kernel": k, "MxNxK": s, "count": n, "ms": round(ms, 3)})
        tot_rows.append({"model": m, "scheme": q, "build": b, "dispatch_ms": round(tot, 3), "graph_ms": round(graph or 0, 3),
                         "n_dispatch": len(rows), "executions": len(execs)})
        print(f"{m} {q} {b}: {tot:.1f} ms in {len(rows)} dispatches (graph {graph or 0:.1f} ms, {len(execs)} executions)")
    for name, rows in [("families.csv", fam_rows), ("kernels.csv", ker_rows), ("totals.csv", tot_rows)]:
        if rows:
            with open(os.path.join(d, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)


if __name__ == "__main__":
    main()
