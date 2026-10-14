#!/usr/bin/env python3
"""attention_families.py <stage/<session>/trace>: the attention block per cell and arm from the warm ETDumps of
trace.sh, with the fused kernel and its K / V copy pass as families of their own (the kit's trace_analysis.py
files both under copy/view/other). Uses the loader of the session's own copy of trace_analysis.py. Prints CSV:
model,scheme,build,qk_ms,softmax_ms,av_ms,fused_ms,kv_copy_ms,attention_ms,linear_gemm_ms,other_ms,total_ms,attention_share
Needs the ETDump inspector (TRACE_PY)."""
import glob, os, sys
C = os.path.abspath(sys.argv[1]); sys.argv = sys.argv[:1]; sys.path.insert(0, os.path.join(C, "tools"))
import trace_analysis as ta
def fam(k, m_rows, opn):
    if "sdpa_fused" in k: return "fused"
    if "sdpa_kvt" in k: return "kv_copy"
    f = ta.family(k, m_rows, opn)
    return {"attention: QK^T": "qk", "attention: softmax": "softmax", "attention: AV": "av", "prefill GEMM (linear layers)": "gemm"}.get(f, "other")
print("model,scheme,build,qk_ms,softmax_ms,av_ms,fused_ms,kv_copy_ms,attention_ms,linear_gemm_ms,other_ms,total_ms,attention_share")
for p in sorted(glob.glob(os.path.join(C, "raw", "4070ti", "trace2", "*.etdp"))):
    m, q, b = os.path.basename(p)[:-5].split("-"); rows, _ = ta.load(p); t = dict.fromkeys(("qk", "softmax", "av", "fused", "kv_copy", "gemm", "other"), 0.0)
    for k, ms, mnk, _, opn in rows: t[fam(k, mnk[0] if mnk else 0, opn)] += ms
    att = t["qk"] + t["softmax"] + t["av"] + t["fused"] + t["kv_copy"]; tot = sum(t.values())
    print(f'{m},{q},{b},{t["qk"]:.2f},{t["softmax"]:.2f},{t["av"]:.2f},{t["fused"]:.2f},{t["kv_copy"]:.2f},{att:.2f},{t["gemm"]:.2f},{t["other"]:.2f},{tot:.2f},{att / tot:.4f}')
