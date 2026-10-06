#!/usr/bin/env python3
"""trace_families.py <trace dir>: per-kernel GPU time of the warm prefill in every <model>-<scheme>-<arm>.etdp.
Vulkan logs one profile event per dispatch, '{<op json with kernel_name>, "dispatch_id": N}', after each
ETVK_COMPUTE_GRAPH_EXECUTE; --warmup gives two executions of the prefill and the last one is used (the warm one).
Prints csv: cell, arm, family, kernel, dispatches, ms, share. Families as the kit's trace_analysis.py groups them.
The deserializer is imported from an exported source tree (PYTHONPATH with an 'executorch' link to it)."""
import collections, glob, os, re, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("ET_PYPATH", "<artifacts>/venv/etpath"))
from executorch.devtools.etdump.serialize import deserialize_from_etdump_flatcc

DISPATCH = re.compile(r'^\{(.+), "dispatch_id": (\d+)\}$', re.S)
KERNEL = re.compile(r'"kernel_name": "([^"]+)"')

def family(k):
    if "sdpa_fused" in k: return "attention: fused"
    if "sdpa_kvt" in k or "sdpa_vt" in k: return "attention: K/V copy"
    if "sdpa_compute_attn_weights" in k or "sdpa_qk" in k: return "attention: QK^T"
    if "softmax" in k: return "attention: softmax"
    if "sdpa_compute_out" in k or "sdpa_av" in k: return "attention: attn*V"
    if "q4gsw" in k or "dq8ca" in k or "linear" in k: return "linear"
    if "quantize" in k: return "quantize"
    if "rotary" in k or "rope" in k: return "rope"
    if "rms_norm" in k: return "rms_norm"
    if "view_copy" in k or "copy" in k or "transfer" in k or "nchw" in k: return "copy"
    if "binary" in k or "unary" in k or "mul" in k or "add" in k or "sigmoid" in k: return "elementwise"
    if "cache" in k: return "kv cache"
    return "other"

def last_execution(path):
    et = deserialize_from_etdump_flatcc(open(path, "rb").read())
    runs = []
    for rd in et.run_data:
        cur = None
        for ev in rd.events or []:
            pe = ev.profile_event
            if pe is None: continue
            name = pe.name or pe.delegate_debug_id_str or ""
            if name == "ETVK_COMPUTE_GRAPH_EXECUTE":
                cur = []; runs.append(cur); continue
            m = DISPATCH.match(name)
            if m and cur is not None:
                k = KERNEL.search(m.group(1))
                cur.append((k.group(1) if k else m.group(1), (pe.end_time - pe.start_time) / 1e6))
    runs = [r for r in runs if r]
    return max(runs[-2:], key=lambda r: sum(t for _, t in r)) if len(runs) >= 2 else (runs[-1] if runs else [])

print("cell,arm,family,kernel,dispatches,ms,share")
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.etdp"))):
    m, q, arm = os.path.basename(p)[:-5].split("-", 2)
    ex = last_execution(p); tot = sum(t for _, t in ex) or 1.0
    agg = collections.defaultdict(lambda: [0, 0.0])
    for k, t in ex: agg[(family(k), k)][0] += 1; agg[(family(k), k)][1] += t
    for (f, k), (n, t) in sorted(agg.items(), key=lambda x: -x[1][1]):
        print(f"{m}-{q},{arm},{f},{k},{n},{t:.3f},{t / tot:.4f}")
    print(f"{m}-{q},{arm},TOTAL,-,{len(ex)},{tot:.3f},1.0000")
