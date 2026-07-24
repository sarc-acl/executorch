#!/usr/bin/env python3
import sys
from collections import defaultdict

from executorch.devtools import Inspector

etdump = sys.argv[1]
insp = Inspector(etdump_path=etdump)


# Bucket GPU kernel time by a coarse category derived from the kernel name.
def bucket(name):
    n = name.lower()
    if "dq8ca" in n and "coopmat" in n:
        return "dq8ca_linear_coopmat"
    if "dq8ca" in n:
        return "dq8ca_linear_tiled"
    if "q4gsw" in n and "coopmat" in n:
        return "q4gsw_linear_coopmat(4w/lm_head)"
    if "q4gsw" in n:
        return "q4gsw_linear_tiled"
    if "sdpa" in n or "attention" in n or "flash" in n:
        return "sdpa"
    if "quant" in n or "pack" in n or "qparam" in n or "choose" in n:
        return "quantize_and_pack"
    if "rope" in n or "rotary" in n:
        return "rope"
    if "norm" in n:
        return "norm"
    if "matmul" in n or "mm" in n or "bmm" in n:
        return "other_matmul"
    return f"other:{n[:40]}"


# Framework envelope events double-count leaf kernels; exclude from the
# per-kernel breakdown. perf_data.raw is in MILLISECONDS (cross-checked: the
# leaf-kernel sum matches wall-clock prefill within ~1%).
ENVELOPE = {
    "Method::execute",
    "DELEGATE_CALL",
    "ETVK_EXECUTE",
    "ETVK_COMPUTE_GRAPH_EXECUTE",
    "Program::load_method",
    "Method::init",
    "ETVK_COPY_INPUTS",
    "ETVK_COPY_OUTPUTS",
    "ETVK_RESIZE",
    "OPERATOR_CALL",
}

cat_ms = defaultdict(float)
cat_cnt = defaultdict(int)
total_ms = 0.0

for block in insp.event_blocks:
    for ev in block.events:
        name = ev.name or ""
        pd = ev.perf_data
        if pd is None or not pd.raw:
            continue
        if name in ENVELOPE:
            continue
        avg = sum(pd.raw) / len(pd.raw)  # per-dispatch, in ms
        b = bucket(name)
        cat_ms[b] += avg
        cat_cnt[b] += 1
        total_ms += avg

print(f"# etdump: {etdump}")
print(
    f"# leaf-kernel GPU time, one 2048-token prefill: {total_ms:.1f} ms "
    f"across {sum(cat_cnt.values())} dispatches\n"
)
print(f"{'category':38s} {'ms':>9s} {'%':>7s} {'disp':>6s}  {'tok/s if -10%':>13s}")
# tok/s sensitivity: prefill wall = total_ms (GPU-bound); tok/s = 2048/(wall_s).
for b, ms in sorted(cat_ms.items(), key=lambda kv: -kv[1]):
    saved = 0.10 * ms
    new_toks = 2048.0 / ((total_ms - saved) / 1000.0)
    print(
        f"{b:38s} {ms:9.1f} {100*ms/total_ms:6.1f}% {cat_cnt[b]:6d}  {new_toks:13.0f}"
    )

base_toks = 2048.0 / (total_ms / 1000.0)
print(
    f"\n# baseline (this traced run): {base_toks:.0f} tok/s "
    f"(leaf-sum {total_ms:.1f} ms)"
)
print("# 'tok/s if -10%' = throughput if that category alone were 10% faster.")
