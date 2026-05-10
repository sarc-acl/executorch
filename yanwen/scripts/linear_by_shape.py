#!/usr/bin/env python3
"""Aggregate linear dispatches by output shape, using the canonical analyzer's
event-skipping rules."""
import json
import sys
from collections import defaultdict

from executorch.devtools import Inspector

etdp = sys.argv[1]
insp = Inspector(etdump_path=etdp)

shape_time = defaultdict(float)
shape_count = defaultdict(int)
shape_kernel = defaultdict(set)
total_linear_ms = 0.0

for evlist in insp.event_blocks:
    for ev in evlist.events:
        name = ev.name or ""
        if not name.startswith("{") or "linear" not in name:
            continue
        try:
            obj = json.loads(name)
        except Exception:
            continue
        kernel = obj.get("kernel_name", "?")
        op = obj.get("operator", {}).get("name", "")
        if op != "aten.linear.default":
            continue
        args = obj.get("operator", {}).get("args", [])
        out_shape = None
        for a in args:
            if a.get("type") == "TENSOR" and a.get("sizes"):
                out_shape = tuple(a["sizes"])
        if out_shape is None:
            continue
        dur = ev.perf_data.raw if ev.perf_data else []
        if not dur:
            continue
        avg_ms = sum(dur) / len(dur)
        shape_time[out_shape] += avg_ms
        shape_count[out_shape] += 1
        shape_kernel[out_shape].add(kernel)
        total_linear_ms += avg_ms

print(
    f"Total aten.linear.default time: {total_linear_ms:.2f} ms across "
    f"{sum(shape_count.values())} dispatches\n"
)
print(
    f"{'Output shape':<24} {'#disp':>6} {'sum ms':>10} {'avg ms':>8} {'kernel(s)':<40}"
)
print("-" * 92)
for shp, t in sorted(shape_time.items(), key=lambda kv: -kv[1]):
    c = shape_count[shp]
    avg = t / c
    ks = ",".join(sorted(shape_kernel[shp]))
    pct = 100 * t / total_linear_ms
    print(f"{str(list(shp)):<24} {c:>6} {t:>10.2f} {avg:>8.2f} {ks:<40} {pct:>5.1f}%")
