#!/usr/bin/env python3
"""Aggregate Vulkan delegate dispatches by kernel_name from an ETDump. usage: etdump_kernels.py etdump.in"""
import sys, json, collections
from executorch.devtools import Inspector
insp = Inspector(etdump_path=sys.argv[1])
cnt = collections.Counter(); dur = collections.defaultdict(float); other = collections.Counter()
for blk in insp.event_blocks:
    for ev in blk.events:
        n = ev.name or ""
        try: d = sum(ev.perf_data.raw) if ev.perf_data else 0.0
        except Exception: d = 0.0
        k = None
        if n.startswith("{"):
            try: k = json.loads(n).get("kernel_name")
            except Exception: k = None
        if k is None:
            other[n[:60]] += 1; k = "<other>"
        cnt[k] += 1; dur[k] += d
tot = sum(dur.values())
print("%6s %12s %6s  kernel" % ("count", "time", "pct"))
for k, c in sorted(cnt.items(), key=lambda x: -dur[x[0]]):
    print(f"{c:6d} {dur[k]:12.3f} {100*dur[k]/tot if tot else 0:6.1f}  {k}")
print("total", round(tot, 3), "blocks", [(b.name, len(b.events)) for b in insp.event_blocks])
