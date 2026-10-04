#!/usr/bin/env python3
"""sdpa_diff.py <dump dir A> <dump dir B>: element-wise difference of the raw fp16 SDPA outputs that
test_llama_microbench --sdpa-correctness-only writes with ET_VK_SDPA_DUMP_DIR (same seeded inputs in both runs).
Per case: elements, elements that differ, max |A - B|, rms, max |A|. Needs numpy (TRACE_PY)."""
import glob, os, sys
import numpy as np
a, b = sys.argv[1], sys.argv[2]
print("case,elements,differing,max_abs_diff,rms_diff,max_abs_a")
for f in sorted(glob.glob(os.path.join(a, "*.bin"))):
    n = os.path.basename(f); g = os.path.join(b, n)
    if not os.path.exists(g): print(f"{n[:-4]},MISSING"); continue
    x = np.fromfile(f, dtype=np.float16).astype(np.float64); y = np.fromfile(g, dtype=np.float16).astype(np.float64)
    if x.size != y.size: print(f"{n[:-4]},SIZE {x.size} vs {y.size}"); continue
    d = np.abs(x - y)
    print(f"{n[:-4]},{x.size},{int((d > 0).sum())},{d.max():.6g},{np.sqrt((d * d).mean()):.6g},{np.abs(x).max():.6g}")
