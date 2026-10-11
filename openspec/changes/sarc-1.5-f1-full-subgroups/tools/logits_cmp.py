#!/usr/bin/env python3
"""logits_cmp.py <probe/broad> : device side. Old build against new build, byte for byte, per prompt: the files
<arm>/<model>-<scheme>.bin of logits_dump hold 41 vocabulary-sized float32 vectors each (probe_run.sh). Arms
f1-old-<env> / f1-new-<env>, env = fused1, noenv. Prints one line per cell and the totals; writes nothing."""
import hashlib, os, sys
P = sys.argv[1]; N = 41
for env in ("fused1", "noenv"):
    tot = dif = 0
    for m in ("1b", "3b", "8b"):
        for q in ("4w", "8da4w"):
            a, b = (open(f"{P}/f1-{x}-{env}/{m}-{q}.bin", "rb").read() for x in ("old", "new"))
            ok = len(a) == len(b) and len(a) % N == 0 and len(a) > 0; n = len(a) // N if ok else 0
            d = sum(a[i * n:(i + 1) * n] != b[i * n:(i + 1) * n] for i in range(N)) if ok else N
            tot += N; dif += d
            print(f"{env} {m} {q}: {len(a)} / {len(b)} bytes, {N} vectors of {n // 4} float32, {d} differ; sha256 old {hashlib.sha256(a).hexdigest()[:16]} new {hashlib.sha256(b).hexdigest()[:16]}")
    print(f"{env}: {tot} logits vectors of the old build, {dif} differ from the new build's (byte for byte)")
