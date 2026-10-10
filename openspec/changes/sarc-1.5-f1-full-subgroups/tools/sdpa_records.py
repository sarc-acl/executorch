#!/usr/bin/env python3
"""sdpa_records.py <stage/<session>> <arm of the 12 passes: cand|table>: the SDPA correctness test does not dump
its outputs; what it prints per case is compared instead, as text: the [sdpa-correctness] result lines, the
[sdpa-kernels] lines (dispatched kernels, pairing) and the [sdpa-error] records (rms_err, max_abs_err, ref_rms
against the fp32 CPU reference). Old build against new build, same session (sdpa-old/{old,new}-<tier>.log), and
the old build's records against each of the 12 gate passes of the new build (sdpa-correctness/<arm>-<tier>-r*.log)."""
import glob, os, re, sys
D, arm = sys.argv[1:3]
def rec(p): return [l.rstrip() for l in open(p, errors="replace") if l.startswith(("[sdpa-correctness] ", "[sdpa-kernels] ", "[sdpa-error] "))]
bad = 0
for tier in ("all", "extended", "full", "peaked"):
    o, n = rec(f"{D}/sdpa-old/old-{tier}.log"), rec(f"{D}/sdpa-old/new-{tier}.log")
    passes = sorted(glob.glob(f"{D}/sdpa-correctness/{arm}-{tier}-r*.log"))
    same = sum(rec(p) == o for p in passes)
    err = [l for l in o if l.startswith("[sdpa-error]")]
    mism = sum(int(m) for l in n for m in re.findall(r"mismatches=(\d+)/", l))
    ok = o == n and bool(o) and same == len(passes); bad += not ok
    print(f"tier {tier}: {len(o)} record lines ({len(err)} [sdpa-error] records); old == new: {'IDENTICAL' if o == n and o else 'DIFFERENT'}; "
          f"gate passes of the new build identical to the old build's records: {same} of {len(passes)}; mismatches summed over the new build's cases: {mism}")
    if o != n:
        for a, b in zip(o, n):
            if a != b: print("  old", a); print("  new", b)
print("SDPA_RECORDS_IDENTICAL" if not bad else f"SDPA_RECORDS_DIFFERENT ({bad} tier(s))")
