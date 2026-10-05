#!/usr/bin/env python3
"""sdpa_error.py <arm>=<log> [<arm>=<log> ...]: the `[sdpa-error]` lines of test_llama_microbench
--sdpa-correctness-only (rms and maximum absolute error of the SDPA output against the fp32 CPU reference, per
case) for several arms side by side, with the dispatched kernels and the mismatch count of each case.
The first arm is the reference arm of the comparison (the parent): for every other arm the last columns say
whether its rms and maximum error are not larger than the first arm's (owner decision 2026-10-04, second, item 1)."""
import re, sys
arms = [a.split("=", 1) for a in sys.argv[1:]]
data = {}
for name, f in arms:
    t = open(f, errors="replace").read()
    for m in re.finditer(r"\[sdpa-error\] (\S+) elements=(\d+) rms_err=(\S+) max_abs_err=(\S+) ref_rms=(\S+)", t):
        d = data.setdefault(m[1], {}); d[name] = dict(n=int(m[2]), rms=float(m[3]), mx=float(m[4]), ref=float(m[5]))
    for m in re.finditer(r"\[sdpa-kernels\] (\S+) qk=(\S+) softmax=(\S+) av=(\S+)", t):
        data.setdefault(m[1], {}).setdefault(name, {}).update(qk=m[2], sm=m[3], av=m[4])
    for m in re.finditer(r"\[sdpa-correctness\] (\S+) .*mismatches=(\d+)/(\d+)", t):
        data.setdefault(m[1], {}).setdefault(name, {}).update(mis=int(m[2]))
print("case,elements,ref_rms,arm,rms_err,max_abs_err,mismatches,rms_not_larger_than_first_arm,max_not_larger_than_first_arm,qk,softmax,av")
bad = 0; first = arms[0][0]
for case, d in data.items():
    for name, _ in arms:
        x = d.get(name)
        if not x or "rms" not in x: print(f"{case},,,{name},missing"); bad += 1; continue
        if name == first or first not in d or "rms" not in d[first]: r = m_ = "-"
        else:
            r = "yes" if x["rms"] <= d[first]["rms"] else "NO"; m_ = "yes" if x["mx"] <= d[first]["mx"] else "NO"; bad += (r == "NO") + (m_ == "NO")
        print(f'{case},{x["n"]},{x["ref"]:.4e},{name},{x["rms"]:.4e},{x["mx"]:.4e},{x.get("mis", "?")},{r},{m_},{x.get("qk", "?")},{x.get("sm", "?")},{x.get("av", "?")}')
print("SDPA_ERROR_OK" if not bad else f"SDPA_ERROR_LARGER ({bad})")
sys.exit(1 if bad else 0)
