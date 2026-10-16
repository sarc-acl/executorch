#!/usr/bin/env python3
"""sdpa_check.py <sdpa-correctness dir>: the SDPA criteria of sarc-1.5-b580-fused-port/tools/gate_check.py --sdpa:
12 passes each of tiers all, extended and full: rc 0, 4 / 8 / 4 cases PASSED, no FAILED, mismatches=0,
pairing=ok, one [sdpa-error] record per case with finite rms_err, max_abs_err and ref_rms. The base test also
prints a second [sdpa-error] line per case (max_abs= rms= max_abs_ref=); only the rms_err records are counted."""
import math, os, re, sys
S = sys.argv[1]
def read(name):
    try: return open(os.path.join(S, name), errors="replace").read()
    except OSError: return ""
def finite(x):
    try: return math.isfinite(float(x))
    except ValueError: return False
rc = {tuple(f[:3]): f[3] for f in (l.strip().split(",") for l in read("rc.csv").splitlines()) if len(f) == 4}
fails = 0
for tier, cases in (("all", 4), ("extended", 8), ("full", 4)):
    bad = []
    for i in range(1, 13):
        t = read(f"cand-{tier}-r{i}.log")
        res = [l for l in t.splitlines() if l.startswith("[sdpa-correctness] ") and re.search(r"mismatches=\d+/\d+", l)]
        ker = [l for l in t.splitlines() if l.startswith("[sdpa-kernels] ")]
        why = []
        if rc.get(("cand", tier, str(i))) != "0": why.append(f'rc={rc.get(("cand", tier, str(i)), "missing")}')
        passed = sum(l.rstrip().endswith("PASSED") for l in res)
        if passed != cases or len(res) != cases: why.append(f"{passed}/{len(res)} passed, want {cases}")
        if any(not re.search(r"mismatches=0/\d+", l) for l in res): why.append("mismatches")
        if "FAILED" in t: why.append("FAILED line")
        if len(ker) != cases or any("pairing=ok" not in l for l in ker):
            why.append(f'pairing ok in {sum("pairing=ok" in l for l in ker)}/{len(ker)} kernel lines, want {cases}')
        err = [l.split()[3:] for l in t.splitlines() if l.startswith("[sdpa-error] ") and " rms_err=" in l]
        if len(err) != cases or any(len(e) != 3 or not all(finite(x.split("=")[-1]) for x in e) for e in err):
            why.append(f"{len(err)} error records, want {cases} finite")
        if why: bad.append(f"r{i}: " + "; ".join(why))
    fails += bool(bad)
    print(f'{"FAIL" if bad else "PASS"} sdpa tier {tier}: 12 passes x {cases} cases, 0 mismatches, finite error, pairing=ok'
          + (" : " + " | ".join(bad[:4]) if bad else ""))
print("SDPA_PASS" if not fails else f"SDPA_FAIL ({fails} tier(s))")
sys.exit(1 if fails else 0)
