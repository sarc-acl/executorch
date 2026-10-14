#!/usr/bin/env python3
"""pdiff_error_table.py <parent dir> <cand dir> [glob=pdiff-*-8da4w-*.log|*.log]: the reference-error comparison of a
linear candidate on the production-diff shapes (owner decision D3, "for a linear kernel the production-diff
shapes"): per shape, the max abs and max rel error against the sampled reference ([sampled reference] lines of
test_llama_microbench --production-diff) of the parent's and the candidate's logs, the dispatched kernels, and
whether the candidate's errors are not larger than the parent's."""
import glob, os, re, sys
def load(d, pat):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, pat))):
        cur = None
        for l in open(f, errors="replace"):
            m = re.match(r"\[production-diff\] (\S+) \(M=", l)
            if m: cur = m.group(1); continue
            m = re.search(r"max abs diff ([0-9.e+-]+) .*max rel diff ([0-9.e+-]+)", l)
            if m and cur: out.setdefault(cur, {})["err"] = (float(m.group(1)), float(m.group(2)))
            m = re.search(r"\[production-diff\] (\S+) -> (\S+) .*correctness=(\S+)", l)
            if m: out.setdefault(m.group(1), {})["k"] = (m.group(2).replace("_texture3d_texture2d_half", "").replace("_buffer_texture2d_half", ""), m.group(3))
    return out
P, C = load(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 else "*.log"), load(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "*.log")
ok = True
print("case,parent_max_abs,cand_max_abs,parent_max_rel,cand_max_rel,cand_not_larger,parent_kernel,cand_kernel,cand_correctness")
for c in sorted(set(P) & set(C)):
    if "err" not in P[c] or "err" not in C[c]: continue
    (pa, pr), (ca, cr) = P[c]["err"], C[c]["err"]; good = ca <= pa and cr <= pr; ok &= good
    print(f'{c},{pa:.4g},{ca:.4g},{pr:.4g},{cr:.4g},{"yes" if good else "NO"},{P[c].get("k", ("?",))[0]},{C[c].get("k", ("?",))[0]},{C[c].get("k", ("", "?"))[1]}')
print("candidate error not larger than the parent's on every production-diff shape:", "yes" if ok else "NO")
