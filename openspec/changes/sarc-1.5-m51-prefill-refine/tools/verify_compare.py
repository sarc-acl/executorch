#!/usr/bin/env python3
"""verify_compare.py <parent verify.out> <candidate verify.out> [supplement]: compares two verify.sh outputs line
by line with the measured rates removed (prefill_tok_s=, decode_tok_s=), as RULES R7 asks. Lines are matched by
their leading key (the text before the first '=' or ':'). Prints SAME / DIFF / ONLY-IN-ONE per line. A third file
(e.g. the parent's pdiff supplement) can stand in for parent lines that verify.sh did not produce, and is named
in the output when it does."""
import re, sys
def norm(l): return re.sub(r"(prefill_tok_s|decode_tok_s)=[0-9.]*", r"\1=<rate>", l.rstrip("\n"))
def key(l):
    m = re.match(r"^(pdiff \S+ \S+ \S+|\S+ \S+ (tiled|default) prefill_tok_s|\S+ \S+ (check|unaligned)|\S+ \S+ decode|linear \S+|correctness|PARTIAL GATE|VERIFY_DONE)", l)
    return m.group(1) if m else l
def load(p): return {key(l): norm(l) for l in open(p) if l.strip()}
P, C = load(sys.argv[1]), load(sys.argv[2]); S = load(sys.argv[3]) if len(sys.argv) > 3 else {}
bad = 0
for k in list(dict.fromkeys(list(P) + list(C))):
    if k == "VERIFY_DONE": continue
    p, c = P.get(k), C.get(k); src = "parent"
    if (p is None or "rc=97" in p or p.endswith("prefill_tok_s=<rate>") and p == norm(k + "=")) and k in S: p, src = S[k], "supplement"
    if p is None or c is None:
        print(f"ONLY-IN-{'CAND' if p is None else 'PARENT'}: {c or p}"); bad += k != "PARTIAL GATE"; continue
    same = p == c
    print(f"{'SAME' if same else 'DIFF'}{'' if src == 'parent' else ' (parent from ' + src + ')'}: {c}" + ("" if same else f"\n      parent: {p}"))
    bad += not same
print(f"verify_compare: {bad} line(s) differ or missing")
