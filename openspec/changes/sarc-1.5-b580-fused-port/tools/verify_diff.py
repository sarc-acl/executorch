#!/usr/bin/env python3
"""verify_diff.py <verify.out A> <verify.out B>: the two outputs of sarc/tools/verify.sh line by line with the
rates removed (the values after prefill_tok_s= and decode_tok_s=). Everything else is compared as printed,
kernel names included. Prints the differing lines and VERIFY_SAME or VERIFY_DIFFERENT (n); exit 0 only if same."""
import re, sys
def lines(p):
    return [re.sub(r"(tok_s=)[0-9.eE+-]+", r"\1#", l.rstrip()) for l in open(p, errors="replace")]
a, b = lines(sys.argv[1]), lines(sys.argv[2])
n = 0
for i in range(max(len(a), len(b))):
    x = a[i] if i < len(a) else "<missing>"; y = b[i] if i < len(b) else "<missing>"
    if x != y: n += 1; print(f"line {i + 1}:\n  A {x}\n  B {y}")
ok = n == 0 and any(l.startswith("VERIFY_DONE rc=0") for l in a)
print(f"VERIFY_SAME ({len(a)} lines)" if ok else f"VERIFY_DIFFERENT ({n} line(s))")
sys.exit(0 if ok else 1)
