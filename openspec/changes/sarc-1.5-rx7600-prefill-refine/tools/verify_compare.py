#!/usr/bin/python3
"""verify_compare.py <snapshot verify.out> <candidate verify.out>: line-by-line comparison of two unmodified
verify.sh outputs with the measured rates removed (prefill_tok_s=, decode_tok_s=). Kernel names, return codes,
correctness, production-diff and next-token lines must be identical; every differing line is printed.
Exit 0 if identical, 1 otherwise."""
import re, sys

def norm(p):
    out = []
    for l in open(p):
        l = re.sub(r"(prefill_tok_s|decode_tok_s)=[0-9.]*", r"\1=<rate>", l.rstrip("\n"))
        out.append(l)
    return out

a, b = norm(sys.argv[1]), norm(sys.argv[2])
diff = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
for i, x, y in diff:
    print(f"line {i + 1}:\n  snapshot : {x}\n  candidate: {y}")
if len(a) != len(b):
    print(f"line count differs: snapshot {len(a)}, candidate {len(b)}")
print(f"VERIFY_COMPARE: {'IDENTICAL' if not diff and len(a) == len(b) else 'DIFFERS'} ({len(a)} / {len(b)} lines, {len(diff)} differ)")
sys.exit(0 if not diff and len(a) == len(b) else 1)
