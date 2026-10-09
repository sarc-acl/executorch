#!/usr/bin/env python3
"""probe_pos.py <probe-timed dir> <model> <scheme> [top=5]: owner decision D1 item 1. For the four arms (parent / cand x default / tiled) of probe_timed.sh:
the top candidates of the next-token logits at the last position, the top-1 / top-2 margin of every arm, and the logit of the parent-default top-2 tokens in every arm
(how far the candidate moved the pair)."""
import struct, sys, os
d, m, q = sys.argv[1:4]; top = int(sys.argv[4]) if len(sys.argv) > 4 else 5
arms = ["parent-default", "parent-tiled", "cand-default", "cand-tiled"]; L = {}
for a in arms:
    b = open(os.path.join(d, f"{m}-{q}-{a}.bin"), "rb").read(); L[a] = struct.unpack("<%df" % (len(b) // 4), b)
V = len(L[arms[0]])
def tops(l): return sorted(range(V), key=lambda i: -l[i])[:top]
print(f"vocab {V}")
for a in arms:
    t = tops(L[a]); print(f"{a}: top{top} " + " ".join(f"{i}:{L[a][i]:.4f}" for i in t) + f" | margin top1-top2 {L[a][t[0]] - L[a][t[1]]:.4f}")
p = tops(L["parent-default"]); c = tops(L["cand-default"])
pair = sorted(set(p[:2]) | set(c[:2]))
print("tokens of interest (top-2 of parent-default and of cand-default):", pair)
for a in arms: print(f"{a}: " + " ".join(f"{i}:{L[a][i]:.4f}" for i in pair))
for x, y in (("cand-default", "parent-default"), ("cand-tiled", "parent-tiled"), ("parent-tiled", "parent-default"), ("cand-tiled", "cand-default")):
    print(f"max |logit diff| {x} vs {y}: {max(abs(u - v) for u, v in zip(L[x], L[y])):.4f}; top-1 {'same' if tops(L[x])[0] == tops(L[y])[0] else 'DIFFERENT'}")
