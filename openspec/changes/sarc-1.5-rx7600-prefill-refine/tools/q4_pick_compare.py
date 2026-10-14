#!/usr/bin/env python3
"""q4_pick_compare.py <4w screen csv without profile> [csv out]: per shape, the incumbent (the round-1 pick of rx7600-refine2: the 780m
g28 cbt kernel on the w2 shapes and 8B w1_w3, the 780m g28 bbt kernel on 1B wk_wv and 3B w1_w3, the release table kernel on the others)
against every rx7600-family kernel of the screen: the worst-round ratio incumbent time / candidate time (>= 1.03 passes the 3 %-in-every-round
rule), and a summary line per kernel with the number of shapes that pass and the geometric mean of the worst-round ratios."""
import csv, collections, math, sys
rows = list(csv.DictReader(open(sys.argv[1])))
P780 = "sarc_dev_780m_x_linear_q4gsw_coopmat_"; PRX = "sarc_dev_rx7600_x_linear_q4gsw_coopmat_"
T = collections.defaultdict(dict)   # kernel -> (shape, round) -> us
for r in rows:
    if r["dispatched"] != "1" and r["cand"] != "table": continue
    T[r["cand"]][((r["model"][-2:], r["op"], int(r["N"]), int(r["K"])), r["round"])] = float(r["kernel_median_us"])
def incumbent(s):
    _, _, N, K = s
    if (N, K) in ((14336, 4096), (4096, 14336), (2048, 8192), (3072, 8192)): return P780 + "t256x128k32g28s32f32cbt"
    if (N, K) in ((512, 2048), (8192, 3072)): return P780 + "t256x128k32g28s32f32bbt"
    return "table"
shapes = sorted({s for k in T.values() for (s, _) in k}); rounds = sorted({r for k in T.values() for (_, r) in k})
cands = sorted(k for k in T if k.startswith(PRX))
print("worst-round ratio incumbent / candidate per shape (>= 1.03 passes)")
print("shape".ljust(28) + "incumbent".ljust(30) + " ".join(c[len(PRX) + len("t256x128k32"):].rjust(12) for c in cands))
res = collections.defaultdict(list)
for s in shapes:
    inc = incumbent(s); line = f"{s[0]} {s[1]:6s} N{s[2]:<6d} K{s[3]:<6d}".ljust(28) + inc.replace(P780, "780m_")[:28].ljust(30)
    for c in cands:
        rat = [T[inc][(s, r)] / T[c][(s, r)] for r in rounds if (s, r) in T[c] and (s, r) in T[inc]]
        w = min(rat) if len(rat) == len(rounds) else None
        res[c].append(w); line += (f"{w:12.3f}" if w is not None else " " * 11 + "-")
    print(line)
print("\nkernel".ljust(60) + "shapes passing   geomean of worst-round ratios")
for c in cands:
    ws = [w for w in res[c] if w is not None]
    print(c.replace(PRX, "").ljust(60) + f"{sum(w >= 1.03 for w in ws):2d} / {len(ws):2d}        {math.exp(sum(map(math.log, ws)) / len(ws)):.3f}")
