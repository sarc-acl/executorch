#!/usr/bin/env python3
"""q4_pick_compare.py <4w screen csv without profile>: per shape, the incumbent (the round-1 pick of rx7600-refine2: 780m g28 cbt on the w2
shapes and 8B w1_w3, 780m g28 bbt on 1B wk_wv and 3B w1_w3, the table kernel on the others) against every candidate kernel: time ratio
(incumbent / candidate) per round, the worst round, and whether the candidate passes the 3 %-in-every-round rule on that shape."""
import csv, collections, sys
rows = list(csv.DictReader(open(sys.argv[1])))
T = collections.defaultdict(dict)   # kernel -> (shape, round) -> us
for r in rows:
    if r["dispatched"] != "1" and r["cand"] != "table": continue
    T[r["cand"].split("coopmat_", 1)[-1] if r["cand"] != "table" else "table"][((r["model"][-2:], r["op"], int(r["N"]), int(r["K"])), r["round"])] = float(r["kernel_median_us"])
def pick(s):
    m, op, N, K = s
    if (N, K) in ((14336, 4096), (4096, 14336), (2048, 8192), (3072, 8192)): return "t256x128k32g28s32f32cbt"            # (780m family; same key without the prefix)
    if (N, K) in ((512, 2048), (8192, 3072)): return "t256x128k32g28s32f32bbt"
    return "table"
shapes = sorted({s for k in T.values() for (s, _) in k})
rounds = sorted({r for k in T.values() for (_, r) in k})
cands = [k for k in T if k not in ("table",) and not k.startswith("t256x128k32g28s32f32cbt") or k.startswith("t256x128k32g2") and ("bp" in k)]
names = sorted({k for k in T if "bp" in k or k.startswith("t256x128k32g2") and k not in ("t256x128k32g28s32f32cbt", "t256x128k32g28s32f32bbt")})
# kernels of the two families share names after the prefix is cut; keep the rx7600 ones by the bp suffix and the copies by an explicit list
print("incumbent per shape, then the worst-round ratio of each candidate (>= 1.03 passes)")
print("shape".ljust(26) + "incumbent".ljust(26) + " ".join(n[-18:].rjust(18) for n in names))
for s in shapes:
    inc = pick(s); line = f"{s[0]} {s[1]:6s} N{s[2]:<6d} K{s[3]:<6d}".ljust(26) + inc.ljust(26)
    for n in names:
        rat = [T[inc][(s, r)] / T[n][(s, r)] for r in rounds if (s, r) in T[n] and (s, r) in T[inc]]
        line += (f"{min(rat):18.3f}" if len(rat) == len(rounds) else " " * 17 + "-")
    print(line)
