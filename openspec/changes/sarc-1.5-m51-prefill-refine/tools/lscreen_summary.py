#!/usr/bin/env python3
"""lscreen_summary.py <screen csv> [incumbent token=base]: per shape (model, op, M, K, N), each token's kernel time
in every round relative to the incumbent's in the same round, the dispatched kernel, and the screen verdict of
RULES R8: a token is selected for a shape only if it is at least 3 % faster than the incumbent in every round and
dispatched its own kernel (not the incumbent's); among those the fastest median wins; a tie keeps the incumbent."""
import collections, csv, statistics as st, sys
rows = list(csv.DictReader(open(sys.argv[1]))); inc = sys.argv[2] if len(sys.argv) > 2 else "base"
T = collections.defaultdict(dict); K = {}
for r in rows:
    s = (r["model"], r["op"], r["M"], r["K"], r["N"]); T[(s, r["token"])][int(r["round"])] = float(r["kernel_median_us"])
    K[(s, r["token"])] = r["kernel"].replace("_texture3d_texture2d_half", "")
shapes = sorted({s for s, _ in T}); toks = sorted({t for _, t in T})
print("model,op,M,K,N,token,kernel,rounds,median_us,ratio_vs_incumbent_per_round,min_gain_pct,own_kernel,selectable")
pick = {}
for s in shapes:
    base = T.get((s, inc), {})
    best = None
    for t in toks:
        v = T.get((s, t))
        if not v: continue
        rr = [v[k] / base[k] for k in sorted(v) if k in base]
        own = t == inc or K[(s, t)] != K.get((s, inc))
        gain = (1 - max(rr)) * 100 if rr else 0.0
        ok = t != inc and own and rr and len(rr) == len(base) and gain >= 3.0
        print(f'{",".join(s)},{t},{K[(s, t)]},{len(v)},{st.median(v.values()):.1f},{"/".join(f"{x:.3f}" for x in rr)},{gain:+.1f},{"yes" if own else "no"},{"yes" if ok else "no"}')
        if ok and (best is None or st.median(v.values()) < st.median(T[(s, best)].values())): best = t
    pick[s] = best or inc
print("\nselection per shape (R8):")
for s in shapes: print(f'  {" ".join(s)}: {pick[s]}')
