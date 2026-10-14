#!/usr/bin/env python3
"""screen_ratio.py <screen csv> <incumbent kernel base> [--time]: per kernel, per round, the ratio incumbent_time / kernel_time
(> 1 = faster) summarised over the twelve shapes: geometric mean per round, and the number of shapes with at least 1.03 in every
round. With --time also the summed kernel time of the twelve shapes per kernel (us, median over rounds). Rows of a kernel that was
not dispatched on a shape (dispatched=0) are ignored with a note."""
import csv, math, statistics as st, sys
rows = list(csv.DictReader(open(sys.argv[1]))); inc = sys.argv[2]; show_time = "--time" in sys.argv
t = {}
for r in rows:
    if r["dispatched"] != "1": continue
    t[(r["cand"], r["round"], (r["model"], r["N"], r["K"]))] = float(r["kernel_median_us"])
kernels = sorted({k for k, _, _ in t}, key=lambda k: (k != inc, k)); rounds = sorted({r for _, r, _ in t})
shapes = sorted({s for _, _, s in t})
print(f"{'kernel':78s} " + " ".join(f"gm_r{r}" for r in rounds) + "  shapes>=1.03(all rounds)" + ("  sum_us(med)" if show_time else ""))
for k in kernels:
    gms = []; ok = 0; n = 0
    for r in rounds:
        rat = [t[(inc, r, s)] / t[(k, r, s)] for s in shapes if (k, r, s) in t and (inc, r, s) in t]
        gms.append(math.exp(sum(map(math.log, rat)) / len(rat)) if rat else float("nan"))
    for s in shapes:
        rr = [t[(inc, r, s)] / t[(k, r, s)] for r in rounds if (k, r, s) in t and (inc, r, s) in t]
        if len(rr) == len(rounds):
            n += 1; ok += all(x >= 1.03 for x in rr)
    line = f"{k:78s} " + " ".join(f"{g:6.3f}" for g in gms) + f"  {ok}/{n}"
    if show_time:
        line += "  %10.0f" % st.median([sum(t[(k, r, s)] for s in shapes if (k, r, s) in t) for r in rounds])
    print(line)
