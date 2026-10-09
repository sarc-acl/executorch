#!/usr/bin/env python3
"""screen_pick2.py <screen.csv> <4w|8da4w> [variant-substring]: the screen rule of R8 against the INCUMBENT OF THE CURRENT PROFILE per shape
(7900xtx-refine5), not against the release table kernel: a variant replaces the incumbent of a shape only if at least 3 % faster in EVERY round (ratio = incumbent
time / variant time, same round, same screen). Only rows with dispatched = 1 count. Prints per shape the incumbent, the best qualifying variant (best worst-round
ratio) or the incumbent, and (stderr) every variant's per-round ratios. The incumbents are the picks of Overrides.cpp (kRefine2 / kRefine4) by (model, layer)."""
import collections, csv, sys
S0, F = sys.argv[1], sys.argv[2]; only = sys.argv[3] if len(sys.argv) > 3 else None
SW = "sarc_linear_dq8ca_coopmat_zpg_sweep_"
DQ = {  # (model, op) -> incumbent kernel of 7900xtx-refine5 ("table" = the release row)
 ("llama-3.1-8b", "w1_w3"): SW + "t256x64k32g24s32", ("llama-3.1-8b", "w2"): "table", ("llama-3.1-8b", "wk_wv"): SW + "t128x64k32g22s32", ("llama-3.1-8b", "wq_wo"): SW + "t128x64k32g22s32",
 ("llama-3.2-1b", "w1_w3"): SW + "t256x64k32g24s32", ("llama-3.2-1b", "w2"): "sarc_dev_780m_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32afmb1", ("llama-3.2-1b", "wk_wv"): SW + "t64x64k32g22s32", ("llama-3.2-1b", "wq_wo"): SW + "t256x64k32g24s32",
 ("llama-3.2-3b", "w1_w3"): SW + "t256x64k32g24s32", ("llama-3.2-3b", "w2"): SW + "t256x64k32g24s32", ("llama-3.2-3b", "wk_wv"): "table", ("llama-3.2-3b", "wq_wo"): SW + "t256x64k32g24s32"}
Q4W = "sarc_linear_q4gsw_coopmat_sweep_t128x128k32g42s32f32cbt"
Q4 = {(m, o): ("table" if m != "llama-3.2-1b" or o == "w1_w3" else Q4W) for m in ("llama-3.1-8b", "llama-3.2-1b", "llama-3.2-3b") for o in ("w1_w3", "w2", "wk_wv", "wq_wo")}
INC = DQ if F == "8da4w" else Q4
t = collections.defaultdict(dict); disp = {}
for r in csv.DictReader(open(S0)):
    if r["kernel_median_us"] in ("", "None"): continue
    k = ((r["model"], r["op"]), r["cand"]); t[k][int(r["round"])] = float(r["kernel_median_us"]); disp[k] = disp.get(k, 1) and int(r["dispatched"])
shapes = sorted({k[0] for k in t}); cands = sorted({k[1] for k in t})
rounds = sorted({r for k in t for r in t[k]})
print("shape,incumbent,pick,worst_round_ratio")
for s in shapes:
    inc = INC[s]; base = t[(s, inc)]
    best = (inc, 1.0)
    for c in cands:
        if c == inc or (only and only not in c) or (s, c) not in t or not disp.get((s, c)): continue
        if any(r not in t[(s, c)] or r not in base for r in rounds): continue
        rat = [base[r] / t[(s, c)][r] for r in rounds]
        print(f"   {s[0]}/{s[1]} {c}: {' '.join(f'{x:.3f}' for x in rat)}", file=sys.stderr)
        if min(rat) >= 1.03 and min(rat) > best[1]: best = (c, min(rat))
    print(f"{s[0]}/{s[1]},{inc.replace(SW, 'sw_')},{best[0]},{best[1]:.3f}")
