#!/usr/bin/env python3
"""probe_analysis.py <stage/<session>/probe>: the logits evidence of tools/probe.sh, per cell.

Arms: P parent-default, PT parent-tiled, C candidate-default, CT candidate-tiled (gate prompts only).
For the real-text windows (names w*) it reports, for C against P and beside it for PT against P (the two arms
of the parent that are both accepted as correct on this device):
  top-1 differences, mean and maximum KL(P || X) of the next-token distribution in nat, the maximum absolute
  logit difference, and the perplexity of the known next token (windows that have one) with its ratio to P;
  bit-identical logit vectors are counted too.
Gross-divergence check (owner decision 2026-10-04, second; thresholds fixed there, not here): a cell FAILS if
the mean KL of C against P exceeds 0.5 nat or the top-1 token differs on more than one third of the windows.
For the gate prompts (names gate-*) it prints the top logits of every arm, the parent's own top-2 margin and
how far the candidate moved it.

Writes summary.csv, per_prompt.csv and differing.md next to the data; exit status 1 if any cell fails the
gross-divergence check or an expected file is missing."""
import csv, glob, json, os, sys
import numpy as np

D = sys.argv[1]
KL_MAX, TOP1_MAX = 0.5, 1.0 / 3.0   # owner decision 2026-10-04 (second), item 3
prompts = [l.split()[0] for l in open(os.path.join(D, "prompts.txt")) if l.strip()]
wins = [p for p in prompts if p.startswith("w")]; gates = [p for p in prompts if p.startswith("gate-")]

def load(cell, arm, p):
    f = os.path.join(D, cell, arm, p)
    if not os.path.exists(f + ".f32"): return None
    return np.fromfile(f + ".f32", dtype=np.float32), json.load(open(f + ".json"))

def logsoftmax(x):
    x = x.astype(np.float64); x = x - x.max()
    return x - np.log(np.exp(x).sum())

def kl(a, b):   # KL(a || b) from logits, nat
    la, lb = logsoftmax(a), logsoftmax(b)
    return float((np.exp(la) * (la - lb)).sum())

cells = sorted(os.path.basename(c) for c in glob.glob(os.path.join(D, "*-*")) if os.path.isdir(c))
bad = 0; rows = []; per = []; md = ["# Logits at the gate prompts and at every window where the candidate's top-1 differs from the parent's", ""]
for cell in cells:
    st = {}
    for other in ("C", "PT"):
        n = miss = d1 = ident = 0; kls = []; mx = 0.0; nllp = []; nllo = []
        for p in wins:
            a, b = load(cell, "P", p), load(cell, other, p)
            if a is None or b is None: miss += 1; continue
            (la, ja), (lb, _) = a, b; n += 1
            t = int(la.argmax()) != int(lb.argmax()); d1 += t
            ident += bool(np.array_equal(la, lb))
            k = kl(la, lb); kls.append(k); m = float(np.abs(la.astype(np.float64) - lb).max()); mx = max(mx, m)
            if ja["next_id"] >= 0:
                nllp.append(-logsoftmax(la)[ja["next_id"]]); nllo.append(-logsoftmax(lb)[ja["next_id"]])
            per.append([cell, other, p, int(la.argmax()), int(lb.argmax()), int(t), f"{k:.6g}", f"{m:.6g}", ja["next_id"]])
        st[other] = dict(n=n, miss=miss, d1=d1, ident=ident, klmean=float(np.mean(kls)) if kls else float("nan"), klmax=max(kls) if kls else float("nan"),
                         mx=mx, pplp=float(np.exp(np.mean(nllp))) if nllp else float("nan"), pplo=float(np.exp(np.mean(nllo))) if nllo else float("nan"), nppl=len(nllp))
    c, t = st["C"], st["PT"]
    ok = c["miss"] == 0 and c["n"] >= 32 and c["klmean"] <= KL_MAX and c["d1"] <= TOP1_MAX * c["n"]
    bad += not ok
    rows.append([cell, c["n"], c["d1"], f'{c["klmean"]:.3e}', f'{c["klmax"]:.3e}', f'{c["mx"]:.4f}', c["ident"], f'{c["pplp"]:.4f}', f'{c["pplo"]:.4f}', f'{c["pplo"] / c["pplp"]:.5f}', c["nppl"],
                 t["n"], t["d1"], f'{t["klmean"]:.3e}', f'{t["klmax"]:.3e}', f'{t["mx"]:.4f}', t["ident"], f'{t["pplo"] / t["pplp"]:.5f}', "ok" if ok else "FAIL"])
    # the gate prompts, and any window where C and P disagree
    show = gates + [r[2] for r in per if r[0] == cell and r[1] == "C" and r[5] == 1]
    for p in show:
        arms = {a: load(cell, a, p) for a in ("PT", "P", "CT", "C")}
        if arms["P"] is None or arms["C"] is None: md.append(f"## {cell} {p}: missing data"); bad += p in gates; continue
        lp = arms["P"][0]; top = list(dict.fromkeys([int(i) for a in ("P", "C", "PT", "CT") if arms[a] is not None for i in np.argsort(-arms[a][0])[:3]]))
        o = np.argsort(-lp); margin = float(lp[o[0]] - lp[o[1]])
        lc = arms["C"][0]; cm = float(lc[o[0]] - lc[o[1]])
        same = int(lp.argmax()) == int(lc.argmax())
        md += [f"## {cell} {p}: parent top-1 {int(o[0])}, candidate top-1 {int(lc.argmax())} ({'SAME' if same else 'DIFFER'})", "",
               f"Parent-default top-2 margin (logit of id {int(o[0])} minus id {int(o[1])}): {margin:+.4f}; the same difference in candidate-default: {cm:+.4f} (moved by {cm - margin:+.4f}). "
               f"KL(P || C) = {kl(lp, lc):.3e} nat, max |logit difference| = {float(np.abs(lp.astype(np.float64) - lc).max()):.4f}.", "",
               "| token id | " + " | ".join(f"{a} ({n})" for a, n in (("PT", "parent tiled"), ("P", "parent default"), ("CT", "candidate tiled"), ("C", "candidate default"))) + " |", "|---|---:|---:|---:|---:|"]
        for i in top:
            md.append(f"| {i} | " + " | ".join(f"{arms[a][0][i]:.4f}" if arms[a] is not None else "not run" for a in ("PT", "P", "CT", "C")) + " |")
        md.append("")
hdr = ["cell", "windows", "top1_differ_C", "kl_mean_C", "kl_max_C", "max_logit_diff_C", "bit_identical_C", "ppl_P", "ppl_C", "ppl_ratio_C", "ppl_windows",
       "windows_PT", "top1_differ_PT", "kl_mean_PT", "kl_max_PT", "max_logit_diff_PT", "bit_identical_PT", "ppl_ratio_PT", "gross_divergence_check"]
with open(os.path.join(D, "summary.csv"), "w", newline="") as f: csv.writer(f).writerows([hdr] + rows)
with open(os.path.join(D, "per_prompt.csv"), "w", newline="") as f:
    csv.writer(f).writerows([["cell", "arm_vs_P", "prompt", "top1_P", "top1_arm", "differ", "kl_nat", "max_logit_diff", "next_id"]] + per)
open(os.path.join(D, "differing.md"), "w").write("\n".join(md) + "\n")
print(f"thresholds (fixed by the owner decision): mean KL(P || C) <= {KL_MAX} nat and top-1 differing on at most 1/3 of the windows, per cell; at least 32 windows")
print(",".join(hdr))
for r in rows: print(",".join(str(x) for x in r))
print("PROBE_CHECK_OK" if not bad and len(cells) else f"PROBE_CHECK_FAIL ({bad})")
sys.exit(1 if bad or not cells else 0)
