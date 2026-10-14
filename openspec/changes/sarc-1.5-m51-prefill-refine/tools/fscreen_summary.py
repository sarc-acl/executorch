#!/usr/bin/env python3
"""fscreen_summary.py <dir> <rounds>: per fused variant, the ETDump time of the fused kernel in every round (families.csv of
fscreen.sh, family 'attention: fused'), the ratio to the profile's variant of the same head_dim, and the R8 verdict: selected
only when faster than the profile's variant by the kernel-screen margin of RULES R8 in every round (MARGIN below)."""
import csv, os, sys
MARGIN = 0.97  # ratio to the incumbent that the R8 margin allows, in every round
d, R = sys.argv[1], int(sys.argv[2])
t = {}
for r in range(1, R + 1):
    p = os.path.join(d, f"round{r}", "families.csv")
    if not os.path.exists(p): continue
    for row in csv.DictReader(open(p)):
        if row["family"] == "attention: fused" and row["arm"].startswith("d"):
            t.setdefault((row["cell"], row["arm"]), {})[r] = float(row["ms"])
print("cell,variant,ms_per_round,ratio_to_incumbent_per_round,selected")
for hd, inc in (("d64", "d64_t32x32g11s32rk"), ("d128", "d128_t16x64g11s32rk")):
    for (cell, arm), v in sorted(t.items()):
        if not arm.startswith(hd + "_"): continue
        b = t.get((cell, inc), {})
        rr = [v[r] / b[r] for r in sorted(v) if r in b and b[r] > 0]
        sel = arm != inc and len(rr) == R and all(x <= MARGIN for x in rr)
        print(f"{cell},{arm},{'/'.join('%.1f' % v[r] for r in sorted(v))},{'/'.join('%.3f' % x for x in rr)},{'SELECTED' if sel else 'keep incumbent' if arm != inc else 'incumbent'}")
