#!/usr/bin/env python3
"""probe_position.py <probe/broad dir> <out dir> <parent default> <parent tiled> <cand default> <cand tiled>

Item 1 of the near-tie decision from the logits_dump files: prompt 0 of the set is the gate's unaligned prompt
(r1304.txt, 1792 tokens, as the runner feeds it), so its row is the position of the `unaligned: default vs
tiled` item. For every cell and arm: the ten largest logits; then, per cell, whether the candidate's two arms
(and the parent's) pick different tokens there. Writes <out dir>/position/<cell>-<arm>.json and summary.csv and
prints the gate items that differ for the candidate (one per line, the wording gate_check.py uses)."""
import csv, json, os, sys
import numpy as np
d, out = sys.argv[1:3]; arms = dict(zip(("parent-default", "parent-tiled", "cand-default", "cand-tiled"), sys.argv[3:7]))
N = sum(1 for _ in open(os.path.join(d, "prompts_meta.csv"))) - 1; os.makedirs(os.path.join(out, "position"), exist_ok=True)
rows = [("cell", "arm", "top1_id", "top1_logit", "top2_id", "top2_logit", "top3_id", "top3_logit", "top1_minus_top2")]; items = []
for m in ("1b", "3b", "8b"):
    for q in ("4w", "8da4w"):
        top = {}
        for a, name in arms.items():
            x = np.fromfile(os.path.join(d, name, f"{m}-{q}.bin"), dtype=np.float32).reshape(N, -1)[0]; i = np.argsort(-x)[:10]
            json.dump({"cell": f"{m}-{q}", "arm": a, "source": name, "prompt": "r1304.txt (prompt 0)", "top10": [[int(k), float(x[k])] for k in i]},
                      open(os.path.join(out, "position", f"{m}-{q}-{a}.json"), "w"))
            top[a] = int(i[0]); rows.append((f"{m}-{q}", a, int(i[0]), f"{x[i[0]]:.6f}", int(i[1]), f"{x[i[1]]:.6f}", int(i[2]), f"{x[i[2]]:.6f}", f"{x[i[0]] - x[i[1]]:.6f}"))
        if m == "1b" and top["cand-default"] != top["cand-tiled"]: items.append(f"verify.sh unaligned 1b {q}: default vs tiled")
csv.writer(open(os.path.join(out, "position", "summary.csv"), "w")).writerows(rows)
print("\n".join(items))
