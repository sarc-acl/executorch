"""SARC vs stock logits from logits_probe (raw_real/<gpu>/probe/<model>-<scheme>-<build>-<prompt>.json).

prompt 'real'  = 2048-token real text (tile-aligned: SARC GEMM kernels engaged)
prompt 'check' = 1972-token check prompt (unaligned: 4h4w 8-bit and 4w rows fall back to stock kernels)
For each cell: top-1 of each build, whether they agree, the top1-top2 margin of each build, and the
largest |logit difference| over the union of both top-10 lists (only top-10 logits are recorded).
Writes report/real/probe.csv and prints a table.
"""

import csv
import json
from pathlib import Path

C = Path(__file__).resolve().parent.parent
GPUS = ["780m", "b580", "b70", "4070ti", "orin", "7900xtx", "m51", "s26"]  # contributed GPUs are picked up when their raw dirs exist
rows = []
for g in GPUS:
    d = C / "raw_real" / g / "probe"
    for m in ["1b", "3b", "8b"]:
        for q in ["4w", "8da4w"]:
            for pr in ["real", "check"]:
                try:
                    s = json.load(open(d / f"{m}-{q}-stock-{pr}.json"))
                    t = json.load(open(d / f"{m}-{q}-sarc-{pr}.json"))
                except FileNotFoundError:
                    continue
                ts = {int(i): v for i, v in s["top10"]}
                tt = {int(i): v for i, v in t["top10"]}
                both = set(ts) & set(tt)
                maxd = max((abs(ts[i] - tt[i]) for i in both), default=float("nan"))
                s1, t1 = s["top10"][0], t["top10"][0]
                rows.append({
                    "gpu": g, "model": m, "scheme": q, "prompt": pr, "n_tokens": s["n_tokens"],
                    "stock_top1": s1[0], "sarc_top1": t1[0], "top1_same": s1[0] == t1[0],
                    "stock_margin": round(s["top10"][0][1] - s["top10"][1][1], 4),
                    "sarc_margin": round(t["top10"][0][1] - t["top10"][1][1], 4),
                    "max_abs_dlogit_top10": round(maxd, 4),
                    "bitwise_identical_top10": s["top10"] == t["top10"],
                    "d_bully_minus_otherwise_stock": round(s["requested"]["45647"] - s["requested"]["6062"], 4),
                    "d_bully_minus_otherwise_sarc": round(t["requested"]["45647"] - t["requested"]["6062"], 4),
                })
out = C / "report" / "real"
out.mkdir(parents=True, exist_ok=True)
with open(out / "probe.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("%-7s %-3s %-6s %-5s | %-6s %-6s | %7s %7s | %7s | %s" % ("gpu", "mdl", "scheme", "prmpt", "top1", "same", "mgn_st", "mgn_sa", "max|d|", "identical"))
for r in rows:
    print("%-7s %-3s %-6s %-5s | %-6s %-6s | %7.3f %7.3f | %7.3f | %s" % (
        r["gpu"], r["model"], r["scheme"], r["prompt"], r["stock_top1"], r["top1_same"],
        r["stock_margin"], r["sarc_margin"], r["max_abs_dlogit_top10"], r["bitwise_identical_top10"]))
