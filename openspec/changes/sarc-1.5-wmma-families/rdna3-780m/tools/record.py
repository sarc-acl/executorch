#!/usr/bin/env python3
"""record.py <matrix artifact dir> <change dir>: copy each finished session's summary and runs.csv (host name and
home path removed) to <change>/results/matrix/<session>/ and rewrite the matrix table between the markers in
<change>/STATUS.md."""
import json, os, re, sys

A, C = sys.argv[1:3]
order = open(f"{A}/ORDER").read().split()
names = {p["candidate"]: p for p in json.load(open(f"{C}/results/candidate-profiles.json"))["profiles"]}
home = os.path.expanduser("~")


def clean(s):
    return s.replace(home, "~").replace(os.uname().nodename, "host")


def copy(session):
    d = f"{C}/results/matrix/{session}"
    os.makedirs(d, exist_ok=True)
    for name in ("raw/runs.csv", "raw/nexttoken.csv", "summary.json", "verdict.txt", "verify.out", "e2e5.out"):
        p = f"{A}/stage/{session}/{name}"
        if os.path.exists(p):
            open(f"{d}/{os.path.basename(name)}", "w").write(clean(open(p).read()))


cells = ["1b-4w", "1b-8da4w", "3b-4w", "3b-8da4w", "8b-4w", "8b-8da4w"]
lines = ["| candidate | linear kernel | correctness | " + " | ".join(cells) + " | geomean (95 % CI) | margin |",
         "|---" * (len(cells) + 5) + "|"]
for s in ["aa"] + order:
    p = f"{A}/stage/{s}/summary.json"
    v = f"{A}/stage/{s}/verdict.txt"
    if not os.path.exists(p) and not os.path.exists(v):
        continue
    copy(s)
    verdict = open(v).read().strip() if os.path.exists(v) else ""
    kern = "control vs control" if s == "aa" else names[s]["shader"].replace("_texture3d_texture2d_half", "")
    if not os.path.exists(p):
        lines.append(f"| {s} | {kern} | {verdict} |" + " |" * (len(cells) + 2))
        continue
    r = json.load(open(p))
    cr = [f'{r["cells"][c]["r"]:.4f}' if "r" in r["cells"].get(c, {}) else "n/a" for c in cells]
    g = f'{r["geomean"]:.4f} ({r["geomean_ci95"][0]:.4f}-{r["geomean_ci95"][1]:.4f})' if "geomean" in r else "incomplete"
    m = ("PASS" if r.get("margin_cells") and r.get("margin_geomean") else "FAIL") if "geomean" in r else "n/a"
    lines.append(f"| {s} | {kern} | {verdict} | " + " | ".join(cr) + f" | {g} | {m} |")
st = open(f"{C}/STATUS.md").read()
b, e = "<!-- matrix table begin -->", "<!-- matrix table end -->"
st = re.sub(re.escape(b) + ".*" + re.escape(e), b + "\n" + "\n".join(lines) + "\n" + e, st, flags=re.S)
open(f"{C}/STATUS.md", "w").write(st)
