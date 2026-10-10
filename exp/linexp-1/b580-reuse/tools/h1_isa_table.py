#!/usr/bin/env python3
"""h1_isa_table.py <rows.csv> <dump dir> <out.csv>: per token of <rows.csv> (column `token`) the kernel(s) the run
dispatched (JSON `kernel`), and for every dpas-containing shader of its INTEL_DEBUG=cs dump: SIMD width, GRF
count, instructions, spills : fills, and for the loop with the most dpas: dpas, shared-memory loads and stores.
Uses the study's isa_counts.py (loops/count/fmt). Compile-only evidence; no timing is read."""
import csv, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isa_counts import loops, count, fmt
rows = list(csv.DictReader(open(sys.argv[1]))); D = sys.argv[2]
out = [["token", "a_tiles", "b_tiles", "chains_equiv", "rc", "dispatched_as_itself", "dispatched", "shader", "simd", "grf", "instructions", "spills", "fills",
        "kloop_instructions", "kloop_dpas", "kloop_slm_loads", "kloop_slm_stores", "kloop_barriers", "note"]]
for r in rows:
    t = r["token"]; base = [t, r.get("a_tiles", ""), r.get("b_tiles", ""), r.get("chains_equiv", "")]
    rc = open(f"{D}/{t}.rc").read().strip() if os.path.exists(f"{D}/{t}.rc") else "not run"
    kern = []
    if os.path.exists(f"{D}/{t}.json"):
        try: kern = sorted({c["kernel"] for c in json.load(open(f"{D}/{t}.json"))["cases"]})
        except Exception as e: kern = [f"json error {e}"]
    itself = bool(kern) and all(k.startswith("sarc_" + t + "_") for k in kern)
    txt = open(f"{D}/{t}.cs.log", errors="replace").read().splitlines() if os.path.exists(f"{D}/{t}.cs.log") else []
    heads = [i for i, l in enumerate(txt) if l.startswith("Native code for")]
    seen = set(); n = 0
    for h, e in zip(heads, heads[1:] + [len(txt)]):
        body = txt[h + 2:e]; stop = next((i for i, l in enumerate(body) if l.startswith("NIR (")), len(body)); body = body[:stop]
        if not any("dpas" in l for l in body): continue
        m = re.search(r"SIMD(\d+) shader: (\d+) instructions\. (\d+) loops\..*? (\d+):(\d+) spills:fills.*?GRF registers: (\d+)", txt[h + 1])
        key = (re.search(r"src_hash (0x[0-9a-f]+)", txt[h])[1], m.groups())
        if key in seen: continue
        seen.add(key); n += 1
        best = max(((count(body, a, b)) for a, b in loops(body)), key=lambda c: c["dpas"], default=None)
        f = fmt(best) if best else [""] * 11
        out.append(base + [rc, itself, ";".join(kern), key[0], m[1], m[6], m[2], m[4], m[5], f[0], f[1], f[2], f[3], f[8], ""])
    if n == 0:
        note = "no dpas shader compiled (not dispatched as a cooperative-matrix kernel)" if txt else "no dump"
        fail = next((l.strip()[:160] for l in txt if re.search(r"error|abort|Assert|what\(\)|threw|failed", l, re.I)), "")
        out.append(base + [rc, itself, ";".join(kern), "", "", "", "", "", "", "", "", "", "", "", (note + ("; " + fail if fail else ""))])
csv.writer(open(sys.argv[3], "w", newline="")).writerows(out)
print(f"{len(rows)} tokens, {sum(1 for o in out[1:] if o[7])} dpas shaders")
