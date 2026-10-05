#!/usr/bin/env python3
"""near_tie.py <evidence dir> <candidate name> <cand env file> <item> [...]: write <evidence dir>/NEAR_TIE.json,
the record gate_check.py --near-tie accepts (owner decision 2026-10-04). The evidence directory must already
hold compare.csv (probe_compare.py, verdict WITHIN twice the noise floor in every cell and metric) and the
position logits of the four arms at each differing position (position/*.json, logits_probe). Refuses otherwise.
<item> names a differing gate item, e.g. "verify.sh unaligned 1b 8da4w: default vs tiled"."""
import glob, hashlib, json, os, sys
d, name, envf = sys.argv[1:4]; items = sys.argv[4:]
cmpf = os.path.join(d, "compare.csv"); lines = open(cmpf).read().splitlines()
if not any(l.startswith("verdict: WITHIN") for l in lines): sys.exit("compare.csv is not WITHIN twice the noise floor: no near-tie acceptance")
pos = sorted(os.path.relpath(f, d) for f in glob.glob(os.path.join(d, "position", "*.json")))
if len(pos) < 4 or not items: sys.exit("needs the position logits of the four arms and at least one differing item")
j = {"decision": "owner decision 2026-10-04", "candidate": name, "cand_env": open(envf).read().split(),
     "cand_env_sha256": hashlib.sha256(open(envf, "rb").read()).hexdigest(), "differing_items": items,
     "verdict": "WITHIN", "rule": "per cell and metric (top-1 differences, mean KL, max KL, max |logit difference|, |ln perplexity ratio|): candidate default vs parent default <= 2 x (parent tiled vs parent default)",
     "compare_csv": "compare.csv", "compare_sha256": hashlib.sha256(open(cmpf, "rb").read()).hexdigest(), "position_logits": pos}
json.dump(j, open(os.path.join(d, "NEAR_TIE.json"), "w"), indent=1); print("written", os.path.join(d, "NEAR_TIE.json"))
