#!/usr/bin/env python3
"""ref_error_rule.py <evidence dir> <candidate name> <cand env file> <parent sdpa-error file> <candidate sdpa-error file> <compare.csv> [<differing item> ...]

Owner decision of 2026-10-04 (second decision, CAMPAIGN.md): a candidate that changes kernel arithmetic is
judged against the fp32 reference, not against the parent. The thresholds are the owner's and are not
parameters of this tool:
  1. on every production-shape case (S = 2048: the 1B, 3B and 8B head configurations of the `full` tier) the
     candidate's rms error AND its maximum error against the fp32 CPU reference are not larger than the
     parent's on the same inputs; every case of both tiers in the candidate file has 0 mismatches;
     (the two files hold the [sdpa-error] and [sdpa-correctness] lines of one extended and one full pass)
  3. gross divergence on the real-text comparison (probe_compare.py output): in no cell may the mean KL of
     candidate-default against parent-default exceed 0.5 nat or the top-1 token differ on more than one third
     of the prompts.
Criterion 1 is applied PER CASE (each head configuration separately), the stricter of the two readings.
Prints every number side by side and a verdict; writes <evidence dir>/REFERENCE_ERROR.json only when both
criteria hold (gate_check.py --reference-error accepts nothing else). The position logits of the four arms
(<evidence dir>/position/*.json) must exist when a differing item is named. Exit 0 = met, 1 = not met."""
import csv, glob, hashlib, json, os, re, sys
d, name, envf, pf, cf, cmpf = sys.argv[1:7]; items = sys.argv[7:]
PROD = ("1b_head_config_s2048", "3b_head_config_s2048", "8b_head_config_s2048")
def errors(path):
    e, mm = {}, {}
    for l in open(path):
        m = re.match(r"\[sdpa-error\] (\S+) max_abs=(\S+) rms=(\S+)", l)
        if m: e[m[1]] = (float(m[3]), float(m[2]))
        m = re.match(r"\[sdpa-correctness\] (\S+) S=.* mismatches=(\d+)/(\d+)", l)
        if m: mm[m[1]] = int(m[2])
    return e, mm
pe, _ = errors(pf); ce, cmm = errors(cf); why = []
print("case,parent_rms,candidate_rms,parent_max,candidate_max,rms_not_larger,max_not_larger")
for c in sorted(set(pe) | set(ce), key=lambda x: (x not in PROD, x)):
    if c not in pe or c not in ce: why.append(f"{c}: missing in one arm"); continue
    r, m = ce[c][0] <= pe[c][0], ce[c][1] <= pe[c][1]
    print(f"{c}{' (production shape)' if c in PROD else ''},{pe[c][0]:.6g},{ce[c][0]:.6g},{pe[c][1]:.6g},{ce[c][1]:.6g},{'yes' if r else 'NO'},{'yes' if m else 'NO'}")
    if c in PROD and not (r and m): why.append(f"criterion 1: {c}: " + ("rms " if not r else "") + ("maximum " if not m else "") + "error larger than the parent's")
for c in PROD:
    if c not in ce: why.append(f"criterion 1: {c}: not measured")
if len(cmm) < 12: why.append(f"criterion 1: {len(cmm)} correctness cases in the candidate file, required 12 (extended + full)")
for c, n in cmm.items():
    if n: why.append(f"criterion 1: {c}: {n} mismatches")
rows = [r for r in csv.reader(open(cmpf)) if len(r) > 4 and r[1].startswith("candidate default vs parent default")]
if len(rows) != 6: why.append(f"criterion 3: {len(rows)} cells in the comparison, required 6")
print("cell,prompts,top1_diff,mean_kl,gross_divergence")
for r in rows:
    n, t, kl = int(r[2]), int(r[3]), float(r[4]); g = kl > 0.5 or 3 * t > n
    print(f"{r[0]},{n},{t},{kl:.6g},{'YES' if g else 'no'}")
    if n < 32: why.append(f"criterion 3: {r[0]}: {n} prompts, required at least 32")
    if g: why.append(f"criterion 3: {r[0]}: mean KL {kl:.4g} nat, top-1 differs on {t} of {n} prompts")
pos = sorted(os.path.relpath(f, d) for f in glob.glob(os.path.join(d, "position", "*.json")))
if items and len(pos) < 4: why.append("criterion 2: position logits of the four arms missing")
if why:
    print("verdict: NOT MET (reference-error rule, owner decision 2026-10-04): " + "; ".join(why)); sys.exit(1)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
j = {"decision": "owner decision 2026-10-04 (second decision): reference-error rule", "candidate": name, "cand_env": open(envf).read().split(),
     "cand_env_sha256": sha(envf), "differing_items": items, "verdict": "MET",
     "rule": "per production-shape case: candidate rms and maximum error against the fp32 reference <= parent's; 0 mismatches in both tiers; no cell with mean KL > 0.5 nat or top-1 differing on more than 1/3 of the prompts",
     "files": {os.path.relpath(p, d): sha(p) for p in (pf, cf, cmpf)}, "position_logits": pos}
json.dump(j, open(os.path.join(d, "REFERENCE_ERROR.json"), "w"), indent=1)
print("verdict: MET (reference-error rule, owner decision 2026-10-04); written " + os.path.join(d, "REFERENCE_ERROR.json"))
