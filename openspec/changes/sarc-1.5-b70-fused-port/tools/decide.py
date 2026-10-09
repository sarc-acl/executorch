#!/usr/bin/env python3
"""decide.py <stage/<session>> [--arithmetic <sdpa-error csv of the S=2048 cases> | --bit-identical]

The acceptance decision of a gated candidate under the owner decisions of 2026-10-04 (CAMPAIGN.md). It reads
only files the gate and the probe left behind and never turns a failure into a plain pass.

  GATE_PASS                      gate.txt has no FAIL line: accepted as a plain pass. With --bit-identical the
                                 probe must also show bit-identical logits on every window of every cell.
  ACCEPTED (reference-error rule, owner decision 2026-10-04)
                                 (--arithmetic: the candidate changes kernel arithmetic) every FAIL line of
                                 gate.txt is a next-token item (parent vs candidate, default vs tiled, and the
                                 session-complete line that only names them), AND the three SDPA tiers passed
                                 12 times with 0 mismatches, AND in the given csv the candidate's rms and
                                 maximum error against the fp32 reference are not larger than the parent's on
                                 every S = 2048 head configuration, AND probe/analysis.txt ends PROBE_CHECK_OK
                                 (at least 32 real-text windows per cell, mean KL <= 0.5 nat, top-1 differing
                                 on at most a third of them).
  REJECTED                       anything else, with the reasons.
Writes <session>/decision.txt and prints it; exit status 0 for GATE_PASS or ACCEPTED."""
import csv, os, re, sys
D = sys.argv[1]; args = sys.argv[2:]
ARITH = args[args.index("--arithmetic") + 1] if "--arithmetic" in args else None; BIT = "--bit-identical" in args
def read(*p):
    try: return open(os.path.join(*p), errors="replace").read()
    except OSError: return ""
gate = read(D, "gate.txt").splitlines(); fails = [l for l in gate if l.startswith("FAIL ")]
out = []; why = []
nt = [l for l in fails if re.match(r"FAIL (next token parent vs cand:|verify: default vs tiled,)", l)
      or re.match(r"FAIL timing: session complete : E2E5_INCOMPLETE( \S+:nexttoken_\S+)+\s*$", l)]
other = [l for l in fails if l not in nt]
probe = read(D, "probe", "analysis.txt"); psum = list(csv.DictReader(open(os.path.join(D, "probe", "summary.csv")))) if os.path.exists(os.path.join(D, "probe", "summary.csv")) else []
if not gate or not re.match(r"GATE_(PASS|FAIL)", gate[-1]): verdict = "REJECTED"; why.append("no gate verdict in gate.txt")
elif not fails:
    verdict = "GATE_PASS"
    if BIT:
        notid = [r["cell"] for r in psum if r["bit_identical_C"] != r["windows"]]
        if len(psum) != 6 or notid: verdict = "REJECTED"; why.append(f"claimed bit-identical, but the probe shows otherwise or is incomplete: {notid or 'cells ' + str(len(psum))}")
        else: out.append("probe: logits bit-identical to the parent arm on every window of all six cells")
elif other: verdict = "REJECTED"; why += ["gate failures other than next-token items:"] + other
elif not ARITH: verdict = "REJECTED"; why.append("next-token items differ and the candidate is not declared an arithmetic change (--arithmetic)")
else:
    verdict = "ACCEPTED (reference-error rule, owner decision 2026-10-04)"
    sd = [l for l in gate if l.startswith("PASS sdpa: tier ") and "12 pass(es)" in l]
    if len(sd) != 3: verdict = "REJECTED"; why.append(f"SDPA tiers with 12 clean passes: {len(sd)} of 3")
    rows = list(csv.DictReader(l for l in open(ARITH) if "," in l)); tail = open(ARITH).read().strip().splitlines()[-1]
    prod = {r["case"] for r in rows if re.fullmatch(r"(1b|3b|8b)_head_config_s2048", r["case"]) and r["rms_not_larger_than_first_arm"] == "yes" and r["max_not_larger_than_first_arm"] == "yes" and r["mismatches"] == "0"}
    if tail != "SDPA_ERROR_OK" or len(prod) != 3: verdict = "REJECTED"; why.append(f"error against the fp32 reference: {tail}, S=2048 head configurations not larger than the parent: {sorted(prod)}")
    else: out.append(f"error against the fp32 reference not larger than the parent's on {sorted(prod)} and every other case of {ARITH}")
    if not probe.strip().endswith("PROBE_CHECK_OK") or len(psum) != 6: verdict = "REJECTED"; why.append("probe: " + (probe.strip().splitlines()[-1] if probe.strip() else "missing"))
    else: out.append("probe: " + "; ".join(f'{r["cell"]} top-1 differs on {r["top1_differ_C"]}/{r["windows"]}, mean KL {r["kl_mean_C"]} nat' for r in psum))
    out.append("next-token items that differ (listed, not waived): " + " | ".join(l[5:] for l in nt))
text = "\n".join([verdict] + out + why) + "\n"
open(os.path.join(D, "decision.txt"), "w").write(text); print(text, end="")
sys.exit(0 if verdict != "REJECTED" else 1)
