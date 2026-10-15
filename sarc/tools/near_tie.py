#!/usr/bin/env python3
"""near_tie.py: the second stage of the gate's next-token comparison (MERGE-PLAN.md M1, owner decision
2026-10-15, OWNER-DECISIONS.md N12). Pure Python, no GPU, no numpy.

When two arms (the old build and the new build, or verify.sh's tiled and default arms) print a different next
token, the flip is accepted only if BOTH hold:
  1. near tie: in the fp32 CPU reference, the top-2 logit margin at that position is below
     NEAR_TIE_MARGIN_LOGIT;
  2. error rule: on every production head configuration (PRODUCTION_CASES) the new arm's rms AND maximum error
     against the fp32 reference are not larger than the old arm's, with 0 mismatches wherever recorded.
Also required: the logits of the two arms reproduce the flip (different top-1). Anything missing fails.
The verdict is TIE_FLIP_ACCEPTED (exit 0) or TIE_FLIP_REJECTED (exit 1), always with the numbers; a caller
reports every accepted flip in its summary, never silently.

  near_tie.py decide --item TEXT --ref LOGITS --old-logits LOGITS --new-logits LOGITS
                     --old-err ERR --new-err ERR [--json-out FILE]
  near_tie.py nexttoken <prompt file> <expected prompt tokens> <old log> <new log> <old rc> <new rc>

LOGITS: a logits_probe .json ({"top10": [[id, logit], ...]}, sarc/tools/logits_probe), a cpu_ref_logits.py or
  e2e cpu_ref.py .json ({"top10": [{"id", "logit"}, ...]}), or a raw float32 vector (.f32, whole vocabulary).
  A missing file counts as missing evidence (rejected), not as an error of the tool.
ERR: the error of the SDPA output against the fp32 CPU reference, per case, in one of these forms:
  - a test_llama_microbench --sdpa-correctness-only log ([sdpa-error] and [sdpa-correctness] lines);
  - <csv>#<arm>: a table of the Intel campaigns' sdpa_error.py (columns case, arm, rms_err, max_abs_err,
    mismatches) or of the 780M's sdpa_error_table.py (columns tier, case, parent_rms, cand_rms, parent_max,
    cand_max; arm parent or cand).
nexttoken: the parent-vs-candidate token comparison of the campaigns' nexttoken.py (4070 Ti campaign), one CSV
  row, verdict SAME / DIFFER / INVALID:<reasons>; two empty or failed outputs are never SAME."""
import argparse, array, csv, hashlib, heapq, json, os, re, sys

# ---- Thresholds. Owner decision 2026-10-15 (OWNER-DECISIONS.md N12, MERGE-PLAN.md M1). Fixed; not options. ----
# Near tie: top-2 logit margin of the fp32 CPU reference at the differing position below this (strictly).
NEAR_TIE_MARGIN_LOGIT = 0.5
# Error criterion: the rule applied so far (D3 item 1): new rms <= old rms AND new max <= old max against the fp32
# reference, on each of the production head configurations (S = 2048, the 1B, 3B and 8B models).
PRODUCTION_CASES = ("1b_head_config_s2048", "3b_head_config_s2048", "8b_head_config_s2048")
DECISION = "owner decision 2026-10-15 (OWNER-DECISIONS.md N12)"


# ---------------------------------------------------------------- logits
def load_logits(path):
    """-> (dict id -> logit, complete vocabulary?) or None when the file does not exist."""
    if not path or not os.path.exists(path):
        return None
    if path.endswith(".f32"):
        a = array.array("f")
        with open(path, "rb") as f:
            a.frombytes(f.read())
        if sys.byteorder != "little":
            a.byteswap()
        return {i: float(v) for i, v in enumerate(a)}, True
    j = json.load(open(path))
    top = j.get("top10") or j.get("top")
    if top is None:
        raise SystemExit(f"{path}: no top10 list")
    vals = {}
    for e in top:
        i, v = (e["id"], e["logit"]) if isinstance(e, dict) else (e[0], e[1])
        vals[int(i)] = float(v)
    return vals, False


def top(vals, n):
    return heapq.nlargest(n, vals.items(), key=lambda kv: (kv[1], -kv[0]))


# ---------------------------------------------------------------- errors against the fp32 reference
# Raw log line formats of test_llama_microbench, in order of preference. A and B are the same fp32 reference
# (A at full printed precision); C is the 780M's fp64 reference from the fp16 inputs (ET_VK_SDPA_ERROR_REPORT=1).
RAW = (
    ("fp32", re.compile(r"\[sdpa-error\] (\S+) max_abs=(\S+) rms=(\S+) max_abs_ref="), lambda m: (m[3], m[2])),
    ("fp32", re.compile(r"\[sdpa-error\] (\S+) elements=\d+ rms_err=(\S+) max_abs_err=(\S+)"), lambda m: (m[2], m[3])),
    ("fp64-of-fp16-inputs", re.compile(r"\[sdpa-error\] (\S+) n=\d+ rms=(\S+) max=(\S+)"), lambda m: (m[2], m[3])),
)
MISMATCH = re.compile(r"\[sdpa-correctness\] (\S+) S=.*mismatches=(\d+)/(\d+)")


def load_errors(spec):
    """-> {case: {format index: (rms, max)}}, {case: mismatches}; None when the file does not exist."""
    path, _, arm = spec.partition("#")
    if not os.path.exists(path):
        return None
    text = open(path, errors="replace").read()
    err, mis = {}, {}
    head = text.split("\n", 1)[0]
    if head.startswith("case,elements,ref_rms,arm,"):  # Intel sdpa_error.py table
        for r in csv.DictReader(l for l in text.splitlines() if "," in l):
            if r.get("arm") == arm and r.get("rms_err") not in (None, "", "missing"):
                err.setdefault(r["case"], {})[1] = (float(r["rms_err"]), float(r["max_abs_err"]))
                if r.get("mismatches", "").isdigit():
                    mis[r["case"]] = int(r["mismatches"])
    elif head.startswith("tier,case,elements,ref_rms,parent_rms,cand_rms"):  # 780M sdpa_error_table.py
        if arm not in ("parent", "cand"):
            raise SystemExit(f"{spec}: arm must be parent or cand")
        for r in csv.DictReader(text.splitlines()):
            err.setdefault(r["case"], {})[2] = (float(r[arm + "_rms"]), float(r[arm + "_max"]))
    elif re.fullmatch(r"[a-z_0-9]+(,[a-z_0-9-]+)+", head.strip()):
        raise SystemExit(f"{spec}: unknown table format")
    else:
        for line in text.splitlines():
            for k, (_, rx, get) in enumerate(RAW):
                m = rx.match(line)
                if m:
                    err.setdefault(m[1], {})[k] = tuple(float(x) for x in get(m))
            m = MISMATCH.match(line)
            if m:  # a log may hold several passes: keep the largest count
                mis[m[1]] = max(mis.get(m[1], 0), int(m[2]))
    return err, mis


# ---------------------------------------------------------------- decide
def decide(item, ref_p, old_p, new_p, old_e, new_e):
    why, out = [], {"item": item, "decision": DECISION, "near_tie_margin_logit": NEAR_TIE_MARGIN_LOGIT,
                    "production_cases": list(PRODUCTION_CASES)}
    ref, old, new = load_logits(ref_p), load_logits(old_p), load_logits(new_p)
    # 1. near tie in the fp32 CPU reference
    if ref is None:
        why.append(f"no fp32 CPU reference logits ({ref_p})")
    else:
        (a, la), (b, lb) = top(ref[0], 2)
        margin = la - lb
        out["reference"] = {"file": ref_p, "top1": [a, la], "top2": [b, lb], "margin": margin,
                            "near_tie": margin < NEAR_TIE_MARGIN_LOGIT}
        if not margin < NEAR_TIE_MARGIN_LOGIT:
            why.append(f"not a near tie: reference top-2 margin {margin:.4f} >= {NEAR_TIE_MARGIN_LOGIT}")
    # the two arms' logits at the position
    for name, x, p in (("old", old, old_p), ("new", new, new_p)):
        if x is None:
            why.append(f"no {name} logits ({p})")
        else:
            (a, la), (b, lb) = top(x[0], 2)
            out[name] = {"file": p, "top1": [a, la], "top2": [b, lb], "margin": la - lb}
            if ref is not None:
                out[name]["top1_in_reference"] = ref[0].get(a)
    if old is not None and new is not None:
        if out["old"]["top1"][0] == out["new"]["top1"][0]:
            why.append(f"the logits do not reproduce the flip (both arms' top-1 is {out['old']['top1'][0]})")
        common = set(old[0]) & set(new[0])
        out["max_abs_logit_diff_on_common_ids"] = max(abs(old[0][i] - new[0][i]) for i in common) if common else None
    # 2. error rule against the fp32 reference
    eo, en = load_errors(old_e), load_errors(new_e)
    if eo is None:
        why.append(f"no error table for the old arm ({old_e})")
    if en is None:
        why.append(f"no error table for the new arm ({new_e})")
    if eo is not None and en is not None:
        rows = []
        for c in PRODUCTION_CASES:
            fo, fn = eo[0].get(c, {}), en[0].get(c, {})
            k = next((k for k in range(len(RAW)) if k in fo and k in fn), None)
            if k is None:
                why.append(f"{c}: not measured in both arms")
                continue
            (ro, mo), (rn, mn) = fo[k], fn[k]
            rok, mok = rn <= ro, mn <= mo
            mis = en[1].get(c)
            rows.append({"case": c, "reference": RAW[k][0], "old_rms": ro, "new_rms": rn, "old_max": mo, "new_max": mn,
                         "rms_not_larger": rok, "max_not_larger": mok, "new_mismatches": mis})
            if not rok:
                why.append(f"{c}: rms error larger than the old arm's ({rn:.4g} > {ro:.4g})")
            if not mok:
                why.append(f"{c}: maximum error larger than the old arm's ({mn:.4g} > {mo:.4g})")
            if mis:
                why.append(f"{c}: {mis} mismatches in the new arm")
        out["errors"] = rows
    out["verdict"] = "TIE_FLIP_REJECTED" if why else "TIE_FLIP_ACCEPTED"
    out["reasons"] = why
    return out


def render(d):
    s = [f"{d['verdict']} {d['item']}"]
    r = d.get("reference")
    if r:
        s.append(f"  reference top-2 margin {r['margin']:.4f} ({'<' if r['near_tie'] else '>='} {d['near_tie_margin_logit']}): "
                 f"id {r['top1'][0]} {r['top1'][1]:.4f} / id {r['top2'][0]} {r['top2'][1]:.4f}")
    for n in ("old", "new"):
        x = d.get(n)
        if x:
            ir = x.get("top1_in_reference")
            s.append(f"  {n} top-1 id {x['top1'][0]} {x['top1'][1]:.4f}, margin {x['margin']:.4f} over id {x['top2'][0]}; "
                     f"in the reference {'%.4f' % ir if ir is not None else 'outside the recorded top-k'}")
    if d.get("max_abs_logit_diff_on_common_ids") is not None:
        s.append(f"  largest |logit difference| old vs new on the recorded ids: {d['max_abs_logit_diff_on_common_ids']:.4f}")
    for e in d.get("errors", []):
        s.append(f"  {e['case']} ({e['reference']} ref): rms {e['old_rms']:.4g} -> {e['new_rms']:.4g} "
                 f"[{'ok' if e['rms_not_larger'] else 'LARGER'}], max {e['old_max']:.4g} -> {e['new_max']:.4g} "
                 f"[{'ok' if e['max_not_larger'] else 'LARGER'}], new mismatches "
                 f"{e['new_mismatches'] if e['new_mismatches'] is not None else 'not recorded'}")
    s += [f"  reason: {w}" for w in d["reasons"]]
    return "\n".join(s)


# ---------------------------------------------------------------- nexttoken (from the 4070 Ti campaign's nexttoken.py)
NT_FIELDS = ["prompt", "prompt_sha256", "expected_tokens", "old_rc", "new_rc", "old_prompt_tokens", "new_prompt_tokens",
             "old_token_hex", "new_token_hex", "old_out_sha256", "new_out_sha256", "verdict"]
RUNTIME = re.compile(rb"PyTorchObserver|^[IWE] |^\[sarc_dev\]")  # the filter of verify.sh's same()


def _one(prompt, want, log, rc):
    why, out, obs = [], b"", None
    try:
        lines = open(log, "rb").read().splitlines(keepends=True)
    except OSError:
        lines = []
    out = b"".join(l for l in lines if not RUNTIME.search(l))
    for l in lines:
        i = l.find(b"PyTorchObserver")
        if i >= 0:
            try:
                obs = json.loads(l[l.index(b"{", i):])
            except ValueError:
                pass
    pt = "" if obs is None else obs.get("prompt_tokens", "")
    if str(rc) != "0": why.append("rc")
    if obs is None: why.append("no_stats")
    elif str(pt) != str(want): why.append("prompt_tokens")
    tok = None
    if not out.strip(): why.append("empty_output")
    elif out.startswith(prompt):
        tok = out[len(prompt):]
        tok = tok[:-1] if tok.endswith(b"\n") else tok
        if not tok: why.append("no_token")
    else: why.append("prompt_not_echoed")
    return pt, tok, hashlib.sha256(out).hexdigest(), why


def nexttoken(prompt_path, want, olog, nlog, orc, nrc):
    prompt = open(prompt_path, "rb").read()
    opt, otok, osha, owhy = _one(prompt, want, olog, orc)
    npt, ntok, nsha, nwhy = _one(prompt, want, nlog, nrc)
    why = [f"old:{w}" for w in owhy] + [f"new:{w}" for w in nwhy]
    v = "INVALID:" + "+".join(why) if why else ("SAME" if osha == nsha and otok == ntok else "DIFFER")
    hx = lambda t: "" if t is None else t.hex()
    return dict(prompt=os.path.basename(prompt_path), prompt_sha256=hashlib.sha256(prompt).hexdigest(), expected_tokens=str(want),
                old_rc=str(orc), new_rc=str(nrc), old_prompt_tokens=str(opt), new_prompt_tokens=str(npt),
                old_token_hex=hx(otok), new_token_hex=hx(ntok), old_out_sha256=osha, new_out_sha256=nsha, verdict=v)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "nexttoken":
        if len(sys.argv) != 8:
            sys.exit(__doc__)
        r = nexttoken(*sys.argv[2:8])
        print(",".join(r[k] for k in NT_FIELDS))
        sys.exit(0 if r["verdict"] == "SAME" else 1)
    ap = argparse.ArgumentParser(usage=__doc__)
    ap.add_argument("cmd", choices=["decide"])
    for a in ("--item", "--ref", "--old-logits", "--new-logits", "--old-err", "--new-err"):
        ap.add_argument(a, required=True)
    ap.add_argument("--json-out")
    a = ap.parse_args()
    d = decide(a.item, a.ref, a.old_logits, a.new_logits, a.old_err, a.new_err)
    print(render(d))
    if a.json_out:
        json.dump(d, open(a.json_out, "w"), indent=1)
    sys.exit(0 if d["verdict"] == "TIE_FLIP_ACCEPTED" else 1)


if __name__ == "__main__":
    main()
