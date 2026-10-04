#!/usr/bin/env python3
"""gate_check.py: decide a gate step from the contents of its result files, not from exit statuses.

  gate_check.py verify  <cand verify.out> <parent control verify.out>
  gate_check.py sdpa    <sdpa-correctness dir>          (cand-{extended,full}-r1..12.log)
  gate_check.py session <stage/<session>/raw>           (runs.csv, nexttoken.csv)

Prints one line per finding and ends with ACCEPT or REJECT; exit 0 only on ACCEPT.

verify: the candidate must have correctness rc=0, 12 production-diff cases with rc=0 and ALL PASSED, default vs
tiled SAME on prompt_check and on the unaligned prompt for both schemes, decode rc=0, a prefill rate for all
twelve (model, scheme, mode) runs, and every status item (correctness, `linear <scheme> rc`, production-diff,
decode rc and token count, SAME lines) must equal the parent control's. The control must itself be complete.
sdpa: per tier 12 passes, each with the tier's case count (extended 8, full 4), every case PASSED with
mismatches=0, qk_coopmat=yes, av_coopmat=yes, and a [sdpa-kernels] line with pairing=ok per case.
session: six cells with at least 5 valid timed runs per arm, and the next token of parent vs candidate SAME on
both prompts in all six cells."""
import csv, glob, os, re, sys
CELLS = [(m, q) for m in ("1b", "3b", "8b") for q in ("4w", "8da4w")]
bad = []
def fail(msg): bad.append(msg); print("FAIL:", msg)

def verify_items(path):
    it = {}
    for l in open(path, errors="replace"):
        l = l.strip()
        if m := re.match(r"correctness rc=(\d+)", l): it["correctness"] = m[1]
        elif m := re.match(r"linear (\S+) rc=(\d+)", l): it[f"linear {m[1]}"] = m[2]
        elif m := re.match(r"(\S+) (\S+) (tiled|default) prefill_tok_s=(\S*)", l): it[f"prefill {m[1]} {m[2]} {m[3]}"] = "present" if m[4] else "MISSING"
        elif m := re.match(r"(\S+) (\S+) (check|unaligned): default vs tiled output (\S+)", l): it[f"nexttoken {m[1]} {m[2]} {m[3]}"] = m[4]
        elif m := re.match(r'(\S+) (\S+) decode rc=(\d+) "generated_tokens":(\d*)', l): it[f"decode {m[1]} {m[2]}"] = f"rc={m[3]} tokens={m[4]}"
        elif m := re.match(r"pdiff (\S+) (\S+) (\S+) rc=(\d+) ?(.*)", l): it[f"pdiff {m[1]} {m[2]} {m[3]}"] = f"rc={m[4]} " + ("ALL PASSED" if "ALL PASSED" in m[5] else "NOT PASSED")
        elif m := re.match(r"VERIFY_DONE rc=(\d+)", l): it["verify_done"] = m[1]
    return it

def required_verify():
    req = {"correctness": "0", "verify_done": "0"}
    for m, q in CELLS:
        for mode in ("tiled", "default"): req[f"prefill {m} {q} {mode}"] = "present"
    for q in ("4w", "8da4w"):
        for k in ("check", "unaligned"): req[f"nexttoken 1b {q} {k}"] = "SAME"
        for md in ("llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b"):
            for st in ("buffer", "texture3d"): req[f"pdiff {md} {q} {st}"] = "rc=0 ALL PASSED"
    return req

def do_verify(cand, parent):
    c, p = verify_items(cand), verify_items(parent)
    for who, it in (("candidate", c), ("parent control", p)):
        for k, v in required_verify().items():
            if it.get(k) != v: fail(f"{who}: {k} = {it.get(k)!r}, required {v!r}")
        for q in ("4w", "8da4w"):
            if not it.get(f"decode 1b {q}", "").startswith("rc=0 tokens=") or it[f"decode 1b {q}"].endswith("="):
                fail(f"{who}: decode 1b {q} = {it.get(f'decode 1b {q}')!r}")
            if f"linear {q}" not in it: fail(f"{who}: no `linear {q}` line")
    for k in sorted(set(c) | set(p)):
        if c.get(k) != p.get(k): fail(f"differs from the parent control: {k}: candidate {c.get(k)!r}, parent {p.get(k)!r}")
    print(f"verify: {len(c)} candidate items, {len(p)} parent items; linear rc candidate/parent: " +
          ", ".join(f"{q} {c.get('linear ' + q)}/{p.get('linear ' + q)}" for q in ("4w", "8da4w")))

def do_sdpa(d):
    names = set()
    for tier, ncase in (("extended", 8), ("full", 4)):
        logs = sorted(glob.glob(os.path.join(d, f"cand-{tier}-r*.log")))
        if len(logs) != 12: fail(f"{tier}: {len(logs)} pass logs, required 12")
        for f in logs:
            cor = [l for l in open(f, errors="replace") if l.startswith("[sdpa-correctness]") and " S=" in l]
            tot = [l for l in open(f, errors="replace") if l.startswith(f"[sdpa-correctness] tier={tier} ")]
            if len(tot) != 1 or f"cases_run={ncase}" not in tot[0].split(): fail(f"{os.path.basename(f)}: summary line {tot}")
            ker = [l for l in open(f, errors="replace") if l.startswith("[sdpa-kernels]")]
            b = os.path.basename(f)
            if len(cor) != ncase: fail(f"{b}: {len(cor)} cases, required {ncase}")
            if len(ker) != ncase: fail(f"{b}: {len(ker)} [sdpa-kernels] lines, required {ncase}")
            for l in cor:
                if not (l.rstrip().endswith(" PASSED") and re.search(r" mismatches=0/\d+", l) and "qk_coopmat=yes" in l and "av_coopmat=yes" in l):
                    fail(f"{b}: {l.strip()[:160]}")
            for l in ker:
                if not l.rstrip().endswith("pairing=ok"): fail(f"{b}: {l.strip()[:200]}")
                names.update(re.findall(r"(?:qk|softmax|av)=(\S+)", l))
    print("sdpa kernels dispatched:", ", ".join(sorted(names)) or "none")

def do_session(d):
    rows = [r for r in csv.DictReader(open(os.path.join(d, "runs.csv"))) if r["log"].startswith("logs/prefill")]
    for m, q in CELLS:
        for b in ("parent", "cand"):
            n = sum(1 for r in rows if (r["model"], r["scheme"], r["build"]) == (m, q, b) and r["valid"] == "1")
            if n < 5: fail(f"cell {m} {q} {b}: {n} valid timed runs, required 5")
    nt = {}
    p = os.path.join(d, "nexttoken.csv")
    for l in (open(p) if os.path.exists(p) else []):
        f = l.strip().split(",")
        if len(f) >= 4: nt[(f[0], f[1])] = (f[2].split(":")[-1], f[3].split(":")[-1])
    for c in CELLS:
        if nt.get(c) != ("SAME", "SAME"): fail(f"next token parent vs candidate {c[0]} {c[1]}: {nt.get(c)}")
    if not os.path.exists(os.path.join(d, "done.txt")): fail("session did not finish (no done.txt)")
    if any(x for r in rows for x in [r["others"]] if x): fail("a run overlapped a GPU process of another owner")

mode = sys.argv[1]
try:
    {"verify": do_verify, "sdpa": do_sdpa, "session": do_session}[mode](*sys.argv[2:])
except Exception as e:
    fail(f"{mode}: cannot evaluate: {e!r}")
print(f"{mode}: {'REJECT' if bad else 'ACCEPT'} ({len(bad)} findings)")
sys.exit(1 if bad else 0)
