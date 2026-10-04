#!/usr/bin/env python3
"""gate_check.py <stage/<session>> [--sdpa] [--parent <parent verify.out>] [--control]

Decides a candidate's gate from the files the gate scripts left in the session directory. Prints one
PASS/FAIL line per requirement and exits 0 only if every one passed. Nothing is inferred from a missing file:
missing or empty evidence is a FAIL.

  verify   verify.out of the unmodified sarc/tools/verify.sh: VERIFY_DONE rc=0, no foreign GPU process,
           correctness rc=0, 12 production-diff cases rc=0 ALL PASSED, prefill tok/s present for the six cells
           (default and tiled), default vs tiled SAME on the real-text and the unaligned prompt for both
           schemes, decode rc=0 with generated tokens, and the same status as the parent control line by line
           (--parent; this is where `linear <scheme> rc=` is compared, since its value is device-specific).
  env      the staged cand/env is exactly the ET_VK*/ETVK* environment verify.sh recorded, the one the timing
           session recorded for the candidate arm, and cand-traced/env.
  timing   raw/runs.csv: 5 valid runs per build in each of the six cells, a clock threshold > 0 in force,
           done.txt = E2E5_OK; raw/nexttoken.csv: SAME in all six cells on each of the timed prompt
           prompt_2048.txt, the real-text prompt prompt_check.txt and the staged unaligned prompt r*.txt,
           matched by prompt name.
  trace    trace/trace.ok and one totals row per cell and arm.
  sdpa     (--sdpa) 12 passes each of tiers all, extended and full with the candidate environment: rc 0, the
           expected number of cases PASSED (4 / 8 / 4), none FAILED, every case mismatches=0 and pairing=ok.
  --control checks only the verify block (and one sdpa pass per tier), for the parent control session."""
import csv, os, re, sys
args = sys.argv[1:]
D = args[0]; SDPA = "--sdpa" in args; CONTROL = "--control" in args
PARENT = args[args.index("--parent") + 1] if "--parent" in args else None
CELLS = [(m, q) for m in ("1b", "3b", "8b") for q in ("4w", "8da4w")]
fails = 0
def check(ok, what, detail=""):
    global fails
    fails += not ok
    print(f'{"PASS" if ok else "FAIL"} {what}{" : " + str(detail) if detail else ""}')
def read(*p):
    try: return open(os.path.join(*p), errors="replace").read()
    except OSError: return ""

def verify_status(text):
    """status lines of a verify.out with the measured numbers and kernel names removed"""
    s = {}
    for l in text.splitlines():
        if m := re.match(r"(correctness|linear \S+|VERIFY_DONE) rc=(\d+)", l): s[m[1]] = "rc=" + m[2]
        elif m := re.match(r"(\S+ \S+ (?:check|unaligned)): default vs tiled output (\S+)", l): s[m[1]] = m[2]
        elif m := re.match(r'(\S+ \S+ decode) rc=(\d+) "generated_tokens":(\d+)', l): s[m[1]] = f"rc={m[2]} tokens={m[3]}"
        elif m := re.match(r"(pdiff \S+ \S+ \S+) rc=(\d+)(.*)", l): s[m[1]] = f'rc={m[2]} {"ALL PASSED" if "ALL PASSED" in m[3] else "not passed"}'
        elif m := re.match(r"(\S+ \S+ (?:tiled|default)) prefill_tok_s=(.*)", l): s[m[1]] = "tok_s" if re.fullmatch(r"[0-9.]+", m[2].strip()) else "no tok_s"
    return s

v = read(D, "verify.out"); s = verify_status(v)
check(s.get("VERIFY_DONE") == "rc=0", "verify: completed", s.get("VERIFY_DONE", "no VERIFY_DONE line"))
check(os.path.exists(os.path.join(D, "verify.others")) and not read(D, "verify.others").strip(), "verify: no foreign GPU process", read(D, "verify.others").strip()[:80])
check(s.get("correctness") == "rc=0", "verify: correctness", s.get("correctness", "missing"))
pd = {k: x for k, x in s.items() if k.startswith("pdiff ")}
check(len(pd) == 12 and all(x == "rc=0 ALL PASSED" for x in pd.values()), "verify: production-diff 12 cases", f"{sum(x == 'rc=0 ALL PASSED' for x in pd.values())} of {len(pd)} passed")
for mode in ("default", "tiled"):
    got = [f"{m} {q}" for m, q in CELLS if s.get(f"{m} {q} {mode}") == "tok_s"]
    check(len(got) == 6, f"verify: prefill ran, {mode}, six cells", f"{len(got)} of 6")
for q in ("4w", "8da4w"):
    for k in ("check", "unaligned"):
        check(s.get(f"1b {q} {k}") == "SAME", f"verify: default vs tiled, {k} prompt, {q}", s.get(f"1b {q} {k}", "missing"))
    d = s.get(f"1b {q} decode", "missing")
    check(d.startswith("rc=0 tokens=") and int(d.split("=")[-1]) > 0, f"verify: decode {q}", d)
    check(f"linear {q}" in s, f"verify: linear {q} dispatch report present", s.get(f"linear {q}", "missing"))
if not CONTROL:
    ps = verify_status(read(PARENT)) if PARENT else {}
    diff = sorted(k for k in set(s) | set(ps) if s.get(k) != ps.get(k))
    check(bool(ps) and ps.get("VERIFY_DONE") == "rc=0" and not diff, "verify: same status as the parent control", f"parent {PARENT}: " + (", ".join(f"{k}: parent [{ps.get(k)}] cand [{s.get(k)}]" for k in diff[:6]) if ps else "missing") if (diff or not ps) else "")

    # one candidate configuration: the staged cand/env must be what verify.sh, the timing session and the traces ran
    cenv = [l for l in read(D, "cand", "env").splitlines() if l]
    venv = [l for l in read(D, "verify", "env.txt").splitlines() if re.match(r"ET_?VK\w*=", l) and not l.startswith("ETVK_DEVICE_INDEX=")]
    tenv = re.search(r"^cand env: (.*)$", read(D, "raw", "env.txt"), re.M)
    check(os.path.exists(os.path.join(D, "cand", "env")) and sorted(cenv) == sorted(venv) and tenv is not None and tenv[1].split() == cenv
          and read(D, "cand-traced", "env").splitlines() == read(D, "cand", "env").splitlines(),
          "one candidate environment in verify, timing and traces", f"cand/env {cenv} verify {venv} timing {tenv[1].split() if tenv else 'missing'}")

    try: rows = list(csv.DictReader(open(os.path.join(D, "raw", "runs.csv"))))
    except OSError: rows = []
    for m, q in CELLS:
        n = {b: sum(1 for r in rows if (r["model"], r["scheme"], r["build"]) == (m, q, b) and r["log"].startswith("logs/prefill") and r["valid"] == "1") for b in ("parent", "cand")}
        check(n["parent"] >= 5 and n["cand"] >= 5, f"timing: {m} {q} 5 valid runs per build", f'parent {n["parent"]} cand {n["cand"]}')
    done = read(D, "raw", "done.txt"); env = read(D, "raw", "env.txt")
    check("E2E5_OK" in done, "timing: session complete", done.strip().splitlines()[-1] if done.strip() else "no done.txt")
    ck = re.search(r"clkmin=(\d+)", env)
    check(bool(ck) and int(ck[1]) > 0, "timing: clock threshold in force", ck[0] if ck else "missing")
    nt = {tuple(l.split(",")[:2]): l.strip().split(",")[2:] for l in read(D, "raw", "nexttoken.csv").splitlines() if l.count(",") >= 3}
    unal = sorted(f for f in os.listdir(D) if re.fullmatch(r"r.*\.txt", f))[:1]   # the file verify.sh picks up
    want = ["prompt_2048.txt", "prompt_check.txt"] + (unal or ["<no unaligned prompt r*.txt staged>"])
    for c in CELLS:
        x = dict(y.rsplit(":", 1) for y in nt.get(c, []) if ":" in y)
        check(all(x.get(p) == "SAME" for p in want), f"next token parent vs cand: {c[0]} {c[1]} on {', '.join(want)}",
              ",".join(f"{p}:{x.get(p, 'missing')}" for p in want))

    tot = read(D, "trace", "report", "evidence", "trace", "totals.csv").splitlines()[1:]
    arms = {tuple(l.split(",")[1:4]) for l in tot}
    check(os.path.exists(os.path.join(D, "trace", "trace.ok")) and all((m, q, b) in arms for m, q in CELLS for b in ("parent", "cand")), "trace: warm ETDump analysed for six cells x two arms", f"{len(arms)} of 12")

if SDPA or CONTROL:
    S = os.path.join(D, "sdpa-correctness"); rc = {}
    for l in read(S, "rc.csv").splitlines():
        f = l.strip().split(",")
        if len(f) == 4: rc[tuple(f[:3])] = f[3]
    arm, passes = ("table", 1) if CONTROL else ("cand", 12)
    for tier, cases in (("all", 4), ("extended", 8), ("full", 4)):
        bad = []
        for i in range(1, passes + 1):
            t = read(S, f"{arm}-{tier}-r{i}.log")
            res = [l for l in t.splitlines() if l.startswith("[sdpa-correctness] ") and re.search(r"mismatches=\d+/\d+", l)]
            ker = [l for l in t.splitlines() if l.startswith("[sdpa-kernels] ")]
            why = []
            if rc.get((arm, tier, str(i))) != "0": why.append(f'rc={rc.get((arm, tier, str(i)), "missing")}')
            if sum(l.rstrip().endswith("PASSED") for l in res) != cases or len(res) != cases: why.append(f'{sum(l.rstrip().endswith("PASSED") for l in res)}/{len(res)} passed, want {cases}')
            if any(not re.search(r"mismatches=0/\d+", l) for l in res): why.append("mismatches")
            if "FAILED" in t: why.append("FAILED line")
            if len(ker) != cases or any("pairing=ok" not in l for l in ker): why.append(f'pairing ok in {sum("pairing=ok" in l for l in ker)}/{len(ker)} kernel lines, want {cases}')
            if why: bad.append(f"r{i}: " + "; ".join(why))
        check(not bad, f"sdpa: tier {tier}, {passes} pass(es) x {cases} cases, 0 mismatches, pairing=ok ({arm})", " | ".join(bad[:4]))

print("GATE_PASS" if not fails else f"GATE_FAIL ({fails} requirement(s))")
sys.exit(1 if fails else 0)
