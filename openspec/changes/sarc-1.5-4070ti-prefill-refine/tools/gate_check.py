#!/usr/bin/env python3
"""gate_check.py: decide a gate step from the contents of its result files, not from exit statuses.

  gate_check.py verify  <cand verify.out> <parent control verify.out>
  gate_check.py sdpa    <sdpa-correctness dir> [<cand env file>]   (cand-{extended,full}-r1..12.log)
  gate_check.py session <stage/<session>/raw> (--clkmin <clkmin.json> | --calibration) [--require-logs]
  gate_check.py env     <stage/<session>>                          (one candidate environment everywhere)

Prints one line per finding and ends with ACCEPT or REJECT; exit 0 only on ACCEPT.

verify: the candidate must have correctness rc=0, 12 production-diff cases with rc=0 and ALL PASSED, default vs
tiled SAME on prompt_check and on the unaligned prompt for both schemes, decode rc=0, a prefill rate for all
twelve (model, scheme, mode) runs, and every status item (correctness, `linear <scheme> rc`, production-diff,
decode rc and token count, SAME lines) must equal the parent control's. The control must itself be complete.
sdpa: per tier 12 passes, each with the tier's case count (extended 8, full 4), every case PASSED with
mismatches=0, qk_coopmat=yes, av_coopmat=yes, and a [sdpa-kernels] line with pairing=ok per case; with an env
file that names a profile, every pass must carry that profile's banner.
session: six cells with at least 5 valid timed runs per arm; every timed run judged against the calibrated
clock threshold of its cell (--clkmin; a record-only session is rejected unless --calibration says it is the
baseline or A/A session, which can never accept a candidate); and, per cell, the next token of parent vs
candidate on the timed prompt, prompt_real_2048.txt (2048 tokens), prompt_check.txt (1972) and r1304.txt (1792).
A next-token row counts only if it is SAME and its evidence holds: the prompt hash is the tracked file's, both
runs are in runs.csv with rc 0 and the expected prompt tokens and no foreign GPU process, both output hashes
are equal and (except for the timed prompt) both tokens are non-empty and equal. With the logs present
(--require-logs makes that mandatory, as in the gates) every row is recomputed from the logs with nexttoken.py.
env: cand/env equals cand-traced/env and parent/env equals parent-traced/env; the ET_VK variables verify.sh
recorded are exactly cand/env; candidate logs carry the profile banner of cand/env and parent logs none."""
import collections, csv, glob, hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nexttoken
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

def do_sdpa(d, envfile=None):
    names = set()
    if envfile: banner_check(sorted(glob.glob(os.path.join(d, "cand-*.log"))), env_lines(envfile), "sdpa")
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

from gate_check_paths import PROMPTS
def sha(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()

def do_session(d, *opts):
    opts = list(opts); clk = None; calibration = "--calibration" in opts; require_logs = "--require-logs" in opts
    if "--clkmin" in opts: clk = json.load(open(opts[opts.index("--clkmin") + 1]))["cells"]
    if (clk is None) == (not calibration): return fail("session needs exactly one of --clkmin <json> and --calibration")
    allrows = list(csv.DictReader(open(os.path.join(d, "runs.csv"))))
    rows = [r for r in allrows if r["log"].startswith("logs/prefill")]
    bylog = {r["log"]: r for r in allrows}
    for m, q in CELLS:
        for b in ("parent", "cand"):
            n = sum(1 for r in rows if (r["model"], r["scheme"], r["build"]) == (m, q, b) and r["valid"] == "1")
            if n < 5: fail(f"cell {m} {q} {b}: {n} valid timed runs, required 5")
    for r in rows:
        applied = r.get("clkmin") or ""
        if calibration:
            if applied != "0": fail(f'{r["log"]}: clkmin {applied!r} in a calibration session, expected 0')
        else:
            want = str(clk.get(f'{r["model"]}-{r["scheme"]}', {}).get("clkmin_mhz", ""))
            if not want.isdigit() or int(want) <= 0: fail(f'no calibrated clock threshold for {r["model"]} {r["scheme"]}')
            elif applied != want: fail(f'{r["log"]}: clock threshold applied {applied!r}, calibrated {want} (record-only or stale)')
            elif r["valid"] == "1" and not (r["clk_med_mhz"] and float(r["clk_med_mhz"]) >= int(want)): fail(f'{r["log"]}: valid but clock {r["clk_med_mhz"]} < {want}')
    if any(r["others"] for r in allrows): fail("a run overlapped a GPU process of another owner")
    nt = collections.defaultdict(dict)
    p = os.path.join(d, "nexttoken.csv")
    for r in (csv.DictReader(open(p)) if os.path.exists(p) else []): nt[(r["model"], r["scheme"])][r["prompt"]] = r
    have_logs = os.path.isdir(os.path.join(d, "logs"))
    if require_logs and not have_logs: fail("no logs directory: next-token rows cannot be recomputed")
    for m, q in CELLS:
        for prompt, (tag, want, tracked) in PROMPTS.items():
            r = nt[(m, q)].get(prompt); w = f"next token {m} {q} {prompt}"
            if r is None: fail(f"{w}: no row"); continue
            if r["verdict"] != "SAME": fail(f'{w}: {r["verdict"]}'); continue
            if r["prompt_sha256"] != sha(tracked): fail(f"{w}: prompt hash is not the tracked file's")
            if r["expected_tokens"] != want: fail(f'{w}: expected_tokens {r["expected_tokens"]}, required {want}')
            timed = tag == "prefill"; logs = {}
            for b in ("parent", "cand"):
                log = f"logs/{tag}-{m}-{q}-{b}-r{1 if timed else 0}.log"; logs[b] = log; run = bylog.get(log)
                if run is None: fail(f"{w}: {log} is not in runs.csv"); continue
                if run["rc"] != "0" or r[f"{b}_rc"] != "0": fail(f'{w}: {b} rc {run["rc"]} (row says {r[f"{b}_rc"]})')
                if run["prompt_tokens"] != want or r[f"{b}_prompt_tokens"] != want: fail(f'{w}: {b} prompt tokens {run["prompt_tokens"]!r}, required {want}')
                if not timed and not r[f"{b}_token_hex"]: fail(f"{w}: {b} produced no token")
            if r["parent_out_sha256"] != r["cand_out_sha256"] or r["parent_token_hex"] != r["cand_token_hex"]: fail(f"{w}: marked SAME but outputs differ")
            if r["parent_out_sha256"] == hashlib.sha256(b"").hexdigest(): fail(f"{w}: empty output")
            if have_logs and len(logs) == 2:
                re_ = nexttoken.evaluate(tracked, want, os.path.join(d, logs["parent"]), os.path.join(d, logs["cand"]), r["parent_rc"], r["cand_rc"], timed)
                diff = [k for k in re_ if k != "prompt" and re_[k] != r[k]]
                if diff: fail(f"{w}: recomputed from the logs differs in {diff} (recomputed verdict {re_['verdict']})")
    if not os.path.exists(os.path.join(d, "done.txt")): fail("session did not finish (no done.txt)")
    if calibration: print("calibration session (record-only clock): for the baseline, A/A and calibrate_clock.py only, never a candidate gate")

def env_lines(path): return sorted(l.strip() for l in open(path) if l.strip())
def profile_of(lines):
    for l in lines:
        if l.startswith("ET_VK_SARC_DEV_PROFILE="): return l.split("=", 1)[1]
    return None
def banner_check(logs, lines, who):
    prof = profile_of(lines)
    for f in logs:
        got = set(re.findall(r"\[sarc_dev\] profile active: (\S+)", open(f, errors="replace").read()))
        if got != ({prof} if prof else set()): fail(f"{who} {os.path.basename(f)}: profile banner {sorted(got)}, environment says {prof}")

def do_env(stage):
    e = {b: env_lines(os.path.join(stage, b, "env")) for b in ("parent", "cand", "parent-traced", "cand-traced")}
    if e["cand"] != e["cand-traced"]: fail(f'cand/env {e["cand"]} != cand-traced/env {e["cand-traced"]}')
    if e["parent"] != e["parent-traced"]: fail(f'parent/env {e["parent"]} != parent-traced/env {e["parent-traced"]}')
    v = os.path.join(stage, "verify", "env.txt")
    if os.path.exists(v):
        got = sorted(l.strip() for l in open(v) if re.match(r"ET_VK_", l))
        if got != e["cand"]: fail(f"verify.sh ran with {got}, cand/env is {e['cand']}")
    else: fail("no verify/env.txt")
    banner_check(sorted(glob.glob(os.path.join(stage, "raw/logs/*-cand-r*.log"))), e["cand"], "session")
    banner_check(sorted(glob.glob(os.path.join(stage, "raw/logs/*-parent-r*.log"))), e["parent"], "session")
    banner_check(sorted(glob.glob(os.path.join(stage, "sdpa-correctness/cand-*.log"))), e["cand"], "sdpa")
    banner_check(sorted(glob.glob(os.path.join(stage, "trace/raw/4070ti/trace2/*-cand.log"))), e["cand"], "trace")
    print("candidate environment:", e["cand"] or "(empty)")

mode = sys.argv[1]
try:
    {"verify": do_verify, "sdpa": do_sdpa, "session": do_session, "env": do_env}[mode](*sys.argv[2:])
except Exception as e:
    fail(f"{mode}: cannot evaluate: {e!r}")
print(f"{mode}: {'REJECT' if bad else 'ACCEPT'} ({len(bad)} findings)")
sys.exit(1 if bad else 0)
