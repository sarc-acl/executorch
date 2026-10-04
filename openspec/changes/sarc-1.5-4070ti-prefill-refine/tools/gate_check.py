#!/usr/bin/env python3
"""gate_check.py: decide a gate step from the contents of its result files, not from exit statuses.

  gate_check.py verify  <cand stage dir> <parent control stage dir>   (verify.out, verify/, verify-runs.jsonl)
  gate_check.py sdpa    <sdpa-correctness dir> [<cand env file>]   (cand-{extended,full}-r1..12.log)
  gate_check.py session <stage/<session>/raw> (--clkmin <clkmin.json> | --calibration) [--require-logs]
  gate_check.py env     <stage/<session>>                          (one candidate environment everywhere)

Prints one line per finding and ends with ACCEPT or REJECT; exit 0 only on ACCEPT.

verify: read from the files behind verify.out, for the candidate and for the parent control alike. Every runner
call verify.sh makes (12 prefill, 8 check/unaligned, 2 decode) must have exactly one recorded exit status of 0
(verify-runs.jsonl, written by llama_main_rc.sh), stats, and the expected prompt tokens. Prefill: a positive
rate equal to the one in verify.out, 0 generated tokens. Default vs tiled on prompt_check (1972) and r1304.txt
(1792): recomputed from the two logs with nexttoken.py, so two failed or empty outputs are INVALID, never SAME.
Decode: generated tokens and a positive rate in the log, text after the prompt, the count verify.out shows.
Microbench: every case of correctness.log PASSED, the 12 production-diff logs with every shape PASSED and
ALL PASSED and rc 0. linear-<scheme>.json: every case parsed; the 24 expected identities (3 models x 4
projections x buffer/texture3d) each exactly once; none crashed, each with a kernel, ok=true, positive times
and a known dispatch state; and per case the shape and the (variant, dispatch) pair must equal the parent's, so
a parent anomaly such as unexpected_coopmat is tolerated only as the same anomaly and a new fallback or crash
is a difference. The same holds per case of correctness.log (coopmat kernel or not). `correctness rc` and
`linear <scheme> rc` are not required to be 0 (on this device the shipped state has had rc=1 for a rank-3 case that does not
dispatch coopmat): they must fit their logs and equal the parent control's, as must the case counts, the set
of cases without coopmat and the decode token counts.
sdpa: per tier 12 passes, each with the tier's case count (extended 8, full 4), every case PASSED with
mismatches=0, qk_coopmat=yes, av_coopmat=yes, and a [sdpa-kernels] line with pairing=ok per case; with an env
file that names a profile, every pass must carry that profile's banner.
session: six cells with at least 5 timed runs per arm that are valid on their own fields (the `valid` column is
not trusted): a unique log and model/scheme/build/repeat identity, rc 0, a positive finite rate, 2048 prompt
tokens, 0 generated tokens, no foreign GPU process, at least 2 clock samples and a median clock at or above the
threshold; with the logs present each of them is recomputed from its log and clock samples (runrow.py) and must
equal its row; every timed run is judged against the calibrated
clock threshold of its cell (--clkmin; a record-only session is rejected unless --calibration says it is the
baseline or A/A session, which can never accept a candidate); and, per cell, the next token of parent vs
candidate on the timed prompt, prompt_real_2048.txt (2048 tokens), prompt_check.txt (1972) and r1304.txt (1792).
A next-token row counts only if it is SAME and its evidence holds: the prompt hash is the tracked file's, both
runs are in runs.csv with rc 0 and the expected prompt tokens and no foreign GPU process, both output hashes
are equal and (except for the timed prompt) both tokens are non-empty and equal. With the logs present
(--require-logs makes that mandatory, as in the gates) every row is recomputed from the logs with nexttoken.py.
env: cand/env equals cand-traced/env and parent/env equals parent-traced/env; the ET_VK variables verify.sh
recorded are exactly cand/env; candidate logs carry the profile banner of cand/env and parent logs none."""
import collections, csv, glob, hashlib, json, math, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nexttoken, runrow
CELLS = [(m, q) for m in ("1b", "3b", "8b") for q in ("4w", "8da4w")]
bad = []
def fail(msg): bad.append(msg); print("FAIL:", msg)

def verify_items(path):
    it = {}
    for l in open(path, errors="replace"):
        l = l.strip()
        if m := re.match(r"correctness rc=(\d+)", l): it["correctness rc"] = m[1]
        elif m := re.match(r"linear (\S+) rc=(\d+)", l): it[f"linear {m[1]} rc"] = m[2]
        elif m := re.match(r"(\S+) (\S+) (tiled|default) prefill_tok_s=(\S*)", l): it[f"prefill {m[1]} {m[2]} {m[3]}"] = m[4]
        elif m := re.match(r"(\S+) (\S+) (check|unaligned): default vs tiled output (\S+)", l): it[f"nexttoken {m[1]} {m[2]} {m[3]}"] = m[4]
        elif m := re.match(r'(\S+) (\S+) decode rc=(\d+) "generated_tokens":(\d*)', l): it[f"decode {m[1]} {m[2]}"] = (m[3], m[4])
        elif m := re.match(r"pdiff (\S+) (\S+) (\S+) rc=(\d+)", l): it[f"pdiff {m[1]} {m[2]} {m[3]} rc"] = m[4]
        elif m := re.match(r"VERIFY_DONE rc=(\d+)", l): it["verify_done"] = m[1]
    return it

LINEAR_OPS = ("wq_wo", "wk_wv", "w1_w3", "w2")
LINEAR_DISPATCH = ("confirmed", "unexpected_coopmat", "fallback_tiled", "not_applicable")
PTE = {"llama3_2-1b": "1b", "llama3_2-3b": "3b", "llama3_1-8b": "8b"}
def runner_records(stage, who):
    """verify-runs.jsonl (llama_main_rc.sh) -> {verify log name: [records]}: the exit status of every runner call."""
    recs = collections.defaultdict(list); p = os.path.join(stage, "verify-runs.jsonl")
    if not os.path.exists(p): fail(f"{who}: no verify-runs.jsonl: the runner exit statuses were not recorded"); return recs
    for l in open(p):
        try: r = json.loads(l)
        except ValueError: fail(f"{who}: unreadable line in verify-runs.jsonl"); continue
        mm = re.match(r"(.+)_vulkan_(4w|8da4w)\.pte$", r.get("model", "")); m = PTE.get(mm[1]) if mm else None
        mode = "tiled" if "ET_VK_FORCE_TILED_LINEAR=1" in r.get("env", "").split() else "default"
        pr, nt, wu = r.get("prompt"), str(r.get("max_new_tokens")), r.get("warmup")
        if not m: name = None
        elif pr == "prompt_2048.txt" and nt == "1" and wu == 1: name = f"prefill-{m}-{mm[2]}-{mode}.log"
        elif pr == "prompt_check.txt" and nt == "1" and wu == 0: name = f"check-{m}-{mm[2]}-{mode}.log"
        elif pr == "r1304.txt" and nt == "1" and wu == 0: name = f"unaligned-{m}-{mm[2]}-{mode}.log"
        elif pr == "prompt_2048.txt" and nt == "32" and wu == 0 and mode == "default": name = f"decode-{m}-{mm[2]}.log"
        else: name = None
        if name is None: fail(f"{who}: runner call that verify.sh does not make: {l.strip()[:160]}")
        else: recs[name].append(r)
    return recs

def num(x):
    try: v = float(x)
    except (TypeError, ValueError): return None
    return v if math.isfinite(v) else None

def verify_stage(stage, who):
    """Check one verify.sh run on its own evidence; return the status items compared between candidate and parent."""
    V = os.path.join(stage, "verify"); out = os.path.join(stage, "verify.out"); st = {}
    if not os.path.exists(out) or not os.path.isdir(V): fail(f"{who}: no verify.out or verify/ directory in {stage}"); return st
    it = verify_items(out); recs = runner_records(stage, who)
    if it.get("verify_done") != "0": fail(f"{who}: VERIFY_DONE rc = {it.get('verify_done')!r}")
    if not os.path.exists(os.path.join(V, "done.txt")): fail(f"{who}: verify.sh did not reach its end (no verify/done.txt)")
    def runner(name, want):
        """The run behind verify/<name>: exactly one recorded call with rc 0, stats, the expected prompt tokens."""
        log = os.path.join(V, name); r = recs.get(name, [])
        if len(r) != 1: fail(f"{who}: {name}: {len(r)} recorded runner calls, required 1"); rc = None
        else:
            rc = r[0].get("rc")
            if rc != 0: fail(f"{who}: {name}: runner exit status {rc}")
        if not os.path.exists(log): fail(f"{who}: {name}: log missing"); return None, rc
        obs = nexttoken.observer(log)
        if obs is None: fail(f"{who}: {name}: no PyTorchObserver stats"); return None, rc
        if str(obs.get("prompt_tokens")) != want: fail(f"{who}: {name}: prompt tokens {obs.get('prompt_tokens')!r}, required {want}")
        return obs, rc
    for m, q in CELLS:
        for mode in ("tiled", "default"):
            name = f"prefill-{m}-{q}-{mode}.log"; obs, _ = runner(name, "2048")
            if obs is None: continue
            rate = num(obs.get("prefill_token_per_sec"))
            if rate is None or rate <= 0: fail(f"{who}: {name}: prefill rate {obs.get('prefill_token_per_sec')!r}")
            if str(obs.get("generated_tokens")) != "0": fail(f"{who}: {name}: generated tokens {obs.get('generated_tokens')!r}, required 0")
            if num(it.get(f"prefill {m} {q} {mode}")) != rate: fail(f"{who}: {name}: verify.out says {it.get(f'prefill {m} {q} {mode}')!r}, the log says {rate}")
    for q in ("4w", "8da4w"):
        for kind, prompt in (("check", "prompt_check.txt"), ("unaligned", "r1304.txt")):
            _, want, tracked = PROMPTS[prompt]; rcs = {}
            for mode in ("tiled", "default"): rcs[mode] = runner(f"{kind}-1b-{q}-{mode}.log", want)[1]
            e = nexttoken.evaluate(tracked, want, os.path.join(V, f"{kind}-1b-{q}-tiled.log"), os.path.join(V, f"{kind}-1b-{q}-default.log"),
                                   0 if rcs["tiled"] == 0 else "?", 0 if rcs["default"] == 0 else "?")
            verdict = e["verdict"].replace("parent:", "tiled:").replace("cand:", "default:")
            if verdict != "SAME": fail(f"{who}: {kind} 1b {q}: default vs tiled is {verdict} on the logs (verify.out says {it.get(f'nexttoken 1b {q} {kind}')!r})")
            if it.get(f"nexttoken 1b {q} {kind}") != "SAME": fail(f"{who}: {kind} 1b {q}: verify.out says {it.get(f'nexttoken 1b {q} {kind}')!r}")
        name = f"decode-1b-{q}.log"; obs, _ = runner(name, "2048"); s = it.get(f"decode 1b {q}")
        if s is None or s[0] != "0": fail(f"{who}: {name}: verify.out decode status {s!r}")
        if obs is not None:
            gen = str(obs.get("generated_tokens")); rate = num(obs.get("decode_token_per_sec"))
            if not gen.isdigit() or int(gen) < 1: fail(f"{who}: {name}: generated tokens {gen!r}")
            if rate is None or rate <= 0: fail(f"{who}: {name}: decode rate {obs.get('decode_token_per_sec')!r}")
            if s is not None and s[1] != gen: fail(f"{who}: {name}: verify.out says {s[1]!r} generated tokens, the log says {gen}")
            text = nexttoken.generated(os.path.join(V, name)); prompt = open(PROMPTS["prompt_2048.txt"][2], "rb").read()
            if not (text.startswith(prompt) and text[len(prompt):].strip()): fail(f"{who}: {name}: no generated text after the prompt")
            st[f"decode 1b {q} generated tokens"] = gen
    # microbench evidence: the logs behind the summary lines
    c = os.path.join(V, "correctness.log"); cases = failed = 0; fallback = []; final = None; kclass = {}
    for l in (open(c, errors="replace") if os.path.exists(c) else []):
        w = l.split()
        if l.startswith("[rank3"):
            cases += 1; mm = re.match(r"\[rank3[^\]]*\] (\S+) -> (\S+) \((.*)\), correctness=(\S+)", l)
            if not mm or mm[4] != "PASSED": failed += 1
            else:
                kclass["rank3 " + mm[1]] = "coopmat" if "coopmat" in mm[2] and "NOT coopmat" not in mm[3] else "other"
                if "NOT coopmat" in mm[3]: fallback.append(mm[1])
        elif l.startswith("[correctness]"): final = l.strip()
        elif w and w[-1] in ("PASSED", "FAILED", "SKIPPED", "CRASHED") and "GFLOP/s" in l:
            cases += 1; failed += w[-1] != "PASSED"
            name = next((w[i - 1] for i in range(1, len(w)) if w[i].startswith("[")), None)
            if name is None or name in kclass: fail(f"{who}: correctness.log: case without a unique name: {l.strip()[:120]}")
            else: kclass[name] = "coopmat" if "coopmat" in w[0] else "other"
    if cases == 0 or final is None: fail(f"{who}: correctness.log missing, empty or without its summary line")
    if failed: fail(f"{who}: correctness.log: {failed} case(s) not PASSED")
    rc = it.get("correctness rc")
    if rc is None: fail(f"{who}: no `correctness rc` line")
    elif (rc == "0") != (final is not None and "PASSED" in final and not fallback): fail(f"{who}: correctness rc={rc} does not fit its log ({final}; fallback cases {fallback})")
    st["correctness rc"] = rc; st["correctness cases"] = cases; st["correctness cases without coopmat"] = sorted(fallback)
    for name, k in kclass.items(): st[f"correctness case {name} kernel class"] = k   # a new fallback shows as a difference
    for md in ("llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b"):
        for q in ("4w", "8da4w"):
            for sto in ("buffer", "texture3d"):
                p = os.path.join(V, f"pdiff-{md}-{q}-{sto}.log"); k = f"pdiff {md} {q} {sto}"
                lines = [l for l in (open(p, errors="replace") if os.path.exists(p) else []) if l.startswith("[production-diff]")]
                if it.get(k + " rc") != "0": fail(f"{who}: {k}: rc {it.get(k + ' rc')!r}")
                res = [l for l in lines if " -> " in l]; n = re.search(r"(\d+) shapes", lines[-1]) if lines else None
                if not lines or "ALL PASSED" not in lines[-1] or not res or any("correctness=PASSED" not in l for l in res) or (n and int(n[1]) != len(res)):
                    fail(f"{who}: {k}: the log does not show every shape PASSED and ALL PASSED")
    for q in ("4w", "8da4w"):
        k = f"linear {q} rc"; p = os.path.join(V, f"linear-{q}.json")
        if k not in it: fail(f"{who}: no `linear {q}` line")
        st[k] = it.get(k)
        try: lc = json.load(open(p))["cases"]; assert isinstance(lc, list)
        except (OSError, ValueError, KeyError, TypeError, AssertionError): fail(f"{who}: linear-{q}.json missing or unreadable"); continue
        # Every case, by identity. Required coverage: 3 models x 4 projections x buffer/texture3d, each once.
        want = {(md, op, sto) for md in ("llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b") for op in LINEAR_OPS for sto in ("buffer", "texture3d")}
        got = set()
        for x in lc:
            if not isinstance(x, dict): fail(f"{who}: linear-{q}.json: a case is not an object"); continue
            ident = (x.get("model"), x.get("op"), str(x.get("storage", "")).lower()); w = f"{who}: linear {q} {' '.join(map(str, ident))}"
            if (x.get("suite", "linear"), x.get("scheme"), x.get("regime")) != ("linear", q, "prefill") or ident not in want: fail(f"{w}: not a case verify.sh asks for"); continue
            if ident in got: fail(f"{w}: listed twice"); continue
            got.add(ident); kern = str(x.get("kernel") or ""); why = []
            if not kern or kern.upper() == "CRASHED" or x.get("variant") in (None, "", "crashed") or x.get("dispatch") in (None, "", "crashed"): why.append("crashed or without a kernel")
            if x.get("ok") is not True: why.append(f"ok={x.get('ok')!r}")
            for f in ("kernel_median_us", "kernel_mean_us", "op_mean_us"):
                v = num(x.get(f))
                if v is None or v <= 0: why.append(f"{f}={x.get(f)!r}")
            if not all(isinstance(x.get(f), int) and x.get(f) > 0 for f in ("M", "K", "N")): why.append("M/K/N")
            if str(x.get("correctness")) not in ("SKIPPED", "PASSED"): why.append(f"correctness={x.get('correctness')!r}")
            if x.get("dispatch") not in LINEAR_DISPATCH: why.append(f"dispatch={x.get('dispatch')!r}")
            if why: fail(f"{w}: kernel {kern!r}: " + "; ".join(why))
            # What must not change against the parent: the shape and how the case dispatched. A parent anomaly
            # (e.g. unexpected_coopmat on texture3d) is the same anomaly in the candidate; anything else is new.
            key = f"linear {q} {ident[0]} {ident[1]} {ident[2]}"
            st[key + " shape"] = (x.get("M"), x.get("K"), x.get("N")); st[key + " dispatch"] = (x.get("variant"), x.get("dispatch"))
        for ident in sorted(want - got): fail(f"{who}: linear {q} {' '.join(ident)}: case missing")
    return st

def do_verify(cand, parent):
    c = verify_stage(cand, "candidate")
    p = c if os.path.realpath(cand) == os.path.realpath(parent) else verify_stage(parent, "parent control")
    for k in sorted(set(c) | set(p)):
        if c.get(k) != p.get(k): fail(f"differs from the parent control: {k}: candidate {c.get(k)!r}, parent {p.get(k)!r}")
    print("verify status (candidate / parent): " + "; ".join(f"{k} {c.get(k)}/{p.get(k)}" for k in sorted(c) if k.endswith(" rc")))
    for arm, s in (("candidate", c), ("parent", p)):
        print(f"{arm} linear dispatch states: " + ", ".join(f"{k[1]}={n}" for k, n in sorted(collections.Counter(v for k, v in s.items() if k.endswith(" dispatch")).items())))

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
    have_logs = os.path.isdir(os.path.join(d, "logs"))
    if require_logs and not have_logs: fail("no logs directory: runs cannot be recomputed")
    # Every timed run is judged on its own fields, never on its `valid` flag, and counted once.
    seen_log, seen_id, good = set(), set(), collections.Counter()
    for r in rows:
        cell = (r["model"], r["scheme"]); log = r["log"]; ident = (*cell, r["build"], r["rep"]); why = []
        if log in seen_log or ident in seen_id: fail(f"{log}: duplicate timed run (log or model/scheme/build/rep repeated)"); continue
        seen_log.add(log); seen_id.add(ident)
        if log != f'logs/prefill-{r["model"]}-{r["scheme"]}-{r["build"]}-r{r["rep"]}.log': why.append("log name does not match the run's identity")
        if cell not in CELLS or r["build"] not in ("parent", "cand"): why.append("unknown cell or arm")
        applied = r.get("clkmin") or ""
        if calibration:
            want = "0"
            if applied != "0": fail(f"{log}: clkmin {applied!r} in a calibration session, expected 0")
        else:
            want = str(clk.get(f'{r["model"]}-{r["scheme"]}', {}).get("clkmin_mhz", ""))
            if not want.isdigit() or int(want) <= 0: fail(f'no calibrated clock threshold for {r["model"]} {r["scheme"]}'); want = None
            elif applied != want: fail(f"{log}: clock threshold applied {applied!r}, calibrated {want} (record-only or stale)")
        if r["rc"] != "0": why.append(f'rc {r["rc"]}')
        try: ok = math.isfinite(float(r["tok_s"])) and float(r["tok_s"]) > 0
        except ValueError: ok = False
        if not ok: why.append(f'tok_s {r["tok_s"]!r}')
        if r["prompt_tokens"] != "2048": why.append(f'prompt_tokens {r["prompt_tokens"]!r}')
        if r["generated_tokens"] != "0": why.append(f'generated_tokens {r["generated_tokens"]!r}')
        if r["others"]: why.append("foreign GPU process")
        try: n = int(r["clk_n"]); cm = float(r["clk_med_mhz"])
        except ValueError: n, cm = 0, float("nan")
        if n < 2: why.append(f'clock samples {r["clk_n"]!r}')
        elif want is None or not cm >= int(want): why.append(f'clock {r["clk_med_mhz"]} MHz below {want}')
        if have_logs and want is not None:
            rec = runrow.evaluate(os.path.join(d, log), os.path.join(d, log[:-4] + ".clk"), "2048", want, r["rc"], r["others"], "prefill")
            diff = [k for k in runrow.FIELDS if rec[k] != r[k]]
            if diff: why.append("recomputed from the log and clock samples differs in " + ", ".join(f"{k} ({r[k]!r} -> {rec[k]!r})" for k in diff))
        if why and r["valid"] == "1": fail(f"{log}: marked valid but " + "; ".join(why))
        elif not why and r["valid"] != "1": fail(f'{log}: marked invalid ({r["reason"]}) but every field is in order')
        elif not why: good[(*cell, r["build"])] += 1
    for m, q in CELLS:
        for b in ("parent", "cand"):
            if good[(m, q, b)] < 5: fail(f"cell {m} {q} {b}: {good[(m, q, b)]} independently valid timed runs, required 5")
    if any(r["others"] for r in allrows): fail("a run overlapped a GPU process of another owner")
    nt = collections.defaultdict(dict)
    p = os.path.join(d, "nexttoken.csv")
    for r in (csv.DictReader(open(p)) if os.path.exists(p) else []): nt[(r["model"], r["scheme"])][r["prompt"]] = r
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
