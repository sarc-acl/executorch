#!/usr/bin/env python3
"""Regression tests of gate_check.py session and nexttoken.py on synthetic sessions (no GPU, no artifacts):
    python3 tools/test_gate_check.py
The second test is the failure found in review: 60 valid-marked timed rows, all next-token runs failing with
rc=134 and no output, and SAME written in nexttoken.csv. It must be rejected. So must
the two found in the next review: failed timed repeats that keep valid=1, and five copies of one timed run."""
import csv, hashlib, json, os, subprocess, sys, tempfile, unittest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import nexttoken, gate_check_paths  # noqa: E402  (prompt table shared with gate_check.py)
CELLS = [(m, q) for m in ("1b", "3b", "8b") for q in ("4w", "8da4w")]
HEADER = "gpu,host,model,scheme,build,rep,slot,tok_s,rc,temp_pre,temp_post,cool_s,clocks,others,utc,log,prompt_tokens,generated_tokens,prefill_ms,clk_n,clk_med_mhz,clk_min_mhz,busy_med,power_med_w,temp_max,valid,reason,clkmin".split(",")
PROMPTS = gate_check_paths.PROMPTS

T0 = 1790000000000  # ms; every synthetic run measures from T0 to T0 + 100 ms
def log_text(prompt, token, want):
    return open(prompt, "rb").read() + token + b"\n" + ('PyTorchObserver {"prompt_tokens":%s,"generated_tokens":0,"prefill_token_per_sec":1000.0,"inference_start_ms":%d,"prompt_eval_end_ms":%d}\n' % (want, T0, T0 + 100)).encode()
def clk_text(mhz=3000): return "".join(f"{(T0 + 20 * i) * 1000} {mhz} 97 250.0 55\n" for i in range(5))
def rewrite(d, fn):
    """Apply fn(row) to every row of runs.csv (fn may return a list of rows to replace it)."""
    p = os.path.join(d, "runs.csv"); out = []
    with open(p) as f: rows = list(csv.DictReader(f))
    for r in rows:
        x = fn(r); out += x if isinstance(x, list) else [r]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, HEADER, lineterminator="\n"); w.writeheader(); w.writerows(out)

def make(d, clkmin="2900", check_rc="0", token=b" tok", cand_token=None, write_logs=True, drop=None, forge_same=False):
    """A complete session in d; the keyword arguments break it in one way each."""
    os.makedirs(os.path.join(d, "logs")); rows = []; nt = []
    def run(m, q, b, tag, rep, prompt, want, rc, tok):
        log = f"logs/{tag}-{m}-{q}-{b}-r{rep}.log"; ok = rc == "0"
        with open(os.path.join(d, log), "wb") as f: f.write(log_text(prompt, tok, want) if ok else b"")
        with open(os.path.join(d, log[:-4] + ".clk"), "w") as f: f.write(clk_text() if ok else "")
        r = dict.fromkeys(HEADER, ""); timed = tag == "prefill"
        r.update(gpu="4070ti", host="t", model=m, scheme=q, build=b, rep=str(rep), tok_s="1000.0" if ok else "", rc=rc, log=log,
                 prompt_tokens=want if ok else "", generated_tokens="0" if ok else "", prefill_ms="100" if ok else "",
                 clk_n="5" if ok else "0", clk_med_mhz="3000.0" if ok else "", clk_min_mhz="3000.0" if ok else "",
                 busy_med="97.0" if ok else "", power_med_w="250.0" if ok else "", temp_max="55" if ok else "",
                 valid="1" if ok else "0", reason="" if ok else "rc+no_tok_s+prompt_tokens", clkmin=clkmin if timed else "0")
        rows.append(r); return log
    for m, q in CELLS:
        for rep in range(1, 6):
            for b in ("parent", "cand"): run(m, q, b, "prefill", rep, PROMPTS["prompt_2048.txt"][2], "2048", "0", b" the")
        for name, (tag, want, path) in PROMPTS.items():
            if name == drop: continue
            if tag == "prefill":
                logs = [f"logs/prefill-{m}-{q}-{b}-r1.log" for b in ("parent", "cand")]; rcs = ("0", "0")
            else:
                logs = [run(m, q, "parent", tag, 0, path, want, check_rc, token), run(m, q, "cand", tag, 0, path, want, check_rc, cand_token or token)]; rcs = (check_rc, check_rc)
            e = nexttoken.evaluate(path, want, os.path.join(d, logs[0]), os.path.join(d, logs[1]), *rcs, tag == "prefill")
            e["prompt"] = name
            if forge_same: e["verdict"] = "SAME"
            nt.append(dict(e, model=m, scheme=q))
    for name, fields, data in (("runs.csv", HEADER, rows), ("nexttoken.csv", nexttoken.FIELDS, nt)):
        with open(os.path.join(d, name), "w", newline="") as f:
            w = csv.DictWriter(f, fields, lineterminator="\n"); w.writeheader(); w.writerows(data)
    open(os.path.join(d, "done.txt"), "w").write("done\n")
    json.dump({"cells": {f"{m}-{q}": {"clkmin_mhz": 2900} for m, q in CELLS}}, open(os.path.join(d, "clkmin.json"), "w"))
    if not write_logs:
        for f in os.listdir(os.path.join(d, "logs")): os.remove(os.path.join(d, "logs", f))
        os.rmdir(os.path.join(d, "logs"))

def check(d, *opts):
    p = subprocess.run([sys.executable, os.path.join(HERE, "gate_check.py"), "session", d, *opts], capture_output=True, text=True)
    return p.returncode, p.stdout

class Session(unittest.TestCase):
    def setUp(self): self.t = tempfile.TemporaryDirectory(); self.d = os.path.join(self.t.name, "raw"); os.mkdir(self.d)
    def tearDown(self): self.t.cleanup()
    def gate(self, *extra): return check(self.d, "--clkmin", os.path.join(self.d, "clkmin.json"), *extra)

    def test_complete_session_is_accepted(self):
        make(self.d); rc, out = self.gate("--require-logs"); self.assertEqual(rc, 0, out); self.assertIn("session: ACCEPT", out)

    def test_review_case_failed_token_runs_marked_same(self):
        # 60 valid timed rows; every next-token run rc=134 with no output; nexttoken.csv says SAME.
        make(self.d, check_rc="134", forge_same=True)
        self.assertEqual(sum(1 for r in csv.DictReader(open(os.path.join(self.d, "runs.csv"))) if r["valid"] == "1"), 60)
        rc, out = self.gate("--require-logs"); self.assertNotEqual(rc, 0, out); self.assertIn("session: REJECT", out)
        self.assertIn("rc 134", out)
        rc, out = self.gate(); self.assertNotEqual(rc, 0, out)   # also without recomputing from the logs

    def test_review_case_without_logs(self):
        make(self.d, check_rc="134", forge_same=True, write_logs=False)
        rc, out = self.gate(); self.assertNotEqual(rc, 0, out); self.assertIn("rc 134", out)
        rc, out = self.gate("--require-logs"); self.assertIn("runs cannot be recomputed", out)

    def test_review_case_failed_timed_repeats_marked_valid(self):
        # Timed repeats 2-5: rc=134, no rate, 17 prompt tokens, 4 generated, no clock samples, empty logs; valid=1 kept.
        make(self.d)
        def brk(r):
            if r["log"].startswith("logs/prefill") and r["rep"] != "1":
                r.update(rc="134", tok_s="", prompt_tokens="17", generated_tokens="4", clk_n="0")
                for ext in (".log", ".clk"): open(os.path.join(self.d, r["log"][:-4] + ext), "w").close()
        rewrite(self.d, brk)
        for extra in (("--require-logs",), ()):
            rc, out = self.gate(*extra); self.assertNotEqual(rc, 0, out); self.assertIn("session: REJECT", out)
            self.assertIn("marked valid but rc 134", out); self.assertIn("1 independently valid timed runs, required 5", out)

    def test_review_case_five_copies_of_one_run(self):
        # Each arm's five timed repeats replaced by five copies of its r1 row.
        make(self.d); first = {}
        def dup(r):
            if not r["log"].startswith("logs/prefill"): return None
            k = (r["model"], r["scheme"], r["build"])
            if r["rep"] == "1": first[k] = dict(r); return None
            return [dict(first[k])]
        rewrite(self.d, dup)
        for extra in (("--require-logs",), ()):
            rc, out = self.gate(*extra); self.assertNotEqual(rc, 0, out)
            self.assertIn("duplicate timed run", out); self.assertIn("1 independently valid timed runs, required 5", out)

    def test_row_edited_without_its_log_is_caught(self):
        # A faster rate typed into runs.csv: every field looks valid, only the recomputation from the log shows it.
        make(self.d); rewrite(self.d, lambda r: r.update(tok_s="1200.0") if r["log"] == "logs/prefill-8b-4w-cand-r3.log" else None)
        rc, out = self.gate("--require-logs"); self.assertNotEqual(rc, 0, out); self.assertIn("recomputed from the log and clock samples differs in tok_s", out)

    def test_low_clock_and_wrong_identity(self):
        make(self.d)
        def brk(r):
            if r["log"] == "logs/prefill-1b-4w-cand-r2.log": r.update(clk_med_mhz="2500.0")
            if r["log"] == "logs/prefill-3b-4w-parent-r4.log": r.update(rep="5x")
        rewrite(self.d, brk); rc, out = self.gate(); self.assertNotEqual(rc, 0, out)
        self.assertIn("clock 2500.0 MHz below 2900", out); self.assertIn("log name does not match", out)

    def test_empty_outputs_are_never_same(self):
        e = os.path.join(self.t.name, "e.log"); open(e, "w").close()
        for rc in ("0", "134"):
            v = nexttoken.evaluate(PROMPTS["r1304.txt"][2], "1792", e, e, rc, rc)["verdict"]
            self.assertTrue(v.startswith("INVALID:"), v)

    def test_unforged_failed_runs_are_invalid(self):
        make(self.d, check_rc="134"); rc, out = self.gate("--require-logs"); self.assertNotEqual(rc, 0); self.assertIn("INVALID:", out)

    def test_differing_token_is_rejected(self):
        make(self.d, cand_token=b" other"); rc, out = self.gate("--require-logs"); self.assertNotEqual(rc, 0); self.assertIn("DIFFER", out)

    def test_same_forged_over_differing_logs_is_caught(self):
        make(self.d, cand_token=b" other", forge_same=True); rc, out = self.gate("--require-logs"); self.assertNotEqual(rc, 0)
        self.assertIn("marked SAME but outputs differ", out)

    def test_missing_r1304_and_real_rows(self):
        for drop in ("r1304.txt", "prompt_real_2048.txt"):
            with tempfile.TemporaryDirectory() as t:
                d = os.path.join(t, "raw"); os.mkdir(d); make(d, drop=drop)
                rc, out = check(d, "--clkmin", os.path.join(d, "clkmin.json")); self.assertNotEqual(rc, 0); self.assertIn(f"{drop}: no row", out)

    def test_wrong_prompt_length_is_rejected(self):
        make(self.d); rewrite(self.d, lambda r: r.update(prompt_tokens="1791") if r["prompt_tokens"] == "1792" else None)
        rc, out = self.gate(); self.assertNotEqual(rc, 0); self.assertIn("prompt tokens", out)

    def test_record_only_clock_cannot_pass_a_gate(self):
        make(self.d, clkmin="0"); rc, out = self.gate("--require-logs"); self.assertNotEqual(rc, 0); self.assertIn("record-only or stale", out)
        rc, out = check(self.d, "--calibration", "--require-logs"); self.assertEqual(rc, 0, out); self.assertIn("never a candidate gate", out)

    def test_clock_option_is_mandatory(self):
        make(self.d); rc, out = check(self.d); self.assertNotEqual(rc, 0, out)
        rc, out = check(self.d, "--calibration"); self.assertNotEqual(rc, 0, out)   # thresholds were applied: not a calibration session

MODELS = {"1b": ("llama-3.2-1b", "llama3_2-1b"), "3b": ("llama-3.2-3b", "llama3_2-3b"), "8b": ("llama-3.1-8b", "llama3_1-8b")}
def make_verify(stage, correctness_rc="0", fallback=False, token=b" tok"):
    """A complete verify.sh result (verify.out, verify/, verify-runs.jsonl) as the gates stage and record it."""
    V = os.path.join(stage, "verify"); os.makedirs(V); out = []; recs = []
    def call(name, m, q, prompt, want, ntok, warm, mode, text, stats):
        with open(os.path.join(V, name), "wb") as f:
            f.write(open(PROMPTS[prompt][2], "rb").read() + text + b"\n" + ("PyTorchObserver " + json.dumps(dict(stats, prompt_tokens=int(want))) + "\n").encode())
        recs.append({"utc": "t", "rc": 0, "model": f"{MODELS[m][1]}_vulkan_{q}.pte", "prompt": prompt, "max_new_tokens": ntok,
                     "warmup": warm, "env": "ET_VK_FORCE_TILED_LINEAR=1" if mode == "tiled" else ""})
    rows = "".join(f"kernel (1,1,1) (1,1,1) case{i} [1x1] 1.0 μs 100.0 GFLOP/s   PASSED\n" for i in range(6))
    rows += "[rank3 batch=1] r3a -> sarc_k (coopmat dispatched), correctness=PASSED\n"
    rows += "[rank3 batch=1] r3b -> tiled_k (%s), correctness=PASSED\n" % ("NOT coopmat -- fallback" if fallback else "coopmat dispatched")
    rows += "[correctness] FAILED -- numeric failure(s) and/or a rank-3 case did not dispatch coopmat\n" if fallback else "[correctness] PASSED\n"
    open(os.path.join(V, "correctness.log"), "w").write(rows)
    out.append(f"correctness rc={correctness_rc} x")
    for q in ("4w", "8da4w"):
        json.dump({"cases": [{"kernel": "sarc_k", "kernel_median_us": 1.0}]}, open(os.path.join(V, f"linear-{q}.json"), "w"))
        out.append(f'linear {q} rc=1 kernels: 1 "sarc_k";')
    for m in MODELS:
        for q in ("4w", "8da4w"):
            for mode in ("tiled", "default"):
                call(f"prefill-{m}-{q}-{mode}.log", m, q, "prompt_2048.txt", "2048", "1", 1, mode, b" the", {"prefill_token_per_sec": 1234.5, "generated_tokens": 0})
                out.append(f"{m} {q} {mode} prefill_tok_s=1234.5")
                if m == "1b":
                    call(f"check-{m}-{q}-{mode}.log", m, q, "prompt_check.txt", "1972", "1", 0, mode, token, {"generated_tokens": 0})
                    call(f"unaligned-{m}-{q}-{mode}.log", m, q, "r1304.txt", "1792", "1", 0, mode, token, {"generated_tokens": 0})
            if m == "1b":
                out += [f"{m} {q} check: default vs tiled output SAME", f"{m} {q} unaligned: default vs tiled output SAME"]
                call(f"decode-{m}-{q}.log", m, q, "prompt_2048.txt", "2048", "32", 0, "default", b" a b c", {"generated_tokens": 31, "decode_token_per_sec": 160.0})
                out.append(f'{m} {q} decode rc=0 "generated_tokens":31 decode_tok_s=160.0')
    for md, _ in MODELS.values():
        for q in ("4w", "8da4w"):
            for sto in ("buffer", "texture3d"):
                open(os.path.join(V, f"pdiff-{md}-{q}-{sto}.log"), "w").write("[production-diff] s1 -> k (coopmat dispatched), correctness=PASSED\n[production-diff] ALL PASSED (x)\n")
                out.append(f"pdiff {md} {q} {sto} rc=0 [production-diff] ALL PASSED (x)")
    out.append("VERIFY_DONE rc=0")
    open(os.path.join(stage, "verify.out"), "w").write("\n".join(out) + "\n"); open(os.path.join(V, "done.txt"), "w").write("d\n")
    with open(os.path.join(stage, "verify-runs.jsonl"), "w") as f: f.writelines(json.dumps(r) + "\n" for r in recs)

def set_rc(stage, pred, rc):
    p = os.path.join(stage, "verify-runs.jsonl"); recs = [json.loads(l) for l in open(p)]
    for r in recs:
        if pred(r): r["rc"] = rc
    with open(p, "w") as f: f.writelines(json.dumps(r) + "\n" for r in recs)

class Verify(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory(); self.c = os.path.join(self.t.name, "cand"); self.p = os.path.join(self.t.name, "parent")
        make_verify(self.c); make_verify(self.p)
    def tearDown(self): self.t.cleanup()
    def check(self):
        p = subprocess.run([sys.executable, os.path.join(HERE, "gate_check.py"), "verify", self.c, self.p], capture_output=True, text=True)
        return p.returncode, p.stdout
    def empty(self, stage, kinds=("check", "unaligned")):
        for f in os.listdir(os.path.join(stage, "verify")):
            if f.startswith(kinds): open(os.path.join(stage, "verify", f), "w").close()

    def test_complete_verify_is_accepted(self):
        rc, out = self.check(); self.assertEqual(rc, 0, out); self.assertIn("verify: ACCEPT", out)

    def test_review_case_empty_check_logs_with_same_in_summary(self):
        # verify.out keeps its SAME lines and VERIFY_DONE rc=0 while every check/unaligned log is empty.
        self.empty(self.c); self.empty(self.p)
        rc, out = self.check(); self.assertNotEqual(rc, 0, out); self.assertIn("verify: REJECT", out)
        self.assertIn("default vs tiled is INVALID", out); self.assertIn("no PyTorchObserver stats", out)

    def test_failed_default_vs_tiled_runs(self):
        self.empty(self.c); set_rc(self.c, lambda r: r["prompt"] in ("prompt_check.txt", "r1304.txt"), 134)
        rc, out = self.check(); self.assertNotEqual(rc, 0, out); self.assertIn("runner exit status 134", out)

    def test_teardown_failure_after_stats_were_printed(self):
        # The log is complete (stats, rate, token); only the recorded exit status shows the crash at exit.
        set_rc(self.c, lambda r: r["model"].startswith("llama3_1-8b") and r["warmup"] == 1 and not r["env"], 139)
        rc, out = self.check(); self.assertNotEqual(rc, 0, out); self.assertIn("prefill-8b-4w-default.log: runner exit status 139", out)
        make_verify(os.path.join(self.t.name, "c2")); self.c = os.path.join(self.t.name, "c2")
        set_rc(self.c, lambda r: r["prompt"] == "r1304.txt" and r["env"], 139)
        rc, out = self.check(); self.assertNotEqual(rc, 0, out); self.assertIn("unaligned-1b-4w-tiled.log: runner exit status 139", out)

    def test_exit_statuses_must_be_recorded_once(self):
        os.remove(os.path.join(self.c, "verify-runs.jsonl")); rc, out = self.check(); self.assertNotEqual(rc, 0); self.assertIn("no verify-runs.jsonl", out)
        self.c = os.path.join(self.t.name, "c3"); make_verify(self.c); p = os.path.join(self.c, "verify-runs.jsonl"); l = open(p).readlines()
        open(p, "w").writelines(l + l[:1]); rc, out = self.check(); self.assertNotEqual(rc, 0); self.assertIn("2 recorded runner calls, required 1", out)

    def test_differing_default_and_tiled_tokens(self):
        f = os.path.join(self.c, "verify", "check-1b-8da4w-default.log"); t = open(f, "rb").read().replace(b" tok\n", b" other\n"); open(f, "wb").write(t)
        rc, out = self.check(); self.assertNotEqual(rc, 0); self.assertIn("check 1b 8da4w: default vs tiled is DIFFER", out)

    def test_prefill_and_decode_evidence(self):
        V = os.path.join(self.c, "verify")
        open(os.path.join(V, "prefill-3b-4w-default.log"), "w").close()
        f = os.path.join(V, "decode-1b-4w.log"); t = open(f).read().replace('"generated_tokens": 31', '"generated_tokens": 0'); open(f, "w").write(t)
        f = os.path.join(V, "prefill-1b-4w-tiled.log"); t = open(f).read().replace("1234.5", "999.0"); open(f, "w").write(t)
        rc, out = self.check(); self.assertNotEqual(rc, 0)
        for s in ("prefill-3b-4w-default.log: no PyTorchObserver stats", "decode-1b-4w.log: generated tokens '0'", "prefill-1b-4w-tiled.log: verify.out says '1234.5', the log says 999.0"): self.assertIn(s, out)

    def test_microbench_logs_are_read(self):
        V = os.path.join(self.c, "verify")
        open(os.path.join(V, "pdiff-llama-3.2-3b-8da4w-buffer.log"), "w").close()
        f = os.path.join(V, "correctness.log"); t = open(f).read().replace("case3 [1x1] 1.0 μs 100.0 GFLOP/s   PASSED", "case3 [1x1] 1.0 μs 100.0 GFLOP/s   FAILED"); open(f, "w").write(t)
        rc, out = self.check(); self.assertNotEqual(rc, 0)
        self.assertIn("pdiff llama-3.2-3b 8da4w buffer: the log does not show", out); self.assertIn("correctness.log: 1 case(s) not PASSED", out)

    def test_correctness_rc1_for_a_fallback_case_must_match_the_parent(self):
        # rc=1 because a rank-3 case does not dispatch coopmat (seen on this device): fine when the parent has the same.
        for d in ("c4", "p4"): make_verify(os.path.join(self.t.name, d), correctness_rc="1", fallback=True)
        self.c, self.p = os.path.join(self.t.name, "c4"), os.path.join(self.t.name, "p4")
        rc, out = self.check(); self.assertEqual(rc, 0, out)
        self.p = os.path.join(self.t.name, "parent"); rc, out = self.check(); self.assertNotEqual(rc, 0)
        self.assertIn("differs from the parent control: correctness rc", out)

if __name__ == "__main__":
    unittest.main(verbosity=1, warnings="ignore")
