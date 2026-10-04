#!/usr/bin/env python3
"""Regression tests of gate_check.py session and nexttoken.py on synthetic sessions (no GPU, no artifacts):
    python3 tools/test_gate_check.py
The second test is the failure found in review: 60 valid-marked timed rows, all next-token runs failing with
rc=134 and no output, and SAME written in nexttoken.csv. It must be rejected."""
import csv, hashlib, json, os, subprocess, sys, tempfile, unittest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import nexttoken, gate_check_paths  # noqa: E402  (prompt table shared with gate_check.py)
CELLS = [(m, q) for m in ("1b", "3b", "8b") for q in ("4w", "8da4w")]
HEADER = "gpu,host,model,scheme,build,rep,slot,tok_s,rc,temp_pre,temp_post,cool_s,clocks,others,utc,log,prompt_tokens,generated_tokens,prefill_ms,clk_n,clk_med_mhz,clk_min_mhz,busy_med,power_med_w,temp_max,valid,reason,clkmin".split(",")
PROMPTS = gate_check_paths.PROMPTS

def log_text(prompt, token, want):
    return open(prompt, "rb").read() + token + b"\n" + ('PyTorchObserver {"prompt_tokens":%s,"generated_tokens":0,"prefill_token_per_sec":1000.0}\n' % want).encode()

def make(d, clkmin="2900", check_rc="0", token=b" tok", cand_token=None, write_logs=True, drop=None, forge_same=False):
    """A complete session in d; the keyword arguments break it in one way each."""
    os.makedirs(os.path.join(d, "logs")); rows = []; nt = []
    def run(m, q, b, tag, rep, prompt, want, rc, tok):
        log = f"logs/{tag}-{m}-{q}-{b}-r{rep}.log"; ok = rc == "0"
        with open(os.path.join(d, log), "wb") as f: f.write(log_text(prompt, tok, want) if ok else b"")
        r = dict.fromkeys(HEADER, ""); timed = tag == "prefill"
        r.update(gpu="4070ti", host="t", model=m, scheme=q, build=b, rep=str(rep), tok_s="1000.0" if ok else "", rc=rc, log=log,
                 prompt_tokens=want if ok else "", generated_tokens="0" if ok else "", clk_n="5", clk_med_mhz="3000.0",
                 valid="1" if ok else "0", reason="" if ok else "rc+no_tok_s", clkmin=clkmin if timed else "0")
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
        rc, out = self.gate("--require-logs"); self.assertIn("cannot be recomputed", out)

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
        make(self.d); p = os.path.join(self.d, "runs.csv"); t = open(p).read().replace(",1792,", ",1791,"); open(p, "w").write(t)
        rc, out = self.gate(); self.assertNotEqual(rc, 0); self.assertIn("prompt tokens", out)

    def test_record_only_clock_cannot_pass_a_gate(self):
        make(self.d, clkmin="0"); rc, out = self.gate("--require-logs"); self.assertNotEqual(rc, 0); self.assertIn("record-only or stale", out)
        rc, out = check(self.d, "--calibration", "--require-logs"); self.assertEqual(rc, 0, out); self.assertIn("never a candidate gate", out)

    def test_clock_option_is_mandatory(self):
        make(self.d); rc, out = check(self.d); self.assertNotEqual(rc, 0, out)
        rc, out = check(self.d, "--calibration"); self.assertNotEqual(rc, 0, out)   # thresholds were applied: not a calibration session

if __name__ == "__main__":
    unittest.main(verbosity=1, warnings="ignore")
