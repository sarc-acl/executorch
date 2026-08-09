"""Measurement-discipline driver for e2e prefill ranking (component A).

Replaces the ad-hoc gate_and_rank.sh with the four disciplines the 035 sweep
lacked:
- device fingerprint recorded into every result file;
- interleaved control runs with a drift ladder (>3% off baseline -> cool-down
  and re-check, then re-baseline or abort in --strict);
- explicit rep policy (1 discarded warmup + median; finalists get 5 reps) and
  a noise floor from 5 baseline control reps;
- output-hash comparison against the control run (temperature 0 => stdout
  minus the PyTorchObserver line must be byte-identical; a mismatch is a
  silent miscompute the 44-case gate did not catch).

Every llama_main/bench invocation is a fresh subprocess: the variant env var
is read once per process and cached (QuantizedLinear.cpp).
"""

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import device_fingerprint as dfp

REPO = Path(__file__).resolve().parent.parent.parent.parent
BENCH = REPO / "cmake-out-vk/backends/vulkan/test/custom_ops/test_coopmat_linear_bench"
RUNNER = REPO / "cmake-out-vk/examples/models/llama/llama_main"

CKPT = Path("/home/doremy/checkpoints/llama3_2_1b")
DEFAULTS = {
    "q4gsw": {
        "pte": CKPT / "pte/llama3_2_1b_4w_buffer_ctx3072.pte",
        "env_var": "ET_VK_Q4GSW_COOPMAT_VARIANT",
        "kernel_base": "linear_q4gsw_coopmat",
    },
    "dq8ca": {
        "pte": CKPT / "pte/llama3_2_1b_8da4w_buffer_ctx3072.pte",
        "env_var": "ET_VK_DQ8CA_COOPMAT_VARIANT",
        "kernel_base": "linear_dq8ca_q4gsw_coopmat",
    },
}
TOKENIZER = CKPT / "original/tokenizer.model"
PROMPT = CKPT / "p2048_exact.txt"

OBSERVER_RE = re.compile(r"PyTorchObserver (\{.*\})")


@dataclass
class RepResult:
    prefill_tok_s: float
    model_load_ms: float
    inference_ms: float
    output_hash: str
    output_tail: str


@dataclass
class GateResult:
    status: str  # pass | correctness_fail | alignment_fallback | missing_shader
    fails: int = 0
    dispatched: int = 0
    detail: str = ""


@dataclass
class Measurement:
    token: str
    stage: str
    reps: list = field(default_factory=list)
    correctness_flag: bool = False

    @property
    def median(self):
        vals = [r.prefill_tok_s for r in self.reps]
        return statistics.median(vals) if vals else 0.0


class DriftAbort(RuntimeError):
    pass


class Session:
    def __init__(
        self,
        shader,
        pte=None,
        tokenizer=TOKENIZER,
        prompt_file=PROMPT,
        out_jsonl=None,
        seq_len=3072,
        control_every=8,
        drift_pct=3.0,
        strict=False,
        inter_run_sleep_s=1.0,
        quirks=(),
        slug_suffix="",
    ):
        cfg = DEFAULTS[shader]
        self.shader = shader
        self.pte = Path(pte or cfg["pte"])
        self.tokenizer = Path(tokenizer)
        self.prompt_file = Path(prompt_file)
        self.env_var = cfg["env_var"]
        self.kernel_base = cfg["kernel_base"]
        self.seq_len = seq_len
        self.control_every = control_every
        self.drift_frac = drift_pct / 100.0
        self.strict = strict
        self.sleep_s = inter_run_sleep_s
        self.fingerprint = dfp.fingerprint()
        self.device_slug = dfp.device_slug(self.fingerprint)
        # device_slug is derived purely from the fingerprint's device_name,
        # which is chip-family-invariant -- two different physical boards of
        # the same chip collide on the same slug. slug_suffix (sweep.py's
        # --slug-suffix) disambiguates results/journal/blocklist filenames
        # when that matters (specs/041-dbuf4-tile-sweep: a second board on
        # the same host as the documented one).
        if slug_suffix:
            self.device_slug = f"{self.device_slug}-{slug_suffix}"
        self.limits = dfp.limits_from_fingerprint(self.fingerprint, quirks)
        self.out_jsonl = Path(
            out_jsonl
            or Path(__file__).resolve().parent.parent
            / "results"
            / f"runs_{self.device_slug}_{shader}.jsonl"
        )
        self.baseline_id = 0
        self.baseline_median = None
        self.noise_floor_cov = None
        self.control_hash = None
        self.since_control = 0
        self.remeasure = []  # tokens measured inside a drifted window
        self._window = []
        if not self.out_jsonl.exists() or self.out_jsonl.stat().st_size == 0:
            self._record({"stage": "fingerprint", **self.fingerprint})

    # ---------- low-level runs ----------

    def _run_llama(self, token=None):
        import os

        env = dict(os.environ)
        if token:
            env[self.env_var] = token
        proc = subprocess.run(
            [
                str(RUNNER),
                "--model_path",
                str(self.pte),
                "--tokenizer_path",
                str(self.tokenizer),
                "--prompt_file",
                str(self.prompt_file),
                "--num_bos",
                "1",
                "--temperature",
                "0",
                "--max_new_tokens",
                "1",
                "--seq_len",
                str(self.seq_len),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        m = OBSERVER_RE.search(proc.stdout)
        if proc.returncode != 0 or not m:
            raise RuntimeError(
                f"llama_main failed (rc={proc.returncode}, token={token}): "
                f"{proc.stderr[-500:]}{proc.stdout[-200:]}"
            )
        stats = json.loads(m.group(1))
        text = OBSERVER_RE.sub("", proc.stdout).strip()
        time.sleep(self.sleep_s)
        return RepResult(
            prefill_tok_s=float(stats["prefill_token_per_sec"]),
            model_load_ms=stats["model_load_end_ms"] - stats["model_load_start_ms"],
            inference_ms=stats["inference_end_ms"] - stats["inference_start_ms"],
            output_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
            output_tail=text[-48:],
        )

    def _record(self, rec):
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "schema_version": "036.1",
            "device_slug": self.device_slug,
            "shader": self.shader,
            "baseline_id": self.baseline_id,
            **rec,
        }
        self.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _control_reps(self, n, stage="control"):
        vals = []
        for i in range(n):
            r = self._run_llama(None)
            vals.append(r)
            self._record(
                {
                    "stage": stage,
                    "candidate_token": "CONTROL",
                    "rep": i,
                    "prefill_tok_s": r.prefill_tok_s,
                    "model_load_ms": r.model_load_ms,
                    "output_hash": r.output_hash,
                }
            )
        return vals

    # ---------- public protocol ----------

    def baseline(self):
        reps = self._control_reps(5, stage="baseline")
        vals = [r.prefill_tok_s for r in reps]
        self.baseline_median = statistics.median(vals)
        mean = statistics.mean(vals)
        self.noise_floor_cov = (statistics.stdev(vals) / mean) if mean else 0.0
        self.control_hash = reps[0].output_hash
        hashes = {r.output_hash for r in reps}
        if len(hashes) != 1:
            raise RuntimeError(f"control output not deterministic: {hashes}")
        self._record(
            {
                "stage": "baseline_summary",
                "median_prefill_tok_s": self.baseline_median,
                "noise_floor_cov": self.noise_floor_cov,
                "control_hash": self.control_hash,
            }
        )
        return self.baseline_median, self.noise_floor_cov

    def gate(self, token):
        import os

        env = {**os.environ, self.env_var: token, "COOPMAT_BENCH_CORRECTNESS_ONLY": "1"}
        proc = subprocess.run(
            [str(BENCH)], env=env, capture_output=True, text=True, timeout=1800
        )
        out = proc.stdout + proc.stderr
        if "Could not find ShaderInfo" in out:
            g = GateResult("missing_shader", detail=out[-300:])
        elif proc.returncode != 0:
            g = GateResult(
                "correctness_fail", detail=f"rc={proc.returncode} {out[-300:]}"
            )
        else:
            fails = len(re.findall(r"\bFAILED\b", out))
            dispatched = len(re.findall(rf"{self.kernel_base}_{token}", out))
            if fails:
                g = GateResult("correctness_fail", fails, dispatched)
            elif dispatched == 0:
                g = GateResult("alignment_fallback", 0, 0)
            else:
                g = GateResult("pass", 0, dispatched)
        self._record(
            {
                "stage": "gate",
                "candidate_token": token,
                "gate_status": g.status,
                "gate_fails": g.fails,
                "gate_dispatched": g.dispatched,
            }
        )
        return g

    def measure(self, token, reps=2, stage="screen"):
        if self.baseline_median is None:
            self.baseline()
        self._maybe_control()
        m = Measurement(token, stage)
        self._run_llama(token)  # warmup, discarded (pipeline compile, cache)
        for i in range(reps):
            r = self._run_llama(token)
            match = r.output_hash == self.control_hash
            if not match:
                m.correctness_flag = True
            m.reps.append(r)
            self._record(
                {
                    "stage": stage,
                    "candidate_token": token,
                    "rep": i,
                    "prefill_tok_s": r.prefill_tok_s,
                    "model_load_ms": r.model_load_ms,
                    "output_hash": r.output_hash,
                    "output_match": match,
                }
            )
        self._window.append(token)
        self.since_control += 1
        return m

    def confirm(self, token, reps=5):
        return self.measure(token, reps=reps, stage="confirm")

    def _maybe_control(self):
        if self.since_control < self.control_every:
            return
        self.since_control = 0
        vals = [r.prefill_tok_s for r in self._control_reps(2)]
        med = statistics.median(vals)
        if abs(med - self.baseline_median) / self.baseline_median <= self.drift_frac:
            self._window = []
            return
        # drift ladder: cool down and re-check up to 3 times
        for attempt in range(3):
            self._record({"stage": "drift", "attempt": attempt, "observed": med})
            time.sleep(60)
            vals = [r.prefill_tok_s for r in self._control_reps(3)]
            med = statistics.median(vals)
            if (
                abs(med - self.baseline_median) / self.baseline_median
                <= self.drift_frac
            ):
                self._window = []
                return
        if self.strict:
            raise DriftAbort(
                f"control drifted to {med:.1f} "
                f"(baseline {self.baseline_median:.1f})"
            )
        self.remeasure.extend(self._window)
        self._window = []
        self.baseline_id += 1
        self._record({"stage": "rebaseline", "reason": f"drift to {med:.1f}"})
        self.baseline()


def main():
    ap = argparse.ArgumentParser(description="gate + measure tsweep tokens")
    ap.add_argument("--shader", required=True, choices=("q4gsw", "dq8ca"))
    ap.add_argument("--tokens", required=True, help="comma-separated tsweep tokens")
    ap.add_argument(
        "--stage", default="screen", choices=("screen", "confirm", "validate")
    )
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--pte")
    ap.add_argument("--tokenizer", default=str(TOKENIZER))
    ap.add_argument("--prompt-file", default=str(PROMPT))
    ap.add_argument("--seq-len", type=int, default=3072)
    ap.add_argument("--out")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--skip-gate", action="store_true")
    args = ap.parse_args()

    s = Session(
        args.shader,
        pte=args.pte,
        tokenizer=args.tokenizer,
        prompt_file=args.prompt_file,
        out_jsonl=args.out,
        seq_len=args.seq_len,
        strict=args.strict,
    )
    reps = args.reps or (5 if args.stage in ("confirm", "validate") else 2)
    base, cov = s.baseline()
    print(f"baseline median {base:.1f} tok/s, noise cov {cov*100:.2f}%")
    for tok in args.tokens.split(","):
        tok = tok.strip()
        if not args.skip_gate:
            g = s.gate(tok)
            if g.status != "pass":
                print(f"{tok}\tGATE:{g.status}")
                continue
        m = s.measure(tok, reps=reps, stage=args.stage)
        flag = " MISCOMPUTE" if m.correctness_flag else ""
        print(f"{tok}\t{m.median:.1f}{flag}")
    if s.remeasure:
        print("re-measure (drifted windows):", ",".join(s.remeasure))


if __name__ == "__main__":
    main()
