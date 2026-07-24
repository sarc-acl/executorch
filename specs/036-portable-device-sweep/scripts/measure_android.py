"""Remote-Android sibling of measure.py's Session, for the M51 (Xclipse 970)
target. measure.py assumes a local Vulkan device (subprocess to a
same-machine binary, local `vulkaninfo`); this box only reaches its GPU over
adb via an ssh hop to the adb host. Everything device-independent (dataclasses,
tile_constraints, yaml_variants' file edits) is reused unmodified; only the
"run a binary and read its stdout" and "fingerprint the device" primitives are
reimplemented here against adb.

Fingerprint is hardcoded from .shared-context/instruction-for-ai/hardware/
README.md (probed once via test_coopmat_probe, specs/022) rather than parsed
from a live `vulkaninfo` -- there's no such binary on this Android build.
"""

import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import device_fingerprint as dfp
import yaml_variants as yv
from measure import DriftAbort, GateResult, Measurement, OBSERVER_RE, RepResult

REPO = Path(__file__).resolve().parent.parent.parent.parent
ANDROID_BUILD = REPO / "cmake-out-android-vk"
BENCH = ANDROID_BUILD / "backends/vulkan/test/custom_ops/test_coopmat_linear_bench"
RUNNER = ANDROID_BUILD / "examples/models/llama/llama_main"

HOST = "yanwen.xu@sj1-dmckee-d01"
SERIAL = "0000088f8e579c33"
DEVICE_DIR = "/data/local/tmp/llama_vk"
NFS_RUNNERS = Path("/sarc-c/gpusw/users/yanwen.xu/android-run/runners")

TOKENIZER = f"{DEVICE_DIR}/tokenizer.model"
PROMPT = f"{DEVICE_DIR}/p2048_exact.txt"

DEFAULTS = {
    "q4gsw": {
        "pte": f"{DEVICE_DIR}/llama3_2_1b_4w_buffer_ctx3072.pte",
        "env_var": "ET_VK_Q4GSW_COOPMAT_VARIANT",
        "kernel_base": "linear_q4gsw_coopmat",
    },
    "dq8ca": {
        "pte": f"{DEVICE_DIR}/llama3_2_1b_8da4w_buffer_ctx3072.pte",
        "env_var": "ET_VK_DQ8CA_COOPMAT_VARIANT",
        "kernel_base": "linear_dq8ca_q4gsw_coopmat",
    },
}

# hardware/README.md: probed live 2026-06-08/2026-07-08, test_coopmat_probe.
# min/max subgroup 32/64, shared-mem/wg-invocation limits confirmed the same
# way specs/022's own tile_constraints.py hardcoded them for M5.
M51_FINGERPRINT_BASE = {
    "device_name": "samsung xclipse 970",
    "device_type": "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU",
    "driver_id": "DRIVER_ID_UNKNOWN",  # closed AMD PAL ICD, no VK_KHR_driver_properties string
    "subgroup_size_default": 64,
    "min_subgroup_size": 32,
    "max_subgroup_size": 64,
    "max_compute_shared_memory_size": 65536,
    "max_compute_work_group_invocations": 1024,
}


def _ssh(remote_cmd, timeout=1800):
    return subprocess.run(
        ["ssh", HOST, remote_cmd], capture_output=True, text=True, timeout=timeout
    )


def _adb_shell(inner_cmd, timeout=1800):
    return _ssh(f'adb -s {SERIAL} shell "{inner_cmd}"', timeout=timeout)


def _driver_md5():
    r = _adb_shell("md5sum /vendor/lib64/hw/vulkan.samsung.so", timeout=30)
    return r.stdout.split()[0] if r.stdout.strip() else "unknown"


def _gpu_clock_state():
    r = _adb_shell(
        "cat /sys/class/devfreq/23400000.sgpu/min_freq "
        "/sys/class/devfreq/23400000.sgpu/max_freq",
        timeout=30,
    )
    vals = r.stdout.split()
    if len(vals) == 2 and vals[0] == vals[1]:
        return f"pinned_{vals[0]}"
    return f"floating_{'-'.join(vals)}" if vals else "unknown"


def fingerprint():
    fp = dict(M51_FINGERPRINT_BASE)
    fp["driver_info"] = f"md5={_driver_md5()}"
    fp["os"] = "android arm64 (s5e9975/erd9975)"
    fp["perf_level"] = _gpu_clock_state()
    fp["git_sha"] = dfp._git_sha()
    fp["captured_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return fp


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
    ):
        cfg = DEFAULTS[shader]
        self.shader = shader
        self.pte = pte or cfg["pte"]
        self.tokenizer = tokenizer
        self.prompt_file = prompt_file
        self.env_var = cfg["env_var"]
        self.kernel_base = cfg["kernel_base"]
        self.seq_len = seq_len
        self.control_every = control_every
        self.drift_frac = drift_pct / 100.0
        self.strict = strict
        self.sleep_s = inter_run_sleep_s
        self.fingerprint = fingerprint()
        self.device_slug = dfp.device_slug(self.fingerprint)
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
        self.remeasure = []
        self._window = []
        if not self.out_jsonl.exists() or self.out_jsonl.stat().st_size == 0:
            self._record({"stage": "fingerprint", **self.fingerprint})

    # ---------- low-level runs ----------

    def _run_llama(self, token=None):
        env_prefix = f"{self.env_var}={token} " if token else ""
        cmd = (
            f"cd {DEVICE_DIR} && {env_prefix}./llama_main "
            f"--model_path={self.pte} --tokenizer_path={self.tokenizer} "
            f"--prompt_file={self.prompt_file} --num_bos=1 --temperature=0 "
            f"--max_new_tokens=1 --seq_len={self.seq_len}"
        )
        proc = _adb_shell(cmd, timeout=1800)
        m = OBSERVER_RE.search(proc.stdout)
        if proc.returncode != 0 or not m:
            raise RuntimeError(
                f"llama_main failed (rc={proc.returncode}, token={token}): "
                f"{proc.stdout[-500:]}{proc.stderr[-200:]}"
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

    # ---------- public protocol (mirrors measure.Session) ----------

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
        env = {self.env_var: token, "COOPMAT_BENCH_CORRECTNESS_ONLY": "1"}
        env_prefix = " ".join(f"{k}={v}" for k, v in env.items())
        proc = _adb_shell(
            f"cd {DEVICE_DIR} && {env_prefix} ./test_coopmat_linear_bench"
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
        self._run_llama(token)  # warmup, discarded
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


# ---------- Android cross-compile + push rebuild (replaces yv.rebuild) ----------

# --target install on this build dir hits a stale-arch libflatccrt.a on the
# unrelated executor_runner target and aborts before copying the backend lib
# (setup/README.md "WMMA coopmat on dev" gotcha 5) -- build vulkan_backend and
# copy it into lib/ ourselves (BEFORE the dependent targets below, which link
# against lib/libvulkan_backend.a) instead of going through install.
STEP_BACKEND = [
    "cmake",
    "--build",
    "cmake-out-android-vk",
    "--target",
    "vulkan_backend",
    "--config",
    "Release",
]
STEPS_AFTER_COPY = (
    [
        "cmake",
        "--build",
        "cmake-out-android-vk/backends/vulkan/test/custom_ops",
        "--target",
        "test_coopmat_linear_bench",
        "--config",
        "Release",
    ],
    [
        "cmake",
        "--build",
        "cmake-out-android-vk/examples/models/llama",
        "--config",
        "Release",
    ],
)


def _push_binaries():
    NFS_RUNNERS.mkdir(parents=True, exist_ok=True)
    for src, name in ((BENCH, "test_coopmat_linear_bench"), (RUNNER, "llama_main")):
        staged = NFS_RUNNERS / f"{name}_tsweep_live"
        staged.write_bytes(src.read_bytes())
        r = _ssh(
            f"adb -s {SERIAL} push {staged} {DEVICE_DIR}/{name} && "
            f"adb -s {SERIAL} shell chmod 755 {DEVICE_DIR}/{name}",
            timeout=300,
        )
        if r.returncode != 0:
            raise RuntimeError(f"push failed for {name}: {r.stderr[-500:]}")


def _run_step(step, repo_root, j, log):
    proc = subprocess.run(
        step + ["-j", j], cwd=repo_root, capture_output=True, text=True
    )
    log.append(f"$ {' '.join(step)}\n{proc.stdout}\n{proc.stderr}")
    return proc


def rebuild(repo_root=REPO, jobs=None, log_path=None):
    j = str(jobs or os.cpu_count())
    log = []

    proc = _run_step(STEP_BACKEND, repo_root, j, log)
    if proc.returncode != 0:
        full = "\n".join(log)
        if log_path:
            Path(log_path).write_text(full)
        failed = sorted(set(yv.TOKEN_RE.findall(proc.stdout + proc.stderr)))
        return yv.BuildResult(False, failed, full[-4000:])

    # Copy the freshly-built lib into lib/ BEFORE building the dependents
    # below, which link against lib/libvulkan_backend.a (the "installed"
    # copy), not backends/vulkan/libvulkan_backend.a directly.
    (repo_root / "cmake-out-android-vk/lib/libvulkan_backend.a").write_bytes(
        (
            repo_root / "cmake-out-android-vk/backends/vulkan/libvulkan_backend.a"
        ).read_bytes()
    )

    for step in STEPS_AFTER_COPY:
        proc = _run_step(step, repo_root, j, log)
        if proc.returncode != 0:
            full = "\n".join(log)
            if log_path:
                Path(log_path).write_text(full)
            failed = sorted(set(yv.TOKEN_RE.findall(proc.stdout + proc.stderr)))
            return yv.BuildResult(False, failed, full[-4000:])

    _push_binaries()
    full = "\n".join(log)
    if log_path:
        Path(log_path).write_text(full)
    return yv.BuildResult(True, [], full[-1000:])
