#!/usr/bin/env python3
"""M51 buffer-storage coopmat A/B at max clock: baseline (ET_VK_DISABLE_COOPMAT=1) vs
wmma (coopmat default-on), holding storage=buffer constant so only dispatch varies.

Matrix: {1B, 3B, 8B} x {4w, 8da4w} x {baseline, wmma} x 3 reps, all on buffer-storage
ctx3072 PTEs, full 2048-prefill + 1024-decode, clocks PINNED AT HARDWARE MAX
(980000/5333000/934000 GPU/MIF/INT) -- not the usual 509 default pin.

Adapted from .shared-context/scripts/run_m5_full_sweep.py (reused: sh/ssh/adb helpers,
Status class, tagged logging, read_clocks, stage_pte's size-verify branch,
load_done/append_result resumable JSONL, run_one's heartbeat+timeout+crash
classification). Deliberately different from that script:
  - No push_binary_once: llama_main_origcm is already staged on-device (dev-branch
    runner with the WMMA coopmat port); we only verify it's present+executable.
  - Only ever touches BUFFER-storage PTEs -- no texture branch at all. The axis this
    script tests is coopmat dispatch (env var), not storage type.
  - New "maxpin" clock mode (980000/5333000/934000), same sysfs set+verify pattern as
    the existing "pinned" (509) mode.
  - Interleaving is scoped PER MODEL: for a given model, 3 rounds each running all 4
    (scheme, config) combos once, in order -- never all-of-baseline-then-all-of-wmma
    (result-and-report/README.md row 5). Then that model's PTEs are removed before
    moving to the next model (bounds on-device storage/page-cache pressure).
  - ET_VK_EXECUTE_NODE_THRESHOLD=32 is set on EVERY run (both configs, all models) --
    holding it constant is what keeps "only coopmat varies" true between the two arms;
    it is also the confirmed fix for the 8B prefill GPU-watchdog crash.

Usage:
  .../venv/bin/python run_maxclock_ab.py --results ../results/maxclock_ab.jsonl \\
      --status-file ../results/maxclock_ab.status.json
  # Resume (already-done combos skipped): re-run with the same --results path.
  # Just re-check 8B: --models 8b
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

HOST = "yanwen.xu@sj1-dmckee-d01"
SERIAL = "0000088f8e579c33"
DEVICE_DIR = "/data/local/tmp/llama_vk"
PTE_OUT = Path("/local/yanwen.xu/workspace/.pte_out")
BINARY = (
    "llama_main_origcm"  # dev-branch runner with the WMMA coopmat port, already staged
)

# hash -> label, per instruction-for-ai/access-and-run/README.md §6.
KNOWN_GOOD_DRIVERS = {
    "c9861e9906d03fa2c7d48b804e1a1c80": "f14c51b6f8 (27.0.1271, documented production default)",
}
REQUIRED_DRIVER_MD5 = (
    "c9861e9906d03fa2c7d48b804e1a1c80"  # this run requires the DEFAULT specifically
)
REFLASH_SO_PATH = "/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so"

MODELS = ["1b", "3b", "8b"]
MODEL_NAMES = {"1b": "llama3_2_1b", "3b": "llama3_2_3b", "8b": "llama3_1_8b"}
SCHEMES = ["4w", "8da4w"]
CONFIGS = [
    "baseline",
    "wmma",
]  # baseline=ET_VK_DISABLE_COOPMAT=1, wmma=coopmat default-on
REPS = 3

MAX_NEW_TOKENS = 1024
PROMPT_FILE = "p2048_exact.txt"
TOKENIZER_FILE = "tokenizer.model"
COHERENCE_PROMPT = "The capital of France is"
# 8B baseline (tiled) decode can be slow even at max clock -- generous per-run timeout.
RUN_TIMEOUT_S = 40 * 60
HEARTBEAT_INTERVAL_S = 30
GPU_RECOVERY_SLEEP_S = 15
MAX_RETRIES = 1

CRASH_SIGNATURES = {
    "device_lost": ["VK_ERROR_DEVICE_LOST", "DEVICE_LOST"],
    "abort": ["Aborted", "libc++abi", "terminate called"],
    "segfault": ["Segmentation fault"],
}

_START_TS = time.time()


def log(tag, msg):
    elapsed = time.time() - _START_TS
    print(f"[{time.strftime('%H:%M:%S')}] [+{elapsed:7.1f}s] [{tag}] {msg}", flush=True)


def bar(done, total, width=24):
    filled = int(width * done / total) if total else 0
    return "[" + "#" * filled + "." * (width - filled) + "]"


class Status:
    def __init__(self, path):
        self.path = path
        self.state = {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "phase": "starting",
            "current_combo": None,
            "current_combo_elapsed_s": None,
            "counts": {"total": 0, "done": 0, "ok": 0, "failed": 0, "remaining": 0},
            "last_event": None,
            "last_error": None,
            "eta_s": None,
        }
        self._write()

    def update(self, **kwargs):
        self.state.update(kwargs)
        self._write()

    def _write(self):
        if self.path is None:
            return
        self.path.write_text(json.dumps(self.state, indent=2))


def sh(cmd, timeout=60, check=True):
    return subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=check,
    )


def ssh(remote_cmd, timeout=60, check=True):
    return sh(
        f"ssh -o ConnectTimeout=15 {HOST} {json.dumps(remote_cmd)}",
        timeout=timeout,
        check=check,
    )


def adb(shell_cmd, timeout=60, check=True):
    return ssh(
        f"adb -s {SERIAL} shell {json.dumps(shell_cmd)}", timeout=timeout, check=check
    )


# --- pre-flight ---------------------------------------------------------------


def check_device_free(auto_proceed):
    out = ssh(f"adb -s {SERIAL} shell ps -A | grep -i llama_main || true").stdout
    if out.strip():
        log("WARN", f"a llama_main-family process is already running on-device:\n{out}")
        if not auto_proceed:
            resp = input("Proceed anyway? [y/N] ")
            if resp.lower() not in ("y", "yes"):
                sys.exit("Aborted -- device appears to be in use.")
    else:
        log("DEVICE", "idle, no llama_main process running.")


def check_driver(auto_reflash):
    out = ssh(f"adb -s {SERIAL} shell md5sum /vendor/lib64/hw/vulkan.samsung.so").stdout
    md5 = out.strip().split()[0]
    if md5 == REQUIRED_DRIVER_MD5:
        log("DRIVER", f"OK: {md5} -> {KNOWN_GOOD_DRIVERS[md5]}")
        return
    log(
        "WARN",
        f"driver {md5} is not the required production default {REQUIRED_DRIVER_MD5}.",
    )
    if not auto_reflash:
        resp = input("Reflash to f14c51b6f8 now? [y/N] ")
        if resp.lower() not in ("y", "yes"):
            sys.exit("Aborted -- wrong driver, not reflashing without confirmation.")
    log("DRIVER", "reflashing to f14c51b6f8 from NFS backup...")
    ssh(f"adb -s {SERIAL} root", timeout=30)
    time.sleep(1)
    ssh(f"adb -s {SERIAL} remount", timeout=30)
    ssh(f"adb -s {SERIAL} shell setenforce 0")
    ssh(f"adb -s {SERIAL} shell stop")
    ssh(
        f"adb -s {SERIAL} push {REFLASH_SO_PATH} /vendor/lib64/hw/vulkan.samsung.so",
        timeout=60,
    )
    ssh(f"adb -s {SERIAL} shell chmod 644 /vendor/lib64/hw/vulkan.samsung.so")
    ssh(f"adb -s {SERIAL} remount", timeout=30)
    ssh(f"adb -s {SERIAL} shell start")
    ssh(f"adb -s {SERIAL} shell setenforce 1")
    time.sleep(6)
    out = ssh(f"adb -s {SERIAL} shell md5sum /vendor/lib64/hw/vulkan.samsung.so").stdout
    md5 = out.strip().split()[0]
    if md5 != REQUIRED_DRIVER_MD5:
        sys.exit(f"ABORT: reflash did not produce the required driver (got {md5}).")
    log("DRIVER", f"reflash OK: {md5} -> {KNOWN_GOOD_DRIVERS[md5]}")


def check_binary_present():
    r = adb(f"test -x {DEVICE_DIR}/{BINARY} && echo OK", check=False)
    if "OK" not in r.stdout:
        sys.exit(f"ABORT: {DEVICE_DIR}/{BINARY} missing or not executable on-device.")
    log("DEVICE", f"{BINARY} present and executable.")


def run_correctness_bench():
    r = adb(
        f"cd {DEVICE_DIR} && COOPMAT_BENCH_CORRECTNESS_ONLY=1 ./test_coopmat_linear_bench_origcm",
        timeout=90,
        check=False,
    )
    passed = r.stdout.count("PASSED")
    failed = r.stdout.count("FAILED")
    log("CORRECTNESS", f"{passed} PASSED, {failed} FAILED")
    if failed > 0 or passed < 16:
        sys.exit(
            f"ABORT: correctness bench not clean ({passed} passed / {failed} failed)."
        )


# --- clocks --------------------------------------------------------------------

PINNED_509 = {"gpu": 509000, "mif": 2730000, "int": 663000}
MAXPIN = {"gpu": 980000, "mif": 5333000, "int": 934000}


def read_clocks():
    out = ssh(
        f"adb -s {SERIAL} shell 'cat /sys/class/devfreq/23400000.sgpu/min_freq "
        f"/sys/class/devfreq/23400000.sgpu/max_freq "
        f"/sys/class/devfreq/17000010.devfreq_mif/scaling_devfreq_min "
        f"/sys/class/devfreq/17000010.devfreq_mif/scaling_devfreq_max "
        f"/sys/class/devfreq/17000020.devfreq_int/scaling_devfreq_min "
        f"/sys/class/devfreq/17000020.devfreq_int/scaling_devfreq_max'"
    ).stdout.split()
    keys = ["gpu_min", "gpu_max", "mif_min", "mif_max", "int_min", "int_max"]
    return dict(zip(keys, (int(v) for v in out)))


def set_clocks(mode):
    target = MAXPIN if mode == "maxpin" else PINNED_509
    ssh(f"adb -s {SERIAL} root", timeout=30, check=False)
    time.sleep(1)
    writes = {
        "/sys/class/devfreq/23400000.sgpu/min_freq": target["gpu"],
        "/sys/class/devfreq/23400000.sgpu/max_freq": target["gpu"],
        "/sys/class/devfreq/17000010.devfreq_mif/scaling_devfreq_min": target["mif"],
        "/sys/class/devfreq/17000010.devfreq_mif/scaling_devfreq_max": target["mif"],
        "/sys/class/devfreq/17000020.devfreq_int/scaling_devfreq_min": target["int"],
        "/sys/class/devfreq/17000020.devfreq_int/scaling_devfreq_max": target["int"],
    }
    for node, val in writes.items():
        ssh(f"adb -s {SERIAL} shell 'echo {val} > {node}'", check=False)

    state = read_clocks()
    ok = (
        state["gpu_min"] == state["gpu_max"] == target["gpu"]
        and state["mif_min"] == state["mif_max"] == target["mif"]
        and state["int_min"] == state["int_max"] == target["int"]
    )
    if not ok:
        raise RuntimeError(
            f"Clock verification failed for mode={mode}: sysfs reads back {state}"
        )
    log("CLOCK", f"verified {mode}: {state}")


# --- PTE staging (size-verify only -- all 6 buffer PTEs already staged) -------


def local_pte_name(model, scheme):
    return f"{MODEL_NAMES[model]}_{scheme}_buffer_ctx3072.pte"


def device_file_size(remote_path):
    r = adb(f"stat -c%s {remote_path}", check=False)
    if r.returncode != 0:
        return None
    return int(r.stdout.strip())


def ensure_pte_staged(model, scheme):
    name = local_pte_name(model, scheme)
    local_path = PTE_OUT / name
    if not local_path.exists():
        raise FileNotFoundError(f"{local_path} does not exist -- export it first.")
    local_size = local_path.stat().st_size
    remote_path = f"{DEVICE_DIR}/{name}"
    existing = device_file_size(remote_path)
    if existing == local_size:
        log("STAGE", f"{name} already staged and size-verified ({local_size} bytes).")
        return remote_path
    log("STAGE", f"staging {name} ({local_size / 1e9:.2f} GB)...")
    host_tmp_free = int(ssh("df --output=avail /tmp | tail -1").stdout.strip()) * 1024
    if host_tmp_free < local_size * 1.2:
        raise RuntimeError(
            f"adb host /tmp only has {host_tmp_free / 1e9:.1f} GB free, need ~"
            f"{local_size * 1.2 / 1e9:.1f} GB headroom for {name}."
        )
    ssh(f"scp -o ConnectTimeout=15 {local_path} {HOST}:/tmp/{name}", timeout=600)
    ssh(f"adb -s {SERIAL} push /tmp/{name} {remote_path}", timeout=600)
    ssh(f"rm -f /tmp/{name}")
    on_device_size = device_file_size(remote_path)
    if on_device_size != local_size:
        adb(f"rm -f {remote_path}")
        raise RuntimeError(
            f"Push verification FAILED for {name}: local={local_size} on-device={on_device_size}."
        )
    log("STAGE", f"{name} staged and size-verified.")
    return remote_path


def remove_model_ptes(model):
    for scheme in SCHEMES:
        name = local_pte_name(model, scheme)
        adb(f"rm -f {DEVICE_DIR}/{name}", check=False)
    log("STAGE", f"removed {model}'s buffer PTEs from device.")


# --- device health checks -----------------------------------------------------


def check_device_headroom(min_avail_gb=2.0):
    out = ssh(f"adb -s {SERIAL} shell cat /proc/meminfo | head -3").stdout
    m = re.search(r"MemAvailable:\s+(\d+)", out)
    avail_gb = int(m.group(1)) / 1024 / 1024 if m else None
    if avail_gb is not None and avail_gb < min_avail_gb:
        log(
            "WARN",
            f"MemAvailable={avail_gb:.2f} GB, below the {min_avail_gb} GB caution threshold.",
        )
    else:
        log("DEVICE", f"MemAvailable={avail_gb:.2f} GB -- OK")
    return avail_gb


def check_oom_killed_recently(since_epoch_s):
    out = ssh(
        f"adb -s {SERIAL} shell dmesg -T 2>/dev/null | grep -i 'killed process' || true",
        check=False,
    ).stdout
    candidates = [
        line
        for line in out.strip().splitlines()
        if BINARY in line or "llama_main" in line
    ]
    if not candidates:
        return None
    last = candidates[-1]
    m = re.search(r"\[(\w+ \w+\s+\d+ \d+:\d+:\d+ \d+)\]", last)
    if m:
        try:
            kill_epoch = time.mktime(time.strptime(m.group(1), "%a %b %d %H:%M:%S %Y"))
            if kill_epoch < since_epoch_s - 60:
                return None
        except ValueError:
            pass
    return last


# --- coherence check -----------------------------------------------------------


def coherence_check(model, scheme, config, pte_remote_path):
    env = {"ET_VK_EXECUTE_NODE_THRESHOLD": "32"}
    if config == "baseline":
        env["ET_VK_DISABLE_COOPMAT"] = "1"
    env_str = " ".join(f"{k}={v}" for k, v in env.items())
    cmd = (
        f"cd {DEVICE_DIR} && {env_str} ./{BINARY} --model_path={pte_remote_path} "
        f"--tokenizer_path={DEVICE_DIR}/{TOKENIZER_FILE} "
        f"--prompt='{COHERENCE_PROMPT}' --seq_len=48 --temperature=0 --warmup=false"
    )
    r = adb(cmd, timeout=60, check=False)
    ok = "Paris" in r.stdout
    label = f"{model}/{scheme}/{config}"
    if ok:
        log("COHERENCE", f"{label}: OK (Paris)")
    else:
        log("COHERENCE", f"{label}: FAILED -- tail: {r.stdout[-300:]}")
    return ok


# --- running one rep -----------------------------------------------------------


def classify_failure(stdout_tail, oom_line):
    if oom_line:
        return "oom_killed"
    for cls, sigs in CRASH_SIGNATURES.items():
        if any(s in stdout_tail for s in sigs):
            return f"crash_{cls}"
    if not stdout_tail.strip():
        return "no_output"
    return "unknown"


def is_transient(failure_class):
    return failure_class in ("timeout", "oom_killed", "no_output")


def run_one(model, scheme, config, rep, pte_remote_path, status):
    env = {"ET_VK_EXECUTE_NODE_THRESHOLD": "32"}
    if config == "baseline":
        env["ET_VK_DISABLE_COOPMAT"] = "1"
    env_str = " ".join(f"{k}={v}" for k, v in env.items())
    out_file = f"{DEVICE_DIR}/maxab_out_{model}_{scheme}_{config}_r{rep}.log"
    combo_label = f"{model}/{scheme}/{config} rep{rep}"

    cmd = (
        f"cd {DEVICE_DIR} && {env_str} ./{BINARY} "
        f"--model_path={pte_remote_path} --tokenizer_path={DEVICE_DIR}/{TOKENIZER_FILE} "
        f"--prompt_file={DEVICE_DIR}/{PROMPT_FILE} --num_bos=1 "
        f"--max_new_tokens={MAX_NEW_TOKENS} --ignore_eos --temperature=0 "
        f"--warmup=true > {out_file} 2>&1"
    )
    start_epoch = time.time()
    proc = subprocess.Popen(
        f"ssh -o ConnectTimeout=15 {HOST} {json.dumps(f'adb -s {SERIAL} shell {json.dumps(cmd)}')}",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    last_heartbeat = start_epoch
    timed_out = False
    while proc.poll() is None:
        time.sleep(1)
        now = time.time()
        elapsed = now - start_epoch
        if now - last_heartbeat >= HEARTBEAT_INTERVAL_S:
            log("HEARTBEAT", f"{combo_label} still running, {elapsed:.0f}s elapsed")
            status.update(
                current_combo=combo_label, current_combo_elapsed_s=round(elapsed, 1)
            )
            last_heartbeat = now
        if elapsed > RUN_TIMEOUT_S:
            timed_out = True
            proc.kill()
            ssh(f"adb -s {SERIAL} shell pkill -f {BINARY}", check=False)
            break

    if timed_out:
        return {
            "ok": False,
            "failure_class": "timeout",
            "elapsed_s": round(time.time() - start_epoch, 1),
        }

    r = ssh(f"adb -s {SERIAL} shell cat {out_file}", check=False)
    stdout = r.stdout
    m = re.search(
        r'"prompt_tokens":(\d+).*?"generated_tokens":(\d+).*?'
        r'"aggregate_sampling_time_ms"',
        stdout,
    )
    m2 = re.search(
        r'"prefill_token_per_sec":([\d.]+).*?"decode_token_per_sec":([\d.]+)', stdout
    )
    elapsed_s = round(time.time() - start_epoch, 1)
    if m2:
        prefill, decode = float(m2.group(1)), float(m2.group(2))
        prompt_tokens = int(m.group(1)) if m else None
        gen_tokens = int(m.group(2)) if m else None
        valid = (
            gen_tokens is not None
            and gen_tokens >= MAX_NEW_TOKENS - 5
            and prompt_tokens == 2048
        )
        if valid:
            return {
                "ok": True,
                "prefill_tok_s": prefill,
                "decode_tok_s": decode,
                "generated_tokens": gen_tokens,
                "prompt_tokens": prompt_tokens,
                "elapsed_s": elapsed_s,
            }
        return {
            "ok": False,
            "failure_class": "short_generation_or_bad_prompt_tokens",
            "generated_tokens": gen_tokens,
            "prompt_tokens": prompt_tokens,
            "elapsed_s": elapsed_s,
        }

    oom_line = check_oom_killed_recently(start_epoch)
    failure_class = classify_failure(stdout, oom_line)
    return {
        "ok": False,
        "failure_class": failure_class,
        "elapsed_s": elapsed_s,
        "oom_line": oom_line,
        "raw_tail": stdout[-500:],
    }


def device_responsive():
    r = ssh(f"adb -s {SERIAL} get-state", check=False)
    return r.returncode == 0 and "device" in r.stdout


# --- results log (resumable) --------------------------------------------------


def load_done(results_path):
    done = set()
    if not results_path.exists():
        return done
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("ok"):
            done.add((row["model"], row["scheme"], row["config"], row["rep"]))
    return done


def append_result(results_path, row):
    with results_path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def print_summary(results_path):
    rows = [
        json.loads(line)
        for line in results_path.read_text().splitlines()
        if line.strip()
    ]
    by_combo = {}
    for r in rows:
        if not r.get("ok"):
            continue
        key = (r["model"], r["scheme"], r["config"])
        by_combo.setdefault(key, []).append((r["prefill_tok_s"], r["decode_tok_s"]))
    log("SUMMARY", "median prefill/decode tok/s per combo:")
    for key in sorted(by_combo):
        vals = by_combo[key]
        pre = statistics.median(v[0] for v in vals)
        dec = statistics.median(v[1] for v in vals)
        print(
            f"    {key[0]:>3}/{key[1]:<6} {key[2]:<8}  prefill_med={pre:8.2f} decode_med={dec:6.2f}  n={len(vals)}"
        )


# --- main sweep ----------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--status-file", type=Path, default=None)
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--schemes", nargs="+", default=SCHEMES, choices=SCHEMES)
    p.add_argument("--configs", nargs="+", default=CONFIGS, choices=CONFIGS)
    p.add_argument("--reps", type=int, default=REPS)
    p.add_argument("--auto-reflash", action="store_true")
    p.add_argument("--auto-proceed", action="store_true")
    p.add_argument(
        "--skip-coherence", action="store_true", help="skip the one-time coherence gate"
    )
    p.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    combos = [
        (s, c) for s in args.schemes for c in args.configs
    ]  # 4 combos/model, in fixed order
    matrix = [
        (m, s, c, r)
        for m in args.models
        for (s, c) in combos
        for r in range(1, args.reps + 1)
    ]
    if args.dry_run:
        print(
            f"{len(matrix)} planned runs across {len(args.models)} model(s), {args.reps} rounds:"
        )
        for m in args.models:
            print(f"  {m}: round order = {combos}")
        return

    args.results.parent.mkdir(parents=True, exist_ok=True)
    status = Status(args.status_file)
    done = load_done(args.results)
    remaining = [k for k in matrix if k not in done]
    log(
        "PLAN",
        f"{len(matrix)} total combos, {len(done)} already done, {len(remaining)} remaining.",
    )
    status.update(
        phase="preflight",
        counts={
            "total": len(matrix),
            "done": len(done),
            "ok": len(done),
            "failed": 0,
            "remaining": len(remaining),
        },
    )

    check_device_free(args.auto_proceed)
    check_driver(args.auto_reflash)
    check_binary_present()
    run_correctness_bench()
    log("CLOCK", "setting maxpin (980000/5333000/934000) for the whole sweep...")
    set_clocks("maxpin")

    completed = len(done)
    failed_count = 0
    rep_durations = []
    checked_coherence = set()

    for model in args.models:
        log("MODEL", f"=== starting {model} ===")
        pte_paths = {}
        for scheme in args.schemes:
            pte_paths[scheme] = ensure_pte_staged(model, scheme)
        check_device_headroom()

        if not args.skip_coherence:
            for scheme, config in combos:
                key = (model, scheme, config)
                if key in checked_coherence:
                    continue
                ok = coherence_check(model, scheme, config, pte_paths[scheme])
                checked_coherence.add(key)
                if not ok:
                    log(
                        "WARN",
                        f"{model}/{scheme}/{config} failed coherence -- reps for this combo will still be attempted and recorded, but treat results with suspicion.",
                    )

        for rep in range(1, args.reps + 1):
            for (
                scheme,
                config,
            ) in combos:  # round-interleaved: all 4 combos once per round
                key = (model, scheme, config, rep)
                if key in done:
                    continue
                combo_label = f"{model}/{scheme}/{config} rep{rep}"
                eta_s = (
                    statistics.mean(rep_durations) * len(remaining)
                    if rep_durations
                    else None
                )
                log(
                    "PROGRESS",
                    f"{bar(completed, len(matrix))} {completed}/{len(matrix)} done "
                    f"({failed_count} failed) | next: {combo_label}"
                    + (f" | ETA ~{eta_s / 60:.0f} min" if eta_s else ""),
                )
                status.update(
                    phase="running",
                    current_combo=combo_label,
                    current_combo_elapsed_s=0,
                    counts={
                        "total": len(matrix),
                        "done": completed,
                        "ok": completed - failed_count,
                        "failed": failed_count,
                        "remaining": len(matrix) - completed,
                    },
                    eta_s=eta_s,
                )

                attempt = 0
                while True:
                    log("RUN", f"{combo_label} (attempt {attempt + 1})...")
                    result = run_one(
                        model, scheme, config, rep, pte_paths[scheme], status
                    )
                    if (
                        result["ok"]
                        or not is_transient(result.get("failure_class"))
                        or attempt >= args.max_retries
                    ):
                        break
                    attempt += 1
                    log(
                        "RETRY",
                        f"{combo_label}: {result.get('failure_class')} -- retrying ({attempt}/{args.max_retries})",
                    )
                    time.sleep(GPU_RECOVERY_SLEEP_S)

                row = {
                    "model": model,
                    "scheme": scheme,
                    "config": config,
                    "rep": rep,
                    **result,
                }
                append_result(args.results, row)
                completed += 1
                rep_durations.append(result.get("elapsed_s", GPU_RECOVERY_SLEEP_S))

                if result["ok"]:
                    log(
                        "OK",
                        f"{combo_label} -> prefill={result['prefill_tok_s']:.2f} decode={result['decode_tok_s']:.2f} ({result['elapsed_s']:.0f}s)",
                    )
                    status.update(last_event=f"OK {combo_label}", last_error=None)
                else:
                    failed_count += 1
                    fc = result.get("failure_class")
                    tag = (
                        "CRASH"
                        if (fc and fc.startswith("crash")) or fc == "oom_killed"
                        else "FAIL"
                    )
                    log(
                        tag,
                        f"{combo_label} -> {fc}"
                        + (
                            f" (dmesg: {result['oom_line']})"
                            if result.get("oom_line")
                            else ""
                        ),
                    )
                    status.update(last_event=f"{tag} {combo_label}", last_error=fc)
                    if not device_responsive():
                        log(
                            "WARN",
                            "device unresponsive after failure -- waiting 30s and re-checking...",
                        )
                        time.sleep(30)
                        if not device_responsive():
                            status.update(
                                phase="aborted", last_error="device unresponsive"
                            )
                            sys.exit(
                                f"[ABORT] device unresponsive. Re-run with the same --results {args.results} to resume."
                            )
                        log("DEVICE", "responsive again -- continuing.")

                time.sleep(GPU_RECOVERY_SLEEP_S)

        remove_model_ptes(model)
        log("MODEL", f"=== finished {model} ===")

    log("CLOCK", "sweep complete. Restoring 509 default pin...")
    set_clocks("pinned")
    print_summary(args.results)
    status.update(phase="done", current_combo=None, last_event="sweep complete")
    log(
        "DONE",
        f"results in {args.results} -- {len(load_done(args.results))} combos done, {failed_count} failed.",
    )


if __name__ == "__main__":
    main()
