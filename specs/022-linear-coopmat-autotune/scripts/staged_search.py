#!/usr/bin/env python3
"""Staged, successive-halving-style on-device search over the shortlist
(research.md Decision 4/7). See contracts/autotune-report-schema.md #3/#4.

Implementation note (discovered while building this): the bench harness
does not support measuring "just one shape" per invocation -- a single
process invocation with COOPMAT_BENCH_M=<M> always runs the full 12-13
production-shape sweep for all 4 ops AND (unconditionally) the small-shape
correctness matrix. So "Round 1 cheap gate" and "Round 2 full shapes" turn
out to cost the same wall-clock time per invocation; the real staging
benefit is candidate COUNT reduction round to round (25 -> top-third ->
top 3-5), not per-invocation cost reduction. Documented here rather than
silently deviating from the plan's stated per-round cost asymmetry.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

DRIVER_MD5_KNOWN_GOOD = "c9861e9906d03fa2c7d48b804e1a1c80"
PIN_SCRIPT = "/sarc-c/gpusw/users/yanwen.xu/android-run/pin_freqs.sh"

PERF_SUMMARY_RE = re.compile(
    r"^linear_q4gsw\s+\((\d+),(\d+)\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s+(\S+)\s*$",
    re.MULTILINE,
)
CORRECTNESS_ROW_RE = re.compile(
    r"^linear_q4gsw\S*\s.*linear_q4gsw_M256_K256_N256_Buffer\s.*\s(PASSED|FAILED)\s*$",
    re.MULTILINE,
)


def ssh_run(ssh_host, remote_cmd, timeout=300):
    result = subprocess.run(
        ["ssh", ssh_host, remote_cmd], capture_output=True, text=True, timeout=timeout
    )
    return result.returncode, result.stdout, result.stderr


def precheck(ssh_host, serial):
    _, out, _ = ssh_run(
        ssh_host, f"adb -s {serial} shell md5sum /vendor/lib64/hw/vulkan.samsung.so"
    )
    driver_hash = out.split()[0] if out.strip() else None
    _, out2, _ = ssh_run(
        ssh_host, f'adb -s {serial} shell "ps -A | grep -iE \\"llama|coopmat\\"" || true'
    )
    device_busy = bool(out2.strip())
    _, out3, _ = ssh_run(ssh_host, f"bash {PIN_SCRIPT}")
    pinned = "509000" in out3
    ok = (driver_hash == DRIVER_MD5_KNOWN_GOOD) and (not device_busy) and pinned
    return ok, {
        "driver_hash": driver_hash,
        "device_busy": device_busy,
        "clocks_pinned": pinned,
    }


def run_bench(ssh_host, serial, remote_bin, token, m=2048, quick=False):
    remote_dir = str(Path(remote_bin).parent)
    remote_name = Path(remote_bin).name
    quick_env = "COOPMAT_BENCH_QUICK=1 " if quick else ""
    cmd = (
        f"cd {remote_dir} && ET_VK_Q4GSW_COOPMAT_VARIANT={token} "
        f"COOPMAT_BENCH_M={m} {quick_env}./{remote_name}"
    )
    _, out, _ = ssh_run(ssh_host, f'adb -s {serial} shell "{cmd}"', timeout=300)
    return out


def parse_output(stdout):
    gflops_per_shape = {}
    tiled_per_shape = {}
    shader_name = None
    for m in PERF_SUMMARY_RE.finditer(stdout):
        k, n, tiled, coopmat, _ratio, shader = m.groups()
        gflops_per_shape[f"{k},{n}"] = float(coopmat)
        tiled_per_shape[f"{k},{n}"] = float(tiled)
        shader_name = shader
    correctness_match = CORRECTNESS_ROW_RE.search(stdout)
    correctness_status = (
        "pass"
        if correctness_match and correctness_match.group(1) == "PASSED"
        else ("fail" if correctness_match else "unknown")
    )
    return gflops_per_shape, tiled_per_shape, correctness_status, shader_name


def mean_stddev(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return mean, var**0.5


def flop_weighted_mean(gflops_per_shape):
    if not gflops_per_shape:
        return 0.0
    return sum(gflops_per_shape.values()) / len(gflops_per_shape)


def load_budget(out_dir):
    path = Path(out_dir) / "budget.json"
    if path.exists():
        return json.loads(path.read_text())
    return {
        "total_valid_universe": 642,
        "configs_measured_on_hardware": 0,
        "total_device_seconds": 0.0,
        "estimated_exhaustive_device_seconds": 0.0,
        "budget_exceeded": False,
    }


def save_budget(out_dir, budget):
    (Path(out_dir) / "budget.json").write_text(json.dumps(budget, indent=2))


def cmd_round(args, round_name, candidates, run_count_per_candidate, quick=False):
    ok, precheck_info = precheck(args.ssh_host, args.serial)
    out_path = Path(args.out_dir) / f"{round_name}_results.json"
    if not ok:
        out_path.write_text(
            json.dumps(
                [{"halted": True, "halt_reason": precheck_info}], indent=2
            )
        )
        print(f"HALTED before {round_name}: {precheck_info}", flush=True)
        return []

    results = []
    budget = load_budget(args.out_dir)
    import time

    for i, token in enumerate(candidates, start=1):
        gflops_samples = []
        tiled_samples = []
        correctness_status = "unknown"
        shader_name = None
        t0 = time.time()
        print(f"[{round_name}] ({i}/{len(candidates)}) starting {token}...", flush=True)
        for _ in range(run_count_per_candidate):
            stdout = run_bench(
                args.ssh_host, args.serial, args.bench_binary, token, m=2048, quick=quick
            )
            gflops_per_shape, tiled_per_shape, c_status, s_name = parse_output(stdout)
            if gflops_per_shape:
                gflops_samples.append(flop_weighted_mean(gflops_per_shape))
                tiled_samples.append(flop_weighted_mean(tiled_per_shape))
            if c_status != "unknown":
                correctness_status = c_status
            if s_name:
                shader_name = s_name
        elapsed = time.time() - t0
        budget["total_device_seconds"] += elapsed

        mean_g, std_g = (mean_stddev(gflops_samples) if gflops_samples else (0.0, 0.0))
        result = {
            "candidate_token": token,
            "round": round_name,
            "correctness_status": correctness_status,
            "shader_name_seen": shader_name,
            "run_count": len(gflops_samples),
            "mean_gflops": round(mean_g, 2),
            "stddev_gflops": round(std_g, 2),
            "tiled_gflops_ref": round(mean_stddev(tiled_samples)[0], 2) if tiled_samples else 0.0,
            "driver_hash": precheck_info["driver_hash"],
            "clocks_pinned": precheck_info["clocks_pinned"],
        }
        eliminated = correctness_status != "pass" or not gflops_samples
        result["eliminated_at"] = eliminated
        result["elimination_reason"] = (
            "correctness_failed_or_no_data" if eliminated else None
        )
        results.append(result)
        print(
            f"[{round_name}] ({i}/{len(candidates)}) {token}: correctness={correctness_status} "
            f"mean_gflops={mean_g:.1f} (n={len(gflops_samples)}) elapsed={elapsed:.1f}s",
            flush=True,
        )
        # Write incrementally after every candidate, not just at the end --
        # a long round otherwise gives zero visibility into progress until
        # it fully completes (a real gap discovered during Round 1 of this
        # search: 45+ minutes with nothing to inspect but process liveness).
        out_path.write_text(json.dumps(results, indent=2))
        budget["configs_measured_on_hardware"] = len(
            set(_.get("candidate_token") for _ in _all_measured(args.out_dir, results))
        )
        budget["budget_exceeded"] = budget["configs_measured_on_hardware"] > 96
        save_budget(args.out_dir, budget)

    print(f"Wrote {len(results)} results to {out_path}", flush=True)
    return results


def _all_measured(out_dir, current_results):
    tokens = {r["candidate_token"] for r in current_results}
    for prior in ["round1_results", "round2_results", "round3_results"]:
        p = Path(out_dir) / f"{prior}.json"
        if p.exists():
            try:
                data = json.loads(p.read_text())
                tokens |= {r["candidate_token"] for r in data if "candidate_token" in r}
            except Exception:
                pass
    return [{"candidate_token": t} for t in tokens]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortlist")
    parser.add_argument("--bench-binary")
    parser.add_argument("--ssh-host")
    parser.add_argument("--serial")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--round", choices=["1", "2", "3"])
    parser.add_argument("--tokens", nargs="*", help="explicit token list override")
    args = parser.parse_args()

    if args.round == "1":
        shortlist = json.loads(Path(args.shortlist).read_text())
        candidates = args.tokens or [
            r["candidate_token"] for r in shortlist if r["shortlisted"]
        ]
        cmd_round(args, "round1", candidates, run_count_per_candidate=1, quick=True)
    elif args.round == "2":
        r1 = json.loads((Path(args.out_dir) / "round1_results.json").read_text())
        survivors = [r for r in r1 if not r.get("eliminated_at", True)]
        survivors.sort(key=lambda r: r["mean_gflops"], reverse=True)
        top_third_n = max(1, len(survivors) // 3)
        candidates = args.tokens or [r["candidate_token"] for r in survivors[:top_third_n]]
        cmd_round(args, "round2", candidates, run_count_per_candidate=1, quick=True)
    elif args.round == "3":
        r2 = json.loads((Path(args.out_dir) / "round2_results.json").read_text())
        survivors = [r for r in r2 if not r.get("eliminated_at", True)]
        survivors.sort(key=lambda r: r["mean_gflops"], reverse=True)
        candidates = args.tokens or [r["candidate_token"] for r in survivors[:5]]
        # Round 3 uses the FULL (non-quick) binary/shape set -- final
        # confirmation must match the jira-tile-sweep.md methodology
        # exactly (Decision 8), not the quick 3-shape subset.
        cmd_round(args, "round3", candidates, run_count_per_candidate=3, quick=False)


if __name__ == "__main__":
    main()
