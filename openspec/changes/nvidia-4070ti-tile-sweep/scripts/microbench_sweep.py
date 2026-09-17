#!/usr/bin/env python3
"""dbuf4 tile sweep scored by the MICROBENCHMARK, not by e2e tok/s.

specs/036's measure_android.py runs the microbench only as a correctness gate
and takes its perf signal from e2e llama_main. This script is the missing
counterpart: for each dbuf4 tile token it runs test_llama_microbench at the 12
real Llama linear dispatch shapes (llama-3.1-8b / 3.2-3b / 3.2-1b, prefill
M=2048) and records per-shape GFLOP/s.

Scope notes, both verified on device rather than assumed:

  * prefill only. At M=1 the linear op takes the is_gemv short-circuit and
    dispatches linear_*_coop, not the tsweep coopmat variant -- decode rows come
    back dispatch=not_applicable with kernel=linear_q4gsw_coop_..., identical for
    every token. Sweeping decode would measure the same kernel 160 times.

  * buffer only, one scheme per run. A q4gsw token only changes 4w+buffer and a
    dq8ca token only changes 8da4w+buffer; texture3d is the tiled baseline.

  * kernel-time GFLOP/s is the ranking signal, not the case-level mean. For
    8da4w the case mean also covers the activation quantize_and_pack dispatch,
    so mean-based GFLOP/s understates the linear shader and does so unevenly
    across tiles. Both are recorded; `kernel_gflops` is the one to rank on.

Every token's rows are checked against the token that was requested. That guard
is the reason emit() grows a trailing `kernel` field: r.variant is
kernel_class(), which collapses every shader to coopmat/coop/tiled, so without
the full shader name an unrecognized ET_VK_*_COOPMAT_VARIANT token would fall
back to the default kernel silently and still produce a plausible ranking.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # <worktree>/executorch
GLSL = REPO / "backends/vulkan/runtime/graph/ops/glsl"
RESULTS = HERE.parent / "results"

HOST = os.environ.get("SWEEP_ADB_HOST", "yanwen.xu@sj1-dmckee-d01")
SERIAL = os.environ.get("SWEEP_ADB_SERIAL", "0000088f8e579c33")
DEVDIR = os.environ.get("SWEEP_DEV_DIR", "/data/local/tmp/llama_vk")
BINARY = os.environ.get("SWEEP_BINARY", "test_llama_microbench_dbuf4")

SCHEMES = {
    "q4gsw": {
        "label": "4w",
        "env_var": "ET_VK_Q4GSW_COOPMAT_VARIANT",
        "yaml": "linear_q4gsw_coopmat_tsweep_dbuf4.yaml",
        "kernel_stem": "linear_q4gsw_coopmat_",
    },
    "dq8ca": {
        "label": "8da4w",
        "env_var": "ET_VK_DQ8CA_COOPMAT_VARIANT",
        "yaml": "linear_dq8ca_q4gsw_coopmat_tsweep_dbuf4.yaml",
        "kernel_stem": "linear_dq8ca_q4gsw_coopmat_",
    },
}

TOKEN_RE = re.compile(r"tsweep_dbuf4_t\d+x\d+k\d+g\d\ds\d+")
EXIT_RE = re.compile(r"^EXIT:(\d+)", re.M)


def ssh(cmd, timeout=1800):
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", HOST, cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def device_driver_md5():
    r = ssh(f"adb -s {SERIAL} shell md5sum /vendor/lib64/hw/vulkan.samsung.so")
    return r.stdout.split()[0] if r.stdout.split() else "unknown"


def clock_state():
    """Read the pin back from sysfs. Deliberately does NOT trust pin_freqs.sh's
    own verify: it cats /sys/kernel/gpu/{min,max}_freq, which no longer exist
    (renamed to gpu_{min,max}_clock), so its confirmation is inert."""
    cmd = (
        f"adb -s {SERIAL} shell 'cat /sys/class/devfreq/23400000.sgpu/min_freq "
        f"/sys/class/devfreq/23400000.sgpu/max_freq "
        f"/sys/class/devfreq/17000010.devfreq_mif/cur_freq "
        f"/sys/class/devfreq/17000020.devfreq_int/cur_freq'"
    )
    vals = ssh(cmd).stdout.split()
    keys = ["gpu_min", "gpu_max", "mif_cur", "int_cur"]
    return dict(zip(keys, vals))


def tokens_from_yaml(scheme):
    text = (GLSL / SCHEMES[scheme]["yaml"]).read_text()
    return sorted(set(TOKEN_RE.findall(text)))


def binary_md5():
    r = ssh(f"adb -s {SERIAL} shell md5sum {DEVDIR}/{BINARY}")
    return r.stdout.split()[0] if r.stdout.split() else "unknown"


def run_token(scheme, token, group_size, storage="buffer"):
    """One invocation: correctness gate (small shapes, buffer, this scheme
    only) followed by the 12-shape prefill perf pass. The gate is NOT skipped --
    the prior e2e sweep saw 62/112 dq8ca tiles fail correctness, and publishing
    GFLOP/s for a kernel computing garbage is this sweep's main exposure."""
    cfg = SCHEMES[scheme]
    # texture IO needs ET_VK_TEXTURE_COOPMAT: without it QuantizedLinear.cpp
    # keeps the hard kBuffer gate and every token silently falls back to the
    # tiled kernel (the guard below would then flag all 160 variant_mismatch).
    tex_env = "ET_VK_TEXTURE_COOPMAT=1 " if storage == "texture3d" else ""
    inner = (
        f"cd {DEVDIR} && {tex_env}{cfg['env_var']}={token} ./{BINARY} "
        f"--linear --scheme={cfg['label']} --storage={storage} --regime=prefill "
        f"--group-size={group_size} "
        f"2>&1; echo EXIT:$?"
    )
    t0 = time.time()
    proc = ssh(f'adb -s {SERIAL} shell "{inner}"')
    elapsed = round(time.time() - t0, 1)
    out = proc.stdout

    m = EXIT_RE.search(out)
    exit_code = int(m.group(1)) if m else -1

    rec = {
        "token": token,
        "scheme": scheme,
        # provenance in every record: a jsonl that does not state its group
        # size cannot be distinguished from one measured at another, and the
        # group-32 run silently made 71/160 tokens fall back to tiled.
        "group_size": group_size,
        "io_storage": storage,
        "elapsed_s": elapsed,
        "exit_code": exit_code,
        "cases": {},
        "kernel_names": [],
    }

    # A segfault inside the gate is a DIFFERENT failure from a wrong answer:
    # the tile crashes the driver rather than miscomputing. Check it first --
    # a crash can happen after some cases already printed FAILED.
    if "Segmentation fault" in out or "signal 11" in out:
        rec["status"] = "crash"
        rec["failed_cases"] = re.findall(r"\[correctness\] (\S+) FAILED", out)[:20]
        rec["tail"] = out.strip().splitlines()[-8:]
        return rec

    if "correctness gate FAILED" in out:
        rec["status"] = "gate_correctness_fail"
        rec["failed_cases"] = re.findall(r"\[correctness\] (\S+) FAILED", out)[:20]
        return rec

    rows = [ln for ln in out.splitlines() if ln.startswith("RESULT,")]
    if not rows:
        rec["status"] = "no_results"
        rec["tail"] = out.strip().splitlines()[-12:]
        return rec

    mismatched = []
    for ln in rows:
        f = ln.rstrip().split(",")
        if len(f) < 17:
            continue
        # RESULT,suite,model,scheme,regime,variant,K,N,mean_us,stdev_us,
        #        gflops,dispatch,correctness,storage,M,kernel_us,kernel
        model, regime, variant = f[2], f[4], f[5]
        K, N, gflops, dispatch = f[6], f[7], f[10], f[11]
        # storage_f = the ROW's storage; do not shadow the `storage` arg
        storage_f, M, kernel_us, kernel = f[13], f[14], f[15], f[16]
        if regime != "prefill" or storage_f != storage:
            continue
        # The guard: did the runtime actually take the requested variant?
        if token not in kernel:
            mismatched.append(kernel)
        ku = float(kernel_us)
        Mi, Ki, Ni = int(M), int(K), int(N)
        rec["cases"][f"{model}_M{Mi}_K{Ki}_N{Ni}"] = {
            "gflops": float(gflops),
            "kernel_us": ku,
            # rank on this: excludes the 8da4w activation-quant dispatch that
            # the case-level mean folds in.
            "kernel_gflops": (2.0 * Mi * Ni * Ki) / (ku * 1e3) if ku > 0 else -1.0,
            "dispatch": dispatch,
            "variant": variant,
        }
        rec["kernel_names"].append(kernel)

    rec["kernel_names"] = sorted(set(rec["kernel_names"]))
    if mismatched:
        rec["status"] = "variant_mismatch"
        rec["mismatched_kernels"] = sorted(set(mismatched))[:5]
    elif len(rec["cases"]) != 12:
        rec["status"] = "incomplete"
    else:
        rec["status"] = "ok"
    return rec


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return 0.0
    return float(__import__("math").exp(sum(map(__import__("math").log, xs)) / len(xs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=sorted(SCHEMES))
    ap.add_argument("--driver", required=True, help="driver label, e.g. f14c51b6f8")
    ap.add_argument("--limit", type=int, default=0, help="first N tokens (smoke)")
    ap.add_argument("--tokens", nargs="*", help="explicit tokens (canary runs)")
    ap.add_argument("--tag", default="", help="extra filename tag, e.g. canary")
    ap.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="linear quant group size; must match the pte (128)",
    )
    ap.add_argument(
        "--storage",
        default="buffer",
        choices=["buffer", "texture3d"],
        help="IO storage; texture3d also sets ET_VK_TEXTURE_COOPMAT",
    )
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_path = RESULTS / (
        f"microbench_{args.driver}_{args.scheme}_dbuf4_g{args.group_size}"
        f"_{args.storage}{tag}.jsonl"
    )

    tokens = args.tokens or tokens_from_yaml(args.scheme)
    if args.limit:
        tokens = tokens[: args.limit]

    done = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            if ln.strip():
                done.add(json.loads(ln)["token"])
    todo = [t for t in tokens if t not in done]

    md5 = device_driver_md5()
    binmd5 = binary_md5()
    clocks = clock_state()
    print(f"driver on device : {md5}")
    print(f"binary md5       : {binmd5}")
    print(f"group_size       : {args.group_size}")
    print(f"io_storage       : {args.storage}")
    print(f"clocks           : {clocks}")
    print(f"scheme           : {args.scheme} ({SCHEMES[args.scheme]['label']})")
    print(f"tokens           : {len(todo)} to run, {len(done)} already done")
    print(f"out              : {out_path}")
    sys.stdout.flush()

    with out_path.open("a") as fh:
        for i, tok in enumerate(todo, 1):
            rec = run_token(args.scheme, tok, args.group_size, args.storage)
            rec["driver_md5"] = md5
            rec["driver_label"] = args.driver
            rec["clocks"] = clocks
            rec["binary_md5"] = binmd5
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            g = geomean([c["kernel_gflops"] for c in rec["cases"].values()])
            print(
                f"[{i}/{len(todo)}] {tok:42s} {rec['status']:22s} "
                f"geomean={g:8.1f} GFLOP/s  {rec['elapsed_s']}s"
            )
            sys.stdout.flush()

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
