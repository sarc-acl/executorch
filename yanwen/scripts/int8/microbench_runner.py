#!/usr/bin/env python3
"""
Phase 2: drive the pavan-report custom_ops microbench binaries at LLaMA shapes
and capture per-shader timings.

Targets driven (paths assume pavan-report tree built at cmake-out-vk/):
  - khr_cm_gemm_int8       (int8 KHR coopmat GEMM)
  - linear_coopmat_bench   (fp32+fp16, linear_vec vs linear_coopmat)
  - q8csw_linear           (int8 W8A8 + W8A16 non-coopmat)

The bench targets were extended in-place to add the LLaMA prefill shape configs
(M=128 with K/N matching FFN gate/up/down, Q/O, K/V). See the cpp files for the
exact configs.

Outputs full stdout/stderr per binary into:
  yanwen/artifacts/int8_microbench/{binary}_{timestamp}.log

Followup: run microbench_summarize.py to extract LLaMA-row timings and compute
the comparison ratios.

Usage:
    cd /home/doremy/sarc-acl/executorch/main/executorch
    python yanwen/scripts/int8/microbench_runner.py
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PAVAN_ROOT = Path("/home/doremy/sarc-acl/executorch/pavan-report/executorch")
BIN_DIR = PAVAN_ROOT / "cmake-out-vk" / "backends" / "vulkan" / "test" / "custom_ops"

BINARIES = [
    "khr_cm_gemm_int8",
    "linear_coopmat_bench",
    "q8csw_linear",
]

DEFAULT_OUT = Path(
    "/home/doremy/sarc-acl/executorch/main/executorch/yanwen/artifacts/int8_microbench"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--binaries",
        nargs="+",
        default=BINARIES,
        help="Subset of bench binaries to run (defaults to all 3).",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "RADV_GTT_PCT": "80"}
    ts = time.strftime("%Y%m%d_%H%M%S")

    failed = []
    for binary in args.binaries:
        path = BIN_DIR / binary
        if not path.exists():
            print(
                f"[runner] MISSING {path} — build it first via cmake --target {binary}"
            )
            failed.append(binary)
            continue

        log_path = args.out_dir / f"{binary}_{ts}.log"
        print(f"\n[runner] === {binary} ===")
        print(f"[runner] writing -> {log_path}")
        t0 = time.perf_counter()
        with open(log_path, "w") as lf:
            lf.write(
                f"# {binary} run @ {ts}\n# cwd={path.parent}\n# env RADV_GTT_PCT=80\n\n"
            )
            proc = subprocess.run(
                [str(path)],
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                check=False,
            )
        dt = time.perf_counter() - t0
        rc = proc.returncode
        print(f"[runner] rc={rc}  wallclock={dt:.1f}s")
        if rc != 0:
            failed.append(binary)
            print(f"[runner]   tail of {log_path}:")
            os.system(f"tail -20 {log_path}")

    print("\n[runner] done.")
    print(f"[runner] logs in {args.out_dir}/")
    if failed:
        print(f"[runner] FAILED: {failed}")
        sys.exit(1)
    print("[runner] next: python microbench_summarize.py")


if __name__ == "__main__":
    main()
