#!/usr/bin/env python3
"""
Phase 2 (+ optional 3): benchmark a pre-exported .pte through executor_runner.
Requires that setup_llama31_pure.py has already produced the .pte.

Default mode (scientific): runs 1 calibration subprocess (N=1) + 3 measurement
subprocesses (N=8 each), and uses (WK - W1) / (K-1) per rep to subtract off
load + iter 0 + teardown. Reports steady-state forward as mean ± stdev.

Why algebraic subtraction:
  - ETDump per-iter samples don't capture GPU sync time on the Vulkan delegate
    (off by ~270x vs wallclock); unusable for per-iter timing.
  - Single-subprocess wallclock/N still folds in load + iter-0 cold-start.
  - Differential cancels both cleanly.

Legacy: --num_executions N → old wallclock/N path.

Usage:
    source /home/doremy/sarc-acl/executorch/main/executorch/.venv/bin/activate
    python bench_llama31_pure.py --n_layers 32 --seq_len 128            # default
    python bench_llama31_pure.py --n_layers 32 --seq_len 128 --reps 5 --iters 16
    python bench_llama31_pure.py --n_layers 32 --seq_len 128 --num_executions 16  # legacy
    python bench_llama31_pure.py --n_layers 32 --seq_len 128 --etdump-analyze     # + per-op
"""

import argparse
import gc
import sys
from pathlib import Path

from run_llama31_pure import (
    _ensure_venv_path,
    _parent_oom_hardening,
    analyze,
    bench_steady_state,
    DEFAULT_OUT,
    env_check,
    run_etdump,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_layers", type=int, default=32, choices=range(1, 33))
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument(
        "--reps", type=int, default=3, help="Measurement subprocess count (default 3)."
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=8,
        help="Iterations per measurement subprocess (default 8).",
    )
    ap.add_argument(
        "--num_executions",
        type=int,
        default=None,
        help="Legacy wallclock/N mode. If set, overrides --reps/--iters.",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-swap-check", action="store_true")
    ap.add_argument(
        "--etdump-analyze",
        action="store_true",
        help="Capture per-op ETDump on the last measurement run "
        "and run Inspector for the per-op breakdown.",
    )
    args = ap.parse_args()

    _parent_oom_hardening()
    _ensure_venv_path()
    env_check(args.skip_swap_check)

    tag = f"llama31_8b_{args.n_layers}L_seq{args.seq_len}_fp16"
    pte_path = args.out_dir / f"{tag}.pte"
    etrecord_path = args.out_dir / f"{tag}.etrecord.bin"
    input_path = args.out_dir / f"{tag}_input0.bin"
    etdump_path = args.out_dir / f"{tag}.etdp"
    mem_log = args.out_dir / f"{tag}.memprobe.tsv"
    tsv_path = args.out_dir / f"{tag}.events.tsv"

    if not pte_path.exists():
        sys.exit(
            f"[bench] missing {pte_path}\n"
            f"        run: python setup_llama31_pure.py "
            f"--n_layers {args.n_layers} --seq_len {args.seq_len}"
        )
    if not input_path.exists():
        sys.exit(f"[bench] missing {input_path} — re-run setup to regenerate.")

    gc.collect()

    if args.num_executions is not None:
        _, probe = run_etdump(
            pte_path,
            input_path,
            etdump_path,
            args.num_executions,
            mem_log,
            want_etdump=args.etdump_analyze,
        )
        if probe is None:
            sys.exit(3)
    else:
        result = bench_steady_state(
            pte_path,
            input_path,
            etdump_path,
            mem_log,
            n_reps=args.reps,
            n_iters_per_rep=args.iters,
        )
        if result is None:
            sys.exit(3)
        probe = None

    if args.etdump_analyze:
        if not etdump_path.exists():
            sys.exit(
                f"[bench] expected etdump at {etdump_path} but missing — "
                f"executor_runner may have been built without "
                f"EXECUTORCH_ENABLE_EVENT_TRACER."
            )
        etr = etrecord_path if etrecord_path.exists() else None
        analyze(etdump_path, etr, tsv_path, probe)

    print(f"\n[bench] artifacts in {args.out_dir}/{tag}.*")


if __name__ == "__main__":
    main()
