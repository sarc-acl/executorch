#!/usr/bin/env python3
"""
Phase 2 (+ optional 3) — int8 variant. Uses main tree's executor_runner and
the same scientific-bench methodology as bench_llama31_pure.py. Expects the
.pte from setup_llama31_int8.py at /home/doremy/llama31_pure_run_int8/.
"""

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_llama31_int8 import (  # noqa: E402
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
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument(
        "--num_executions",
        type=int,
        default=None,
        help="Legacy wallclock/N mode. Overrides --reps/--iters.",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-swap-check", action="store_true")
    ap.add_argument("--etdump-analyze", action="store_true")
    args = ap.parse_args()

    _parent_oom_hardening()
    _ensure_venv_path()
    env_check(args.skip_swap_check)

    tag = f"llama31_8b_{args.n_layers}L_seq{args.seq_len}_int8"
    pte_path = args.out_dir / f"{tag}.pte"
    etrecord_path = args.out_dir / f"{tag}.etrecord.bin"
    input_path = args.out_dir / f"{tag}_input0.bin"
    etdump_path = args.out_dir / f"{tag}.etdp"
    mem_log = args.out_dir / f"{tag}.memprobe.tsv"
    tsv_path = args.out_dir / f"{tag}.events.tsv"

    if not pte_path.exists():
        sys.exit(f"[bench] missing {pte_path}\n" f"        run setup first.")
    if not input_path.exists():
        sys.exit(f"[bench] missing {input_path} — re-run setup.")

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
            sys.exit(f"[bench] expected etdump at {etdump_path} but missing.")
        etr = etrecord_path if etrecord_path.exists() else None
        analyze(etdump_path, etr, tsv_path, probe)

    print(f"\n[bench] artifacts in {args.out_dir}/{tag}.*")


if __name__ == "__main__":
    main()
