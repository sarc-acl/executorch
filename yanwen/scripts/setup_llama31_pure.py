#!/usr/bin/env python3
"""
Phase 1 only: export pure, original LLaMA 3.1 8B (fp16) to .pte for the
Vulkan delegate. One-time per (n_layers, seq_len) — output is cached, so
re-running with the same args is a no-op.

Heavy on RAM during torch.export + to_executorch (~16 GiB for 32L). Needs
swap on. Pair with bench_llama31_pure.py for the actual benchmark.

Usage:
    source /home/doremy/Desktop/samsung/executorch/.venv/bin/activate
    sudo swapon /swapfile   # one-time before first 32L run
    python setup_llama31_pure.py --n_layers 32 --seq_len 128
"""

import argparse
import gc
from pathlib import Path

from run_llama31_pure import (
    _ensure_venv_path,
    _parent_oom_hardening,
    DEFAULT_OUT,
    env_check,
    export_pte,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_layers", type=int, default=32, choices=range(1, 33))
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-swap-check", action="store_true")
    ap.add_argument(
        "--etrecord",
        action="store_true",
        help="Also write an ETRecord (per-module names in report). "
        "Heavy on RAM during export.",
    )
    args = ap.parse_args()

    _parent_oom_hardening()
    _ensure_venv_path()
    env_check(args.skip_swap_check)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    export_pte(args.n_layers, args.seq_len, args.out_dir, want_etrecord=args.etrecord)
    gc.collect()

    tag = f"llama31_8b_{args.n_layers}L_seq{args.seq_len}_fp16"
    print(f"\n[setup] done. .pte -> {args.out_dir}/{tag}.pte")
    print(
        f"[setup] benchmark:  python bench_llama31_pure.py "
        f"--n_layers {args.n_layers} --seq_len {args.seq_len}"
    )


if __name__ == "__main__":
    main()
