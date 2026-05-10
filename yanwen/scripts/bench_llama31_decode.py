#!/usr/bin/env python3
"""
Real autoregressive decode bench: load a KV-cache-enabled .pte and run
N decode steps (each at input shape [1, 1], with input_pos auto-incrementing),
measuring per-step wallclock and total wallclock.

Doesn't sample real tokens — feeds a fixed token each step (we're benchmarking
GPU compute, not generation quality). KV cache state IS persistent across
forwards within this Python process, so per-step time reflects real decode
(cache reads grow linearly with step number).

Usage:
    source /home/doremy/sarc-acl/executorch/main/executorch/.venv/bin/activate
    python yanwen/scripts/bench_llama31_decode.py --n_layers 32 --steps 1024
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

# Force RADV_GTT_PCT before any Vulkan init
os.environ.setdefault("RADV_GTT_PCT", "80")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import torch  # noqa: E402
from executorch.extension.pybindings import portable_lib  # noqa: E402, F401

from executorch.extension.pybindings.portable_lib import (  # noqa: E402, F401
    _load_for_executorch,
)


DEFAULT_PTE_DIR = Path("/home/doremy/llama31_decode_run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_layers", type=int, default=32)
    ap.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Must match the value used at export time.",
    )
    ap.add_argument(
        "--steps", type=int, default=1024, help="Number of decode forwards to run."
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Untimed warmup steps before measurement begins.",
    )
    ap.add_argument("--pte-dir", type=Path, default=DEFAULT_PTE_DIR)
    args = ap.parse_args()

    tag = f"llama31_8b_{args.n_layers}L_decode_max{args.max_seq_len}_fp16"
    pte_path = args.pte_dir / f"{tag}.pte"
    if not pte_path.exists():
        sys.exit(
            f"[bench] missing {pte_path}\n"
            f"        run: python yanwen/scripts/setup_llama31_decode.py "
            f"--n_layers {args.n_layers} --max_seq_len {args.max_seq_len}"
        )

    if args.steps + args.warmup > args.max_seq_len:
        sys.exit(
            f"[bench] steps + warmup ({args.steps + args.warmup}) > "
            f"max_seq_len ({args.max_seq_len}) — KV cache would overflow."
        )

    print(f"[bench] loading {pte_path.name} ({pte_path.stat().st_size/1e9:.2f} GiB)")
    t_load_start = time.perf_counter()
    model = _load_for_executorch(str(pte_path))
    t_load_end = time.perf_counter()
    print(f"[bench] load_method: {t_load_end - t_load_start:.2f}s")

    # Fixed input token — content doesn't matter for compute-shape benchmarking
    tokens = torch.tensor([[1]], dtype=torch.int64)

    total_steps = args.warmup + args.steps
    per_step_ms = []

    print(
        f"[bench] {args.warmup} warmup + {args.steps} measured steps, "
        f"input_pos=0..{total_steps-1}"
    )
    t_run_start = time.perf_counter()
    for step in range(total_steps):
        input_pos = torch.tensor([step], dtype=torch.int64)
        t0 = time.perf_counter()
        _ = model.forward((tokens, input_pos))
        dt = (time.perf_counter() - t0) * 1000.0  # ms
        if step >= args.warmup:
            per_step_ms.append(dt)
        if step in (
            0,
            1,
            args.warmup,
            args.warmup + 100,
            args.warmup + 500,
            total_steps - 1,
        ):
            print(
                f"  step {step:>4} (input_pos={step:>4}): {dt:.1f} ms"
                + (" [warmup]" if step < args.warmup else "")
            )
    t_run_end = time.perf_counter()
    total_wallclock_s = t_run_end - t_run_start

    mean_ms = statistics.mean(per_step_ms)
    stdev_ms = statistics.stdev(per_step_ms) if len(per_step_ms) >= 2 else 0.0
    median_ms = statistics.median(per_step_ms)
    cv = 100 * stdev_ms / mean_ms if mean_ms else 0.0
    measured_wallclock_s = sum(per_step_ms) / 1000.0

    # First-half vs second-half — to expose any context-length scaling
    half = len(per_step_ms) // 2
    first_half_mean = statistics.mean(per_step_ms[:half]) if half else 0.0
    second_half_mean = statistics.mean(per_step_ms[half:]) if half else 0.0

    print("\n=== Decode benchmark ===")
    print(f"  Total wallclock (incl warmup):  {total_wallclock_s:.2f}s")
    print(f"  Measured wallclock ({args.steps} steps): {measured_wallclock_s:.2f}s")
    print(
        f"  Per-step mean ± stdev:          {mean_ms:.1f} ± {stdev_ms:.1f} ms "
        f"(cv={cv:.1f}%)"
    )
    print(f"  Per-step median:                {median_ms:.1f} ms")
    print(
        f"  Per-step min / max:             {min(per_step_ms):.1f} / "
        f"{max(per_step_ms):.1f} ms"
    )
    print(f"  First-half mean (steps 0..{half}):  {first_half_mean:.1f} ms")
    print(f"  Second-half mean (steps {half}..{args.steps}): {second_half_mean:.1f} ms")
    print(
        f"  Growth (sh-fh):                 {second_half_mean - first_half_mean:+.1f} ms"
    )
    print(f"  Tokens/sec (mean):              {1000/mean_ms:.2f} tok/s")

    # Persist results
    out_path = args.pte_dir / "logs" / f"bench_{tag}.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("step\tper_step_ms\n")
        for i, ms in enumerate(per_step_ms):
            f.write(f"{i}\t{ms:.3f}\n")
    print(f"\n[bench] per-step times written to {out_path}")


if __name__ == "__main__":
    main()
