#!/usr/bin/env python3
"""
W8A8 (XNNPACK static int8) export wrapper. Mirrors setup_llama31_int8.py but
points at run_llama31_w8a8_xnn (which swaps in XNNPACKQuantizer + calibration).

Usage:
    source /home/doremy/sarc-acl/executorch/pavan-report/executorch/.venv/bin/activate
    sudo swapon /swapfile
    python yanwen/scripts/int8/setup_llama31_w8a8_xnn.py --n_layers 32 --seq_len 128

For initial validation, start with --n_layers 4 to verify the export pipeline
end-to-end before committing to the ~5-10 min L=32 export.
"""

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_llama31_w8a8_xnn import (  # noqa: E402
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
    ap.add_argument("--etrecord", action="store_true")
    args = ap.parse_args()

    _parent_oom_hardening()
    _ensure_venv_path()
    env_check(args.skip_swap_check)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gc.collect()
    export_pte(args.n_layers, args.seq_len, args.out_dir, want_etrecord=args.etrecord)
    print(f"\n[setup] artifacts in {args.out_dir}/")


if __name__ == "__main__":
    main()
