#!/usr/bin/env python3
"""
Phase 1 — coopmat variant. Exports the .pte with storage_type_override=BUFFER
so the runtime dispatches linear_coopmat (when M >= 64).
"""

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_llama31_coopmat import (  # noqa: E402
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
