#!/usr/bin/env python3
"""
Run pure, original LLaMA 3.1 8B (fp16) on the AMD 780M iGPU via ExecuTorch
Vulkan delegate, then analyze with ETDump.

Three phases, each isolated so memory is released between them:
  Phase 1  Python in-process: load weights, torch.export(strict=False),
           lower with stock VulkanPartitioner({}), write .pte + etrecord.bin.
  Phase 2  subprocess: cmake-out-vk/executor_runner --etdump_path=...
           Hardened against system-wide OOM via RLIMIT_AS + oom_score_adj.
  Phase 3  Python in-process: Inspector loads etdump+etrecord, prints
           top-N ops, aggregate-by-op, per-module breakdown, TSV.

Usage:
    source /home/doremy/Desktop/samsung/executorch/.venv/bin/activate
    sudo swapon /swapfile   # one-time, before first run
    python run_llama31_pure.py --n_layers 32 --seq_len 128

Sweep:
    for L in 4 8 16 32; do for S in 128 512 1024 2048; do
      python run_llama31_pure.py --n_layers $L --seq_len $S || echo "SKIP $L $S"
    done; done
"""

import argparse
import gc
import json
import os
import resource
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

WEIGHTS_DIR = Path("/home/doremy/llama3_1_8b/original")
CKPT = WEIGHTS_DIR / "consolidated.00.pth"
PARAMS = WEIGHTS_DIR / "params.json"

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "cmake-out-vk" / "executor_runner"

DEFAULT_OUT = Path("/home/doremy/llama31_pure_run")


# ---------------------------------------------------------------------------
# OOM-safety helpers
# ---------------------------------------------------------------------------


def _read_meminfo():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, rest = line.partition(":")
            info[k.strip()] = int(rest.split()[0]) * 1024
    return info


def _parent_oom_hardening():
    # Keep parent at default oom_score_adj. Bumping it to a high value
    # makes the kernel pick the parent as the OOM victim during the
    # 16+GiB Python export peak — that's the opposite of what we want.
    # Only the child (executor_runner) gets oom_score_adj=1000.
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")


def _ensure_venv_path():
    venv_bin = Path(sys.prefix) / "bin"
    if (venv_bin / "flatc").exists():
        cur = os.environ.get("PATH", "")
        if str(venv_bin) not in cur.split(":"):
            os.environ["PATH"] = f"{venv_bin}:{cur}"


def _child_oom_hardening_factory(cap_bytes: int):
    def _preexec():
        try:
            resource.setrlimit(resource.RLIMIT_AS, (cap_bytes, cap_bytes))
        except (ValueError, OSError):
            pass
        try:
            with open("/proc/self/oom_score_adj", "w") as f:
                f.write("1000\n")
        except Exception:
            pass

    return _preexec


def _budget_bytes(reserve_gb: float = 2.0) -> int:
    info = _read_meminfo()
    total = info.get("MemTotal", 0) + info.get("SwapTotal", 0)
    cap = total - int(reserve_gb * 1024**3)
    return max(cap, 4 * 1024**3)


# ---------------------------------------------------------------------------
# Phase 0: env + swap guardrails
# ---------------------------------------------------------------------------


def env_check(skip_swap_check: bool):
    info = _read_meminfo()
    ram_gb = info.get("MemTotal", 0) / 1024**3
    swap_gb = info.get("SwapTotal", 0) / 1024**3
    print(f"[env] RAM total: {ram_gb:.1f} GiB    Swap total: {swap_gb:.1f} GiB")

    if not RUNNER.exists():
        sys.exit(f"[env] FATAL: {RUNNER} not found. Build cmake-out-vk first.")

    os.environ["RADV_GTT_PCT"] = "80"
    print("[env] RADV_GTT_PCT=80 set")

    if swap_gb < 1.0 and not skip_swap_check:
        print(
            "[env] WARNING: no active swap. The 8B fp16 export needs paging headroom."
        )
        print(
            "[env] To activate the existing /swapfile (24 GiB on /home, requires sudo):"
        )
        print("        sudo swapon /swapfile && swapon --show")
        print("[env] Or the LV alternative (14.6 GiB):")
        print("        sudo swapon /dev/rl_proxmox-ryzen/swap")
        print("[env] Re-run with --skip-swap-check to proceed anyway.")
        sys.exit(2)


# ---------------------------------------------------------------------------
# Phase 1: export
# ---------------------------------------------------------------------------


def load_model(n_layers: int, seq_len: int):
    import torch
    from executorch.examples.models.llama.llama_transformer import construct_transformer
    from executorch.examples.models.llama.model_args import ModelArgs

    with open(PARAMS) as f:
        params = json.load(f)
    original_layers = params.get("n_layers", 32)
    if n_layers < original_layers:
        print(f"[export] subsetting layers: {n_layers} of {original_layers}")
        params["n_layers"] = n_layers

    model_args = ModelArgs(
        max_seq_len=seq_len + 16,
        max_context_len=seq_len + 16,
        **params,
    )

    with torch.device("meta"):
        model = construct_transformer(model_args)

    print(f"[export] mmap-loading checkpoint {CKPT}")
    t0 = time.perf_counter()
    checkpoint = torch.load(CKPT, map_location="cpu", mmap=True)  # noqa: TOR102
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    print(f"[export] checkpoint open in {time.perf_counter()-t0:.1f}s")

    if n_layers < original_layers:
        filtered = {}
        for k, v in checkpoint.items():
            if k.startswith("layers."):
                idx = int(k.split(".")[1])
                if idx < n_layers:
                    filtered[k] = v
            else:
                filtered[k] = v
        checkpoint = filtered

    missing, unexpected = model.load_state_dict(checkpoint, strict=False, assign=True)
    miss_w = [k for k in missing if k.endswith(".weight")]
    if miss_w:
        print(f"[export] WARNING missing weights: {miss_w[:3]}...")
    if unexpected:
        print(f"[export] ignored {len(unexpected)} unexpected keys")

    model = model.half().eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[export] params: {n_params/1e9:.2f}B fp16 ({n_params*2/1e9:.1f} GiB)")
    return model, model_args, checkpoint


def export_pte(n_layers: int, seq_len: int, out_dir: Path, want_etrecord: bool):
    import torch
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from torch.export import export

    tag = f"llama31_8b_{n_layers}L_seq{seq_len}_fp16"
    pte_path = out_dir / f"{tag}.pte"
    etrecord_path = out_dir / f"{tag}.etrecord.bin"
    input_path = out_dir / f"{tag}_input0.bin"

    have_etrecord = etrecord_path.exists()
    if (
        pte_path.exists()
        and input_path.exists()
        and (have_etrecord or not want_etrecord)
    ):
        print(
            f"[export] cached: {pte_path} ({pte_path.stat().st_size/1e9:.2f} GiB)"
            + (" + etrecord" if have_etrecord else " (no etrecord)")
        )
        return tag, pte_path, etrecord_path if have_etrecord else None, input_path

    out_dir.mkdir(parents=True, exist_ok=True)

    model, margs, checkpoint = load_model(n_layers, seq_len)
    example_tokens = torch.randint(0, margs.vocab_size, (1, seq_len), dtype=torch.int64)
    example_inputs = (example_tokens,)

    print(f"[export] torch.export(strict=False) tokens={tuple(example_tokens.shape)}")
    t0 = time.perf_counter()
    with torch.no_grad():
        prog = export(model, example_inputs, strict=False)
    print(f"[export] torch.export done in {time.perf_counter()-t0:.1f}s")

    # Drop the original model + checkpoint refs now that prog owns the weights.
    del model, checkpoint
    gc.collect()

    print("[export] to_edge_transform_and_lower (stock VulkanPartitioner)")
    t0 = time.perf_counter()
    edge = to_edge_transform_and_lower(
        prog,
        compile_config=EdgeCompileConfig(_skip_dim_order=False),
        partitioner=[VulkanPartitioner({})],
    )
    et = edge.to_executorch()
    print(f"[export] lowered in {time.perf_counter()-t0:.1f}s")

    # Release prog + edge before writing et.buffer — at L=32 both can each
    # carry ~16 GB of tensor refs, and keeping them alive while et.buffer
    # is materialized + written has OOM'd on the 28.9 GB box.
    if not want_etrecord:
        del prog, edge
        gc.collect()

    print(f"[export] writing .pte -> {pte_path}")
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    size_gb = pte_path.stat().st_size / 1e9
    print(f"[export] .pte size: {size_gb:.2f} GiB")

    print(f"[export] writing input -> {input_path}")
    example_tokens.detach().numpy().astype("int64").tofile(input_path)

    if want_etrecord:
        # generate_etrecord deepcopies the edge program — needs ~1.5–2x the
        # weight size in extra RAM. Skipped by default to avoid OOM.
        print(
            f"[export] writing etrecord -> {etrecord_path} (this can OOM without swap)"
        )
        try:
            from executorch.devtools import generate_etrecord

            generate_etrecord(str(etrecord_path), edge, et, prog)
        except Exception as e:
            print(f"[export] WARNING generate_etrecord failed: {e}")

    # prog + edge may already be del'd above (non-etrecord path)
    if "prog" in locals():
        del prog
    if "edge" in locals():
        del edge
    del et, example_tokens, example_inputs
    gc.collect()
    return (
        tag,
        pte_path,
        etrecord_path if (want_etrecord and etrecord_path.exists()) else None,
        input_path,
    )


# ---------------------------------------------------------------------------
# Phase 2: run executor_runner with ETDump + memory probe
# ---------------------------------------------------------------------------


class MemProbe(threading.Thread):
    def __init__(self, log_path: Path, interval_s: float = 0.5):
        super().__init__(daemon=True)
        self.log_path = log_path
        self.interval = interval_s
        self.stop_evt = threading.Event()
        self.peak_shmem = 0
        self.min_free = 1 << 62
        self.peak_cached = 0
        self.peak_swap_used = 0

    def run(self):
        with open(self.log_path, "w") as out:
            out.write("t_s\tShmem_MB\tMemFree_MB\tCached_MB\tSwapUsed_MB\n")
            t0 = time.time()
            while not self.stop_evt.is_set():
                info = _read_meminfo()
                shm = info.get("Shmem", 0)
                mf = info.get("MemFree", 0)
                ca = info.get("Cached", 0)
                su = info.get("SwapTotal", 0) - info.get("SwapFree", 0)
                self.peak_shmem = max(self.peak_shmem, shm)
                self.min_free = min(self.min_free, mf)
                self.peak_cached = max(self.peak_cached, ca)
                self.peak_swap_used = max(self.peak_swap_used, su)
                out.write(
                    f"{time.time()-t0:.2f}\t{shm/1e6:.1f}\t{mf/1e6:.1f}\t{ca/1e6:.1f}\t{su/1e6:.1f}\n"
                )
                out.flush()
                self.stop_evt.wait(self.interval)


def run_etdump(
    pte_path: Path,
    input_path: Path,
    etdump_path: Path,
    num_executions: int,
    mem_log: Path,
    want_etdump: bool,
):
    cap = _budget_bytes(reserve_gb=2.0)
    cap_gb = cap / 1024**3
    print(f"[run] RLIMIT_AS cap on child: {cap_gb:.1f} GiB (oom_score_adj=1000)")

    cmd = [
        str(RUNNER),
        f"--model_path={pte_path}",
        f"--inputs={input_path}",
        f"--num_executions={num_executions}",
    ]
    if want_etdump:
        cmd.append(f"--etdump_path={etdump_path}")
    print(f"[run] $ {' '.join(cmd)}")

    probe = MemProbe(mem_log)
    probe.start()
    t0 = time.perf_counter()
    rc = -1
    try:
        env = {**os.environ, "RADV_GTT_PCT": "80", "MALLOC_ARENA_MAX": "2"}
        proc = subprocess.run(
            cmd,
            env=env,
            preexec_fn=_child_oom_hardening_factory(cap),
            check=False,
        )
        rc = proc.returncode
    finally:
        probe.stop_evt.set()
        probe.join(timeout=2.0)
    dt = time.perf_counter() - t0

    if rc != 0:
        print(
            f"[run] executor_runner exited rc={rc} after {dt:.1f}s — likely OOM-killed."
        )
        print(
            f"[run] memprobe peaks: Shmem={probe.peak_shmem/1e6:.0f} MB  "
            f"MinFree={probe.min_free/1e6:.0f} MB  "
            f"SwapUsed={probe.peak_swap_used/1e6:.0f} MB"
        )
        return None, probe
    print(
        f"[run] executor_runner wall-clock: {dt:.2f}s for {num_executions} executions"
    )
    print(f"[run]   ~ {dt*1000/num_executions:.1f} ms / execution (subprocess wall)")
    return dt / num_executions, probe


# ---------------------------------------------------------------------------
# Phase 2.5: benchmark with explicit warmup
# ---------------------------------------------------------------------------


def bench_steady_state(
    pte_path: Path,
    input_path: Path,
    etdump_path: Path,
    mem_log: Path,
    n_reps: int = 3,
    n_iters_per_rep: int = 8,
):
    """Scientific bench via algebraic subtraction — sidesteps the unreliable
    ETDump per-iter timing on the Vulkan delegate.

    Runs (1 + n_reps) subprocesses:
      - calibration: N=1 → W1 = load + iter 0 + teardown
      - n_reps × at N=K → WK_i = load + iter 0 + (K-1)*steady + teardown
    Per-rep steady forward = (WK_i - W1) / (K-1). load+iter-0+teardown cancels.

    Returns dict (with mean ± stdev across reps) or None on failure.
    """
    print("\n=== Calibration: N=1 (load + iter 0 + teardown) ===")
    cal_per_exec, _ = run_etdump(
        pte_path, input_path, etdump_path, 1, mem_log, want_etdump=False
    )
    if cal_per_exec is None:
        return None
    W1_ms = cal_per_exec * 1000.0
    print(f"  W1 = {W1_ms:.1f} ms")

    print(f"\n=== Measurement: {n_reps} reps × N={n_iters_per_rep} ===")
    forwards_ms = []
    walls_ms = []
    for i in range(n_reps):
        wall_per_exec, _ = run_etdump(
            pte_path,
            input_path,
            etdump_path,
            n_iters_per_rep,
            mem_log,
            want_etdump=False,
        )
        if wall_per_exec is None:
            return None
        WK_ms = wall_per_exec * 1000.0 * n_iters_per_rep
        steady_ms = (WK_ms - W1_ms) / (n_iters_per_rep - 1)
        forwards_ms.append(steady_ms)
        walls_ms.append(wall_per_exec * 1000.0)
        print(
            f"  rep {i+1}: WK={WK_ms:.1f} ms  wall/N={wall_per_exec*1000:.1f} ms"
            f"  steady=(WK-W1)/{n_iters_per_rep-1}={steady_ms:.1f} ms"
        )

    mean_ms = statistics.mean(forwards_ms)
    stdev_ms = statistics.stdev(forwards_ms) if n_reps >= 2 else 0.0
    cv = 100 * stdev_ms / mean_ms if mean_ms else 0.0
    wall_mean = statistics.mean(walls_ms)

    print(
        "\n=== Steady-state forward (iter 0 + load+teardown algebraically excluded) ==="
    )
    print("  per-rep steady: " + "  ".join(f"{x:.1f}" for x in forwards_ms) + " ms")
    print(
        f"  mean ± stdev:   {mean_ms:.1f} ± {stdev_ms:.1f} ms  "
        f"(cv={cv:.1f}%, min={min(forwards_ms):.1f}, max={max(forwards_ms):.1f})"
    )
    print(
        f"  wallclock/N mean: {wall_mean:.1f} ms  "
        f"(legacy metric; +{wall_mean - mean_ms:.1f} ms inflation from load+iter0/N)"
    )

    return {
        "W1_ms": W1_ms,
        "forwards_ms": forwards_ms,
        "walls_ms": walls_ms,
        "mean_ms": mean_ms,
        "stdev_ms": stdev_ms,
        "cv_pct": cv,
        "n_reps": n_reps,
        "n_iters_per_rep": n_iters_per_rep,
    }


# ---------------------------------------------------------------------------
# Phase 3: analyze
# ---------------------------------------------------------------------------


def analyze(etdump_path: Path, etrecord_path, tsv_path: Path, probe):  # noqa: C901
    from executorch.devtools import Inspector

    print("\n=== ETDump analysis ===")
    insp = Inspector(
        etdump_path=str(etdump_path),
        etrecord=str(etrecord_path) if etrecord_path is not None else None,
    )
    if etrecord_path is None:
        print(
            "[analyze] (no etrecord — module hierarchy unavailable; op names from runtime)"
        )

    rows = []
    for block in insp.event_blocks:
        for ev in block.events:
            samples = getattr(getattr(ev, "perf_data", None), "raw", None)
            if not samples:
                continue
            mean_us = sum(samples) / len(samples)
            mh = getattr(ev, "module_hierarchy", None) or {}
            rows.append((mean_us, ev.name, mh))

    if not rows:
        print("[analyze] no perf events found in etdump")
        return

    rows.sort(reverse=True)
    total_us = sum(r[0] for r in rows)
    print(f"[analyze] events: {len(rows)}    measured total: {total_us/1000:.2f} ms")

    print("\n--- Top-30 ops by mean latency ---")
    print(f"{'rank':>4}  {'mean_us':>10}  {'%':>6}  name")
    for i, (us, name, _) in enumerate(rows[:30]):
        pct = 100.0 * us / total_us if total_us else 0.0
        print(f"{i+1:>4}  {us:>10.1f}  {pct:>5.1f}%  {name}")

    print("\n--- Aggregate by op type ---")
    agg = {}
    for us, name, _ in rows:
        key = name.split("[")[0].split("::")[-1].strip() or name
        a = agg.setdefault(key, [0, 0.0])
        a[0] += 1
        a[1] += us
    by_total = sorted(agg.items(), key=lambda kv: kv[1][1], reverse=True)
    print(f"{'op':<40}  {'count':>6}  {'total_ms':>10}  {'%':>6}")
    for op, (n, tot) in by_total[:30]:
        pct = 100.0 * tot / total_us if total_us else 0.0
        print(f"{op:<40}  {n:>6}  {tot/1000:>10.2f}  {pct:>5.1f}%")

    print("\n--- Per-submodule breakdown ---")
    sub = {}
    for us, _, mh in rows:
        bucket = "(unmapped)"
        if mh:
            for stack in mh.values():
                names = (
                    list(stack)
                    if isinstance(stack, (list, tuple))
                    else list(stack.keys()) if isinstance(stack, dict) else [str(stack)]
                )
                joined = ".".join(str(s) for s in names)
                for kw in (
                    "attention",
                    "feed_forward",
                    "ffn",
                    "norm",
                    "output",
                    "tok_embeddings",
                    "rope",
                ):
                    if kw in joined.lower():
                        bucket = kw
                        break
                else:
                    bucket = names[0] if names else bucket
                break
        s = sub.setdefault(bucket, [0, 0.0])
        s[0] += 1
        s[1] += us
    for name, (n, tot) in sorted(sub.items(), key=lambda kv: kv[1][1], reverse=True):
        pct = 100.0 * tot / total_us if total_us else 0.0
        print(f"  {name:<24}  count={n:>5}  total={tot/1000:>9.2f} ms  {pct:>5.1f}%")

    try:
        df = insp.to_dataframe()
        df.to_csv(tsv_path, sep="\t", index=False)
        print(f"\n[analyze] wrote full event TSV -> {tsv_path}")
    except Exception as e:
        print(f"[analyze] to_dataframe failed: {e}")

    if probe is not None:
        info = _read_meminfo()
        gtt_cap = int(0.8 * info.get("MemTotal", 0))
        print("\n--- Memory profile ---")
        print(f"  peak Shmem (RADV GTT-backed): {probe.peak_shmem/1e6:>9.0f} MB")
        print(f"  GTT cap (RADV_GTT_PCT=80):    {gtt_cap/1e6:>9.0f} MB")
        print(f"  peak Cached (.pte mmap):      {probe.peak_cached/1e6:>9.0f} MB")
        print(f"  min MemFree:                  {probe.min_free/1e6:>9.0f} MB")
        print(f"  peak Swap used:               {probe.peak_swap_used/1e6:>9.0f} MB")
        if probe.peak_shmem > 0.9 * gtt_cap:
            print(
                "  [warn] peak Shmem within 10% of GTT cap — close to Vulkan OOM ceiling."
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_layers", type=int, default=32, choices=range(1, 33))
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--num_executions", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-swap-check", action="store_true")
    ap.add_argument(
        "--etrecord",
        action="store_true",
        help="Also write an ETRecord (per-module names in report). "
        "Heavy on RAM during export; needs swap on for 32L.",
    )
    ap.add_argument(
        "--etdump",
        action="store_true",
        help="Capture per-op ETDump during run + analyze. Requires "
        "executor_runner built with EXECUTORCH_ENABLE_EVENT_TRACER. "
        "Off by default — wallclock timing works without it.",
    )
    ap.add_argument(
        "--phase", choices=["all", "export", "run", "analyze"], default="all"
    )
    args = ap.parse_args()

    _parent_oom_hardening()
    _ensure_venv_path()
    env_check(args.skip_swap_check)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"llama31_8b_{args.n_layers}L_seq{args.seq_len}_fp16"
    pte_path = out_dir / f"{tag}.pte"
    etrecord_path = out_dir / f"{tag}.etrecord.bin"
    input_path = out_dir / f"{tag}_input0.bin"
    etdump_path = out_dir / f"{tag}.etdp"
    mem_log = out_dir / f"{tag}.memprobe.tsv"
    tsv_path = out_dir / f"{tag}.events.tsv"

    if args.phase in ("all", "export"):
        export_pte(args.n_layers, args.seq_len, out_dir, want_etrecord=args.etrecord)
    if args.phase == "export":
        return

    probe = None
    if args.phase in ("all", "run"):
        if not pte_path.exists():
            sys.exit(f"[main] missing {pte_path} — run --phase export first.")
        gc.collect()
        _, probe = run_etdump(
            pte_path,
            input_path,
            etdump_path,
            args.num_executions,
            mem_log,
            want_etdump=args.etdump,
        )
        if probe is None:
            sys.exit(3)
    if args.phase == "run":
        return

    if args.phase in ("all", "analyze"):
        if not args.etdump:
            print(
                "[main] --etdump not set; skipping Inspector analysis. "
                "Wallclock timing was reported above."
            )
        elif not etdump_path.exists():
            sys.exit(f"[main] missing {etdump_path} — run --phase run --etdump first.")
        else:
            etr = etrecord_path if etrecord_path.exists() else None
            analyze(etdump_path, etr, tsv_path, probe)

    print(f"\n[main] artifacts in {out_dir}/{tag}.*")


if __name__ == "__main__":
    main()
