# INSTRUCTIONS — for AI agents

Goal: bench pure LLaMA 3.1 8B fp16 prefill on AMD 780M iGPU via the ExecuTorch Vulkan delegate, and produce ETDump shader-level breakdowns. **Scope: L=32 only.**

## Repo layout

```
yanwen/
├── INSTRUCTIONS.md          # this file
├── REPORT.md                # findings (read this for context)
├── README.md                # short pointer
├── scripts/
│   ├── run_llama31_pure.py    # Phase 1+2+3 helper + combined CLI
│   ├── setup_llama31_pure.py  # Phase 1: export .pte (heavy, ~5 min, ~16 GiB Python RAM)
│   ├── bench_llama31_pure.py  # Phase 2: bench .pte (default = scientific mode)
│   ├── etvk_breakdown.py      # parse events.tsv → per-shader GPU breakdown
│   └── linear_by_shape.py     # parse .etdp → linear dispatches by output shape
├── artifacts/L32/             # symlinks to /home/doremy/llama31_pure_run/ outputs
└── old/                       # outdated reports / superseded methodology
```

External (NOT in repo, paths absolute):

```
/home/doremy/llama31_pure_run/                  # all .pte / .etdp / .events.tsv / memprobe / logs
/home/doremy/llama3_1_8b/original/              # source weights (consolidated.00.pth, params.json)
/home/doremy/sarc-acl/executorch/main/executorch/cmake-out-vk/executor_runner   # the runner binary
/home/doremy/sarc-acl/executorch/main/executorch/.venv                          # Python venv with ET installed
/swapfile                                        # 24 GiB swap, must be on for L=32 export
```

## Hard constraints

- **Working directory** for invoking scripts: `/home/doremy/sarc-acl/executorch/main/executorch/` (the `executorch` repo root). The scripts use relative imports, so must be run from there or with that on `PYTHONPATH`.
- **Venv:** `source .venv/bin/activate` before any Python command. Required for `executorch.devtools.Inspector`, `torch`, the partitioner, etc.
- **Swap:** `sudo swapon /swapfile` is required only for **export of L=32** (Phase 1 peaks ~16 GiB RAM). Phase 2 (bench) does **not** need swap on for L=32 S=128.
- **`RADV_GTT_PCT=80`** is set automatically by every script (see `env_check`). Do not override.
- **Don't spawn `executor_runner` directly** unless you intend to bypass the OOM hardening (`RLIMIT_AS`, `oom_score_adj=1000`). Always go through the Python wrappers.
- **Do not skip the calibration subprocess** in scientific bench mode — it's how `(load + iter 0)` is subtracted out.

## Common workflows

### 1. Bench an existing .pte (scientific mode — DEFAULT)

The .pte at `/home/doremy/llama31_pure_run/llama31_8b_32L_seq128_fp16.pte` already exists. Just bench it:

```bash
cd /home/doremy/sarc-acl/executorch/main/executorch
source .venv/bin/activate
python yanwen/scripts/bench_llama31_pure.py --n_layers 32 --seq_len 128
```

Default flags: `--reps 3 --iters 8`. Runs 1 calibration subprocess at N=1 + 3 measurement subprocesses at N=8 each. Output:

```
=== Calibration: N=1 (load + iter 0 + teardown) ===
  W1 = <ms>
=== Measurement: 3 reps × N=8 ===
  rep 1: WK=...  wall/N=...  steady=(WK-W1)/7=... ms
  rep 2: ...
  rep 3: ...
=== Steady-state forward (iter 0 + load+teardown algebraically excluded) ===
  per-rep steady: ...
  mean ± stdev: <X> ± <Y> ms (cv=Z%)
  wallclock/N mean: <legacy>
```

The headline number is **`mean ± stdev`** under "Steady-state forward". The legacy `wallclock/N` number is reported for back-compat with the original 2026-05-07 sweep — **don't cite it as the forward time**.

### 2. Capture an ETDump for shader-level analysis

ETDump capture requires `executor_runner` built with `EXECUTORCH_ENABLE_EVENT_TRACER=ON` (already on in `cmake-out-vk/`).

```bash
python yanwen/scripts/bench_llama31_pure.py --n_layers 32 --seq_len 128 \
    --num_executions 8 --etdump-analyze
```

This uses **legacy mode** (single subprocess, `wallclock/N`) — required for ETDump because the scientific mode's calibration subprocess turns off ETDump collection.

Output files (in `/home/doremy/llama31_pure_run/`, symlinked under `yanwen/artifacts/L32/`):

```
llama31_8b_32L_seq128_fp16.etdp                # binary ETDump
llama31_8b_32L_seq128_fp16.events.tsv          # human-readable event table
llama31_8b_32L_seq128_fp16.memprobe.tsv        # /proc/meminfo samples during run
```

### 3. Per-shader GPU breakdown from .etdp

**Use the canonical analyzer** at `pavan-report/executorch/yanwen_plan/analyze_etdump.py`. It correctly skips wrapper events (`DELEGATE_CALL`, `OPERATOR_CALL`, `ETVK_EXECUTE`, `ETVK_COMPUTE_GRAPH_EXECUTE`) and surfaces the kernel-name leaf events.

```bash
python /home/doremy/sarc-acl/executorch/pavan-report/executorch/yanwen_plan/analyze_etdump.py \
    /home/doremy/llama31_pure_run/llama31_8b_32L_seq128_fp16.etdp
```

Output: per-category time/share/count + top 15 ops by total time.

`perf_data.raw` is **already in milliseconds**. Don't divide by 1000.

### 4. Per-shader breakdown discarding iter 0 (steady-state)

Use `yanwen/scripts/etvk_breakdown.py`:

```bash
python yanwen/scripts/etvk_breakdown.py \
    /home/doremy/llama31_pure_run/llama31_8b_32L_seq128_fp16.events.tsv 8
```

Second arg = `num_executions` from the bench. Reports iter 0 time and steady-state (mean of iter 1..N-1) per kernel + bucketed by family.

### 5. Linear dispatches by output shape

```bash
python yanwen/scripts/linear_by_shape.py \
    /home/doremy/llama31_pure_run/llama31_8b_32L_seq128_fp16.etdp
```

Output: `[seq, hidden]` shape × #dispatches × kernel(s) × % of linear time.

### 6. Re-export a .pte (only if needed)

Heavy: ~5 min, ~16 GiB Python RAM peak at L=32, requires swap.

```bash
sudo swapon /swapfile && swapon --show     # if not on
python yanwen/scripts/setup_llama31_pure.py --n_layers 32 --seq_len 128
```

Output: `/home/doremy/llama31_pure_run/llama31_8b_32L_seq128_fp16.pte` (~16 GB).

## Empirical limits at L=32

| seq | Status | Forward | Notes |
|---:|---|---:|---|
| 128 | ✓ performant | 1.77 s ± 6 ms | The only acceptable config |
| 512 | ◐ runs but unusable | ~95 s/forward | Memory thrash; near-OOM |
| 1024 | ✗ OOM during calibration | — | Swap saturated at 25.8 GB |
| 2048 | ✗ OOM | — | 5/5 prior attempts (2026-05-06) too |

**Don't attempt `seq ≥ 1024` at L=32 on this hardware.** It's a memory-architecture bound, not a shader bound.

## Methodology gotchas

1. **`bench_steady_state()` returns the proper number.** `wallclock/N` from a single subprocess folds in `(load + iter 0)/N` — for L=32 that's ~1.3 s of inflation per "exec" (~80% of true forward). Always prefer scientific mode for headline numbers.
2. **ETDump per-iter samples on Vulkan delegate**: `perf_data.raw` for kernel events IS GPU time in ms (from `vkCmdWriteTimestamp`). For wrapper events (`DELEGATE_CALL`, `Method::execute`), the value is CPU dispatch time, much smaller than GPU. Skip wrappers when summing.
3. **`analyze()` in `run_llama31_pure.py:328` is broken** — it doesn't deduplicate wrappers or extract kernel_name from JSON event names. Use the canonical `analyze_etdump.py` instead.
4. **MemProbe writes `/proc/meminfo` to `<tag>.memprobe.tsv` every 500 ms.** Columns are `t_s, Shmem_MB, MemFree_MB, Cached_MB, SwapUsed_MB`. Each subprocess **overwrites** the file — for multi-rep runs only the last subprocess's probe survives.
5. **OOM-killed runs return `rc=-9`** from the runner. The Python wrapper logs `[run] executor_runner exited rc=-9 after Xs — likely OOM-killed.` and returns `None`.
6. **Kernel counts in events.tsv are total unique events × N samples each.** E.g., 4557 kernel events with `len(raw)==8` means there are 4557 distinct dispatch sites in the graph (across 8 iter samples each). Per-iter dispatch count = 4557 (not 4557/8).

## Configs tested in this update

All artifacts under `yanwen/artifacts/L32/`. ETDump captured for S=128. S=512 ETDump is being captured at the time of writing (cliff regime, ~5 min). S=1024 and S=2048 OOM'd before any ETDump was produced — only `memprobe.tsv` is available.

| seq | .etdp | .events.tsv | .memprobe.tsv | bench log |
|---:|:---:|:---:|:---:|:---:|
| 128 | ✓ | ✓ | ✓ | `S128_bench.log` |
| 512 | ✓ | ✓ | ✓ | `S512_legacy.log` + `S512_etdump.log` |
| 1024 | — (OOM before capture) | — | ✓ (truncated, OOM-killed) | `S1024_oom.log` |
| 2048 | — | — | — | (no run this update) |
