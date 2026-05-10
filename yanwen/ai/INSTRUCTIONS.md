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

All baseline artifacts under `yanwen/artifacts/L32/`. Coopmat artifacts under `yanwen/artifacts/L32_coopmat/`. S=1024 and S=2048 OOM'd before any ETDump was produced — only `memprobe.tsv` is available.

| seq | run | .etdp | .events.tsv | .memprobe.tsv | bench log |
|---:|---|:---:|:---:|:---:|:---:|
| 128 | linear_vec (baseline) | ✓ | ✓ | ✓ | `L32/S128_bench.log` |
| 128 | **linear_coopmat** | ✓ | ✓ | ✓ | `L32_coopmat/S128_bench.log` |
| 512 | linear_vec | ✓ | ✓ | ✓ | `L32/S512_legacy.log` + `S512_etdump.log` |
| 1024 | linear_vec | — (OOM before capture) | — | ✓ (truncated, OOM-killed) | `L32/S1024_oom.log` |
| 2048 | linear_vec | — | — | — | (no run this update) |

## Coopmat workflow (linear_coopmat shader from pavan-report branch)

When the user wants to enable the cooperative-matrix path for fp16 linears
(measured 3.03× whole-forward speedup at L=32 S=128):

### Hard requirements

- **Pavan-report tree**: `/home/doremy/sarc-acl/executorch/pavan-report/executorch/`
  must be on the `pavan-report` branch (verify with `git -C ... branch --show-current`).
- **Pavan-report's runner** must be built (separate from main's `cmake-out-vk/`).
  See "One-time setup" below. The runner has `linear_coopmat`, `matmul_coopmat`,
  and `addmm_khr_cm` GLSL compiled into its `spv.cpp`.
- **Pavan-report's venv** must be activated before any Python invocation:
  `source /home/doremy/sarc-acl/executorch/pavan-report/executorch/.venv/bin/activate`.
  (The modified `tag_memory_meta_pass.py` and `vulkan_preprocess.py` live there.)
- **Re-exported .pte** with `storage_type_override=BUFFER`. The .pte from the
  baseline (`/home/doremy/llama31_pure_run/`) will NOT use coopmat at runtime
  even if you run it through pavan-report's runner — the storage layout
  baked at export time has to be `BUFFER`.

### One-time setup (build pavan-report's runner)

```bash
cd /home/doremy/sarc-acl/executorch/pavan-report/executorch
.venv/lib/python3.12/site-packages/cmake/data/bin/cmake . -Bcmake-out-vk \
    --preset linux \
    -DCMAKE_INSTALL_PREFIX=cmake-out-vk \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON \
    -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
    -DEXECUTORCH_ENABLE_LOGGING=OFF
.venv/lib/python3.12/site-packages/cmake/data/bin/cmake --build cmake-out-vk -j$(nproc) --target install
```

If the build fails with `std::find` / `std::rotate` "no matching function" errors
on `runtime/graph/containers/SharedObject.cpp` or `runtime/graph/ops/impl/Squeeze.cpp`
or similar files, **add `#include <algorithm>` to the offending files**.
A one-shot Python fix:

```bash
python3 -c "
import os, re
root = '/home/doremy/sarc-acl/executorch/pavan-report/executorch/backends/vulkan/runtime'
patterns = re.compile(r'std::(find|sort|rotate|reverse|transform|fill|copy|count|min_element|max_element|unique|partition|all_of|any_of|none_of|for_each|swap|max|min)\b')
includes = re.compile(r'#include\s*<algorithm>')
for d, _, files in os.walk(root):
    for fn in files:
        if not fn.endswith(('.cpp', '.h', '.hpp')): continue
        p = os.path.join(d, fn)
        try: txt = open(p).read()
        except: continue
        if patterns.search(txt) and not includes.search(txt):
            lines = txt.splitlines(keepends=True)
            last_inc = max((i for i, l in enumerate(lines) if l.startswith('#include')), default=-1)
            if last_inc >= 0:
                lines.insert(last_inc + 1, '\n#include <algorithm>\n')
                open(p, 'w').writelines(lines)
                print('FIXED:', p)
"
```

(One-off bug on the `pavan-report` branch from a now-stricter GCC; fixed in this session.)

### Re-export the .pte for coopmat

```bash
source /home/doremy/sarc-acl/executorch/pavan-report/executorch/.venv/bin/activate
sudo swapon /swapfile     # 32L export needs ~16 GiB Python RAM peak
cd /home/doremy/sarc-acl/executorch/main/executorch
python yanwen/scripts/coopmat/setup_llama31_coopmat.py --n_layers 32 --seq_len 128
```

Output: `/home/doremy/llama31_pure_run_coopmat/llama31_8b_32L_seq128_fp16.pte` (16.06 GB; same size as baseline). Heavy: ~5 min.

### Bench (scientific mode)

```bash
source /home/doremy/sarc-acl/executorch/pavan-report/executorch/.venv/bin/activate
cd /home/doremy/sarc-acl/executorch/main/executorch
python yanwen/scripts/coopmat/bench_llama31_coopmat.py --n_layers 32 --seq_len 128
```

Expected: steady-state ~580 ms, throughput ~220 tok/s. ~2 min wallclock total.

### Capture ETDump for shader breakdown

```bash
python yanwen/scripts/coopmat/bench_llama31_coopmat.py --n_layers 32 --seq_len 128 \
    --num_executions 8 --etdump-analyze
```

### Verify coopmat actually fired

```bash
grep -c 'linear_coopmat' /home/doremy/llama31_pure_run_coopmat/logs/bench_L32_S128_*.log
# expect ~896 (224 dispatches per iter × multiple subprocesses)

grep -c 'matmul_coopmat' /home/doremy/llama31_pure_run_coopmat/logs/bench_L32_S128_*.log
# expect ~256 (32 BMM dispatches per iter × multiple subprocesses)

grep -c 'linear_vec' /home/doremy/llama31_pure_run_coopmat/logs/bench_L32_S128_*.log
# expect ~4 (only lm_head fallback, M=1 < 64)
```

If `linear_coopmat` count is 0, debug:

- Verify pavan-report's venv is active (`which python` should point inside `pavan-report/.venv`)
- Verify `RUNNER` in `run_llama31_coopmat.py` points at pavan-report's runner
- Verify `compile_options = {"storage_type_override": VkStorageType.BUFFER}` made it through
- `VK_DISABLE_COOPMAT` env var should NOT be set
- The 780M does support `VK_KHR_cooperative_matrix` — confirm with `vulkaninfo | grep -i cooperative`
