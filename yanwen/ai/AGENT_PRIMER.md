# AGENT_PRIMER — READ THIS FIRST

You are an AI agent picking up the **LLaMA 3.1 8B prefill on AMD Radeon 780M iGPU** benchmarking work. This file is your single-page orientation. Read it end to end before doing anything else, then follow the "Read order" at the bottom for deeper context.

## What this work is

The goal is to measure end-to-end execution time of **pure, original LLaMA 3.1 8B fp16** (no quantization, no shader overrides) on the AMD Radeon 780M iGPU using the ExecuTorch Vulkan delegate. We compare two paths:

1. **Baseline** — stock `VulkanPartitioner({})`, which today on `main` ends up dispatching `linear_vec_buffer_texture2d_half` (GEMV-style shader, fp16, buffer activations + texture2d weights).
2. **Coopmat** — `VulkanPartitioner({"storage_type_override": VkStorageType.BUFFER})` against the **`pavan-report`** branch's runner, which has `linear_coopmat.glsl` (KHR cooperative matrix, fp16, buffer+buffer storage).

Both run the same model architecture, the same hardware, the same fp16 dtype, the same prefill (single forward over `[1, seq_len]`).

## Scope and non-scope

**In scope:**

- L=32 (full 32-layer model), seq_len=128, fp16, prefill only
- Compare baseline vs coopmat shader paths
- Per-shader GPU breakdown via ETDump kernel events
- Memory architecture investigation (working set, swap, OOM boundaries)

**Out of scope (do not redo unless asked):**

- L != 32. Sub-L32 results in `../old/` are outdated (different methodology, before we discovered the `wallclock/N` inflation issue).
- seq_len ≥ 1024 at L=32. Empirically OOM-killed; hardware can't fit the working set. Don't waste time retrying.
- seq_len = 512 at L=32. Completes but cliffs (~95 s/forward due to RAM saturation + swap thrashing). ETDump captured once; not worth re-running.
- Decoding / autoregressive generation. We only measure prefill.
- Quantization (int4 / int8 / etc.). The work is fp16-only.
- Anything that requires `git commit` or modifying production code. This work is a benchmarking study, not productionization.

## Hard environmental facts

| Item | Value |
|---|---|
| Hardware | AMD Radeon 780M (RADV PHOENIX), RDNA3+ mobile iGPU, Mesa 25.0.7 |
| Host RAM | 28.9 GiB DDR5 |
| Host swap | 24 GiB on `/swapfile` (priority -2) — must `sudo swapon /swapfile` before 32L export |
| GTT cap env | `RADV_GTT_PCT=80` — auto-set by every bench script |
| Weights | `/home/doremy/llama3_1_8b/original/{consolidated.00.pth, params.json}` (~15 GB checkpoint) |
| Repo root (main) | `/home/doremy/sarc-acl/executorch/main/executorch/` |
| Repo root (pavan-report) | `/home/doremy/sarc-acl/executorch/pavan-report/executorch/` |
| Main's runner | `main/executorch/cmake-out-vk/executor_runner` |
| Pavan-report's runner | `pavan-report/executorch/cmake-out-vk/executor_runner` |
| Main's venv | `main/executorch/.venv/` |
| Pavan-report's venv | `pavan-report/executorch/.venv/` |
| Baseline output dir | `/home/doremy/llama31_pure_run/` |
| Coopmat output dir | `/home/doremy/llama31_pure_run_coopmat/` |

The two ExecuTorch trees are **independent**: each has its own venv (with its own installed `executorch` package and modified partitioner code), and its own compiled `executor_runner` (with its own embedded SPIR-V shader table). **Don't cross-pollinate** — use main's runner with main's venv and main's .pte for the baseline; use pavan-report's runner with pavan-report's venv and pavan-report's .pte for coopmat.

## Confirmed numbers (so you can verify the environment is intact)

| Config | Steady-state forward (cv) | Throughput | W1 (load + iter 0) |
|---|---:|---:|---:|
| **L=32 S=128 baseline** (prefill) | 1.766 ± 0.006 s (0.3%) | 72.5 tok/s | 21.4 s |
| **L=32 S=128 coopmat** (prefill) | 0.583 ± 0.002 s (0.4%) | 219.7 tok/s | 19.3 s |
| Speedup (prefill) | 3.03× | 3.03× | — |
| **L=4 S=1 baseline** (decode-shape) | 41.1 ms ETDump GPU | — | — |
| **L=4 S=1 coopmat** (decode-shape) | 40.7 ms ETDump GPU | — | — |
| Speedup (decode-shape) | **1.01× (no-op)** — gate fails at M=1 | — | — |
| **L=32 decode** (extrapolated) | ~240 ms / token | ~4.2 tok/s | — |

Verification dispatches per forward (from grepping the runner stderr):

- Baseline: 224× `linear_vec_buffer_texture2d_half` + 2× `linear_vec_buffer_buffer_half` (lm_head) + 32× `matmul_vec_*` (attention BMMs)
- Coopmat: 224× `linear_coopmat_half` + 2× `linear_vec_buffer_buffer_half` (lm_head, M=1 fallback) + 32× `matmul_coopmat_float`

If you re-run and get numbers materially different from these (>5% on the headline), something has drifted. Investigate (check runner build, .pte freshness, venv).

## Read order for full context

1. **`AGENT_PRIMER.md`** — this file
2. **`METHODOLOGY.md`** — bench methodology (scientific mode, ETDump usage, anti-patterns)
3. **`SHADER_DISPATCH.md`** — how Vulkan picks shader variants, coopmat dispatch conditions
4. **`COOPMAT_WORKFLOW.md`** — exact step-by-step reproduction of the coopmat experiment
5. **`GOTCHAS.md`** — bugs, traps, and methodological mistakes we made and corrected
6. **`INSTRUCTIONS.md`** — quick-reference commands (terse, exact)
7. **`../reports/REPORT.md`** + **`../reports/L32_S128_coopmat_REPORT.md`** — human-readable findings

For deep-dive into specific shaders: `../reports/L32_S128_shader_breakdown.md` (baseline) and `../reports/L32_S128_coopmat_shader_breakdown.md` (coopmat).

For raw bench data: `../artifacts/L32/` and `../artifacts/L32_coopmat/` (.etdp, .events.tsv, memprobe, logs).

## Verify-before-trusting-memory checklist

Run these checks before relying on anything in this primer (they are cheap and catch environment drift):

```bash
# 1. pavan-report tree is on the right branch
git -C /home/doremy/sarc-acl/executorch/pavan-report/executorch branch --show-current
# Expected: pavan-report

# 2. main's runner exists and is built with EVENT_TRACER
ls -la /home/doremy/sarc-acl/executorch/main/executorch/cmake-out-vk/executor_runner
grep '^EXECUTORCH_ENABLE_EVENT_TRACER' /home/doremy/sarc-acl/executorch/main/executorch/cmake-out-vk/CMakeCache.txt
# Expected: file exists ~18 MB; flag is ON

# 3. pavan-report's runner exists and is built with EVENT_TRACER
ls -la /home/doremy/sarc-acl/executorch/pavan-report/executorch/cmake-out-vk/executor_runner
grep '^EXECUTORCH_ENABLE_EVENT_TRACER' /home/doremy/sarc-acl/executorch/pavan-report/executorch/cmake-out-vk/CMakeCache.txt
# Expected: file exists ~18 MB; flag is ON
# If missing: see COOPMAT_WORKFLOW.md "One-time setup"

# 4. Both .pte files exist
ls -la /home/doremy/llama31_pure_run/llama31_8b_32L_seq128_fp16.pte
ls -la /home/doremy/llama31_pure_run_coopmat/llama31_8b_32L_seq128_fp16.pte
# Expected: both ~16 GB
# If missing: re-export via the appropriate setup script

# 5. Coopmat scripts exist (they vanished once during this session — assume worst case)
ls /home/doremy/sarc-acl/executorch/main/executorch/yanwen/scripts/coopmat/
# Expected: 3 .py files (run, setup, bench)
```

If any check fails: see `GOTCHAS.md` and `COOPMAT_WORKFLOW.md` for recovery.

## What success looks like for a new agent

If the user asks "redo the coopmat experiment" or "verify the 3× speedup", you should:

1. Skim this primer + `COOPMAT_WORKFLOW.md`
2. Run the verify checklist above
3. Re-run the bench commands from `COOPMAT_WORKFLOW.md`
4. Compare to the "Confirmed numbers" table in this primer
5. Report any deviations

Do **not** invent new methodology or re-investigate things already documented in `GOTCHAS.md`. The pitfalls documented there cost real session time to discover — don't relearn them.
