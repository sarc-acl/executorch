# yanwen — LLaMA 3.1 8B prefill on AMD Radeon 780M iGPU

End-to-end benchmark of LLaMA 3.1 8B prefill on the AMD Radeon 780M, sweeping shader variants (fp16 baseline `linear_vec`, fp16 `linear_coopmat`, int8 W8A16 `linear_qcs8w_tiled`) and one shader-level prediction for int8 KHR cooperative matrix.

**Headline results at L=32, seq=128:**

| | Steady forward | Throughput | vs fp16 baseline |
|---|---:|---:|---:|
| Baseline fp16 (`linear_vec`) | 1.766 ± 0.006 s | 72.5 tok/s | 1.0× |
| fp16 coopmat (`linear_coopmat`) | 0.583 ± 0.002 s | 219.7 tok/s | **3.03×** |
| int8 W8A16 (`linear_qcs8w_tiled`, `vulkan_8w` default) | 2.108 ± 0.029 s | 60.7 tok/s | **0.84× (slower)** |
| int8 KHR coopmat (projected from microbench v2, deferred E2E) | ~0.49 s | ~261 tok/s | **~3.6×** |

**Scope:** L=32 only, fp16 and W8A16 int8, prefill, seq=128. fp16 cliffs at S≥512 (RAM saturation), OOMs at S≥1024. int8 has ~440× less GPU-visible Shmem and 4 GB more MemFree headroom — likely pushes the cliff out, but exploration deferred to a future study.

## Audiences

This folder has two entry points depending on who you are.

### For humans → [`reports/`](reports/)

Prose-y findings docs with breakdowns, tables, comparisons. Read order:

1. [`reports/REPORT.md`](reports/REPORT.md) — fp16 baseline (linear_vec) findings, memory architecture, optimization roadmap
2. [`reports/L32_S128_coopmat_REPORT.md`](reports/L32_S128_coopmat_REPORT.md) — fp16 coopmat findings, speedup decomposition, Amdahl analysis
3. [`reports/L32_S128_int8_baseline_REPORT.md`](reports/L32_S128_int8_baseline_REPORT.md) — **NEW: int8 W8A16 (`vulkan_8w` default) E2E baseline.** 2.108 s steady forward, 60.7 tok/s (1.19× slower than fp16). Half-size `.pte` (8.56 GB), 440× less Shmem, +3.8 GB MemFree headroom. Per-shape regression at FFN gate/up explains the wallclock loss. Documents three upstream patches needed to make `vulkan_8w` work with fp16 LLaMA at all.
4. [`reports/int8_coopmat_microbench.md`](reports/int8_coopmat_microbench.md) — int8 KHR coopmat shader microbench at LLaMA shapes (v1). Validates the user's "coopmat ~4× over non-coopmat" hypothesis (mean R4 = 3.7×, up to 6× at FFN). int8 KHR cm vs fp32 cm = 4-5× at FFN (consistent with H1: ~2× over fp16). Projects ~305 tok/s E2E if `matmul_khr_cm_int8` were wired into the LLaMA linear dispatch (deferred). ⚠ **Superseded by #5 for fp16 cm ratios — uses fp32 cm × 2 as proxy.**
5. [`reports/int8_coopmat_microbench_v2.md`](reports/int8_coopmat_microbench_v2.md) — **Track A Phase 1 (tile-schedule sweep).** Real fp16 cm LLaMA baseline (replaces v1 proxy). Tile-schedule sweep of `matmul_khr_cm_int8_wave64` (TILE_K, subgroup layout, BColMajor). **Best variant: existing baseline. Weighted-by-dispatch ratio = 0.613 → 1.63× over fp16 cm (target was 2×). Phase 2 E2E port NOT started per gate.** Adds spec-constant tuning surface + sweep harness + tight-tolerance validation infrastructure.
6. [`reports/L32_S128_shader_breakdown.md`](reports/L32_S128_shader_breakdown.md) — per-GLSL-shader inventory for fp16 baseline
7. [`reports/L32_S128_coopmat_shader_breakdown.md`](reports/L32_S128_coopmat_shader_breakdown.md) — per-shader inventory for fp16 coopmat
8. [`reports/decode_GEMV_ceiling_check.md`](reports/decode_GEMV_ceiling_check.md) — M=1 (decode-shape) coopmat check at L=4: coopmat does NOT fire at M=1; both paths identical at ~40 ms/forward
9. [`reports/L32_decode_step_breakdown.md`](reports/L32_decode_step_breakdown.md) — L=32 *no-cache* decode-shape breakdown. 310.6 ms / step. ⚠️ **Superseded by #10 for throughput numbers** — the no-cache proxy underestimated by 16×.
10. [`reports/L32_real_decode_benchmark.md`](reports/L32_real_decode_benchmark.md) — **AUTHORITATIVE: real autoregressive decode** with `use_kv_cache=True`. **5.0 s / step → 0.20 tok/s; 1024-step total ≈ 85 min.** 78% of wallclock is memory-wait outside GPU dispatch (page-cache eviction + CPU-fallback `index_put`). Manager-spec'd answer.
11. Three HTML reports in `reports/` for visual comparison of the fp16 work

### For AI agents → [`ai/`](ai/)

Operational docs designed for a fresh agent picking up this work. Captures the methodology + gotchas we discovered during the session, so future agents don't re-derive them. Read order:

1. [`ai/AGENT_PRIMER.md`](ai/AGENT_PRIMER.md) — **START HERE** — single-page orientation, environment facts, confirmed numbers, read-order pointers
2. [`ai/METHODOLOGY.md`](ai/METHODOLOGY.md) — how to bench correctly (scientific mode, ETDump usage, anti-patterns)
3. [`ai/SHADER_DISPATCH.md`](ai/SHADER_DISPATCH.md) — how Vulkan picks shader variants, coopmat dispatch gates, storage type decisions
4. [`ai/COOPMAT_WORKFLOW.md`](ai/COOPMAT_WORKFLOW.md) — exact step-by-step reproduction of the coopmat experiment
5. [`ai/GOTCHAS.md`](ai/GOTCHAS.md) — bugs, traps, methodological mistakes we made and corrected (13 items, each with symptom → cause → fix)
6. [`ai/INSTRUCTIONS.md`](ai/INSTRUCTIONS.md) — terse quick-reference commands

A fresh agent can be told `"read yanwen/ai/AGENT_PRIMER.md and follow the read order"` and have everything needed to reproduce the experiment and avoid re-deriving the pitfalls.

## Other contents

- [`scripts/`](scripts/) — Python bench scripts (`run_llama31_pure.py`, `setup_llama31_pure.py`, `bench_llama31_pure.py`, analyzers); coopmat variants in [`scripts/coopmat/`](scripts/coopmat/); **int8 variants and microbench drivers in [`scripts/int8/`](scripts/int8/)**
- [`artifacts/L32/`](artifacts/L32/), [`artifacts/L32_coopmat/`](artifacts/L32_coopmat/), and **[`artifacts/L32_int8/`](artifacts/L32_int8/)** — captured ETDumps, events.tsv, memprobe, bench logs. Plus **[`artifacts/int8_microbench/`](artifacts/int8_microbench/)** with per-binary microbench output for Phase 2 of the int8 study.
- [`old/`](old/) — superseded 2026-05-07 session doc (outdated methodology, kept for historical context)
