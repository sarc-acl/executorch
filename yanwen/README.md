# yanwen — LLaMA 3.1 8B prefill on AMD Radeon 780M iGPU

End-to-end benchmark of pure (non-quantized) LLaMA 3.1 8B fp16 prefill on the AMD Radeon 780M, comparing the baseline `linear_vec` shader against the `linear_coopmat` (KHR cooperative matrix) shader.

**Headline result at L=32, seq=128:**

| | Steady forward | Throughput | Speedup |
|---|---:|---:|---:|
| Baseline (`linear_vec`) | 1.766 ± 0.006 s | 72.5 tok/s | 1.0× |
| Coopmat (`linear_coopmat`) | 0.583 ± 0.002 s | 219.7 tok/s | **3.03×** |

**Scope:** L=32 only, fp16, prefill, seq=128 (the only performant config — S≥512 cliffs due to RAM saturation, S≥1024 OOMs).

## Audiences

This folder has two entry points depending on who you are.

### For humans → [`reports/`](reports/)

Prose-y findings docs with breakdowns, tables, comparisons. Read order:

1. [`reports/REPORT.md`](reports/REPORT.md) — baseline (linear_vec) findings, memory architecture, optimization roadmap
2. [`reports/L32_S128_coopmat_REPORT.md`](reports/L32_S128_coopmat_REPORT.md) — coopmat findings, speedup decomposition, Amdahl analysis
3. [`reports/L32_S128_shader_breakdown.md`](reports/L32_S128_shader_breakdown.md) — per-GLSL-shader inventory for baseline
4. [`reports/L32_S128_coopmat_shader_breakdown.md`](reports/L32_S128_coopmat_shader_breakdown.md) — per-shader inventory for coopmat
5. Three HTML reports in `reports/` for visual comparison

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

- [`scripts/`](scripts/) — Python bench scripts (`run_llama31_pure.py`, `setup_llama31_pure.py`, `bench_llama31_pure.py`, analyzers); coopmat variants in [`scripts/coopmat/`](scripts/coopmat/)
- [`artifacts/L32/`](artifacts/L32/) and [`artifacts/L32_coopmat/`](artifacts/L32_coopmat/) — captured ETDumps, events.tsv, memprobe, bench logs (symlinks to `/home/doremy/llama31_pure_run*/`)
- [`old/`](old/) — superseded 2026-05-07 session doc (outdated methodology, kept for historical context)
