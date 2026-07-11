# Implementation Plan: MiniPC No-WMMA Baseline Benchmarks

**Branch**: `001-minipc-baseline-benchmarks` | **Date**: 2026-07-03 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-minipc-baseline-benchmarks/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Establish credible, statistically sound "before WMMA" baseline numbers on the `rocky-ryzen` RDNA3 MiniPC for the coopmat/WMMA workstream: end-to-end (decode + prefill tokens/sec, at a fixed 2048-token prefill / 1024-token decode) and shader-level microbenchmark (real per-model GEMM/GEMV shapes) results, for Llama 3.1 8B / 3.2 3B / 3.2 1B at the `4w` and `8da4w` int4 quantization schemes — six configurations total, all measured with the coopmat/WMMA dispatch path excluded. Technical approach: reuse existing project infrastructure end-to-end (the `export_llama` CLI, the standard `llama_main` runner and its `Stats` reporting, and the `backends/vulkan/test/custom_ops` microbenchmark harness) rather than building new tooling, and add one small, off-by-default runtime toggle so the *same* build/binary produces both this baseline and, later, the WMMA-enabled numbers — keeping the two directly comparable.

## Technical Context

**Language/Version**: C++17 (Vulkan backend, runner, microbenchmark harness) and Python 3.10+ (export tooling), matching the rest of the repo; all Python work runs inside the project's `uv`-managed `.venv` per the coopmat/WMMA constitution's Environment & Build Bootstrap section.

**Primary Dependencies**: ExecuTorch Vulkan backend (`backends/vulkan`); `export_llama` CLI (`examples/models/llama/export_llama_lib.py`, `-qmode 4w|8da4w`, `-V/--vulkan`); the standard LLM runner (`examples/models/llama/main.cpp` + `extension/llm/runner/stats.h`) for e2e tokens/sec; the existing coopmat/tiled microbenchmark harness (`backends/vulkan/test/custom_ops`, `BenchmarkResult` in `utils.h`/`utils.cpp`).

**Storage**: Flat files only — `.pte` exports, per-model tokenizer-verified 2048-token prompt files, and the Baseline Report (raw JSON + a rendered markdown summary) under this feature's `results/` directory. No database.

**Testing**: `ctest` for the C++ microbenchmark binary (consistent with how `backends/vulkan/test/custom_ops` is already built/run); e2e numbers are validated by repetition against the constitution's statistical-soundness bar rather than a pass/fail unit test.

**Target Platform**: Linux MiniPC (`rocky-ryzen`, AMD Ryzen APU, RDNA3 integrated GPU), built per the constitution's Reference Build Recipe (`--preset "linux"`, `EXECUTORCH_BUILD_VULKAN=ON`).

**Project Type**: Internal measurement/tooling effort inside the existing ExecuTorch monorepo — no new project, no frontend/backend split.

**Performance Goals**: None — this feature *defines* the goalposts (the "before" numbers) that a future feature will compare a WMMA-enabled build against; there is no latency/throughput target to hit here.

**Constraints**: Every measurement MUST run with the coopmat/WMMA dispatch path excluded (spec FR-004); baseline and future WMMA-enabled numbers MUST come from the same build/binary configuration to avoid confounding the comparison; all measurements MUST follow the constitution's statistically-sound methodology (explicit warmup, multiple iterations, reported variance).

**Scale/Scope**: 3 models × 2 quantization schemes = 6 benchmark configurations, 1 device, fixed 2048-token prefill / 1024-token decode for the e2e tier (per spec Clarifications).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against `.specify/memory/constitution.md` (Vulkan Cooperative-Matrix (WMMA) GEMM Constitution, v1.1.0):

- **I. Correctness Before Performance** — PASS. This feature only measures the already-validated tiled dispatch path; the one code change it introduces (Research Decision 1's toggle) is a no-op when unset, so it cannot regress correctness of the existing coopmat path.
- **II. Samsung RDNA3 iGPU Is the Target, Not a Fallback** — PASS. The constitution explicitly names `rocky-ryzen` as "the primary local dev/test platform, used before every Android build"; this feature operates entirely within that already-endorsed scope and does not claim MiniPC results as a substitute for eventual on-device Samsung validation.
- **III. Explicit Eligibility Gating, Safe Fallback Always** — PASS. The new dispatch-exclusion toggle (Research Decision 1) is itself an explicit, off-by-default, testable gate — it adds a control point rather than relaxing an existing one.
- **IV. Two-Tier, Statistically Sound Benchmarking** — PASS (this is what the feature implements). Both the microbenchmark tier (via the existing `BenchmarkResult` harness) and the model-level tier (via `llama_main`/`stats.h`) are used, at real shapes/models, with variance reported.
- **V. Document Every Driver Workaround at the Point of Use** — N/A for now (no new driver-specific workaround is anticipated for the already-shipping tiled path); if one is discovered during this work, it must be documented inline per this principle before being merged.

No violations identified. Complexity Tracking table is not needed.

*Post-Phase-1 re-check (after research.md, data-model.md, contracts/, quickstart.md were produced):* the design added no new dependencies, dispatch paths, or gates beyond the single toggle already evaluated above; all five principles remain satisfied and the report schema in `contracts/baseline-report-schema.md` explicitly encodes Principle IV's "no lone mean" and Principle II's `dispatch_path` labeling requirements. No new violations.

## Project Structure

### Documentation (this feature)

```text
specs/001-minipc-baseline-benchmarks/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp
  # + one off-by-default runtime toggle (env var) to force the tiled
  # (non-coopmat) dispatch path for controlled baseline measurement

backends/vulkan/test/custom_ops/
  # + one new benchmark source reusing BenchmarkResult/utils.h, parameterized
  # by each target model's real prefill (M=2048) and decode (M=1) GEMM/GEMV
  # shapes for the 4w and 8da4w schemes, dispatched tiled-only

examples/models/llama/
  # no code changes — existing export_llama CLI (-qmode 4w|8da4w, -V) and
  # the existing llama_main runner + stats.h (prefill/decode tokens/sec) are
  # invoked as-is for the six .pte exports and six e2e runs

specs/001-minipc-baseline-benchmarks/
├── plan.md                    # this file
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── baseline-report-schema.md
└── results/                   # produced during implementation, not planning
    ├── baseline-report.md
    ├── raw/                   # one JSON per (model, scheme) with e2e + microbench data
    └── prompts/                # per-model, tokenizer-verified 2048-token prompt files
```

**Structure Decision**: Single-project layout — this feature adds one small,
reversible C++ change inside the existing Vulkan backend, one new benchmark
binary inside the existing `custom_ops` test suite, and produces its own data
artifacts under the feature's `specs/` directory. No new top-level project,
service, or frontend is introduced.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations — this section is intentionally empty.
