# Implementation Plan: ETDump E2E Shader Profiling Breakdown

**Branch**: `002-etdump-shader-profiling` | **Date**: 2026-07-04 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/002-etdump-shader-profiling/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Produce a per-kernel time/shape breakdown (aggregated by kernel+shape, with raw per-invocation data preserved) for prefill and decode of all six baseline configurations from `001-minipc-baseline-benchmarks`, using ETDump. Technical approach: rebuild the existing `llama_main` runner with the event tracer enabled (`-DEXECUTORCH_ENABLE_EVENT_TRACER=ON -DEXECUTORCH_BUILD_DEVTOOLS=ON`), reuse the six `.pte` files already exported in `001` unmodified (event tracing is a runtime/runner build flag, not an export-time concern), capture one ETDump per phase per configuration (prefill at the same 2048 tokens as `001`; decode over a short representative window rather than the full 1024 steps), and parse the dumps with a small Python script. Critically, shape attribution does **not** require the ETRecord/Inspector graph-correlation workflow — the Vulkan delegate already embeds each dispatch's tensor shapes directly into the ETDump event name as JSON (`backends/vulkan/runtime/graph/Logging.cpp`'s `make_operator_json`/`make_arg_json`, gated on the same event-tracer flag), so a single `--etdump_path` capture is sufficient.

## Technical Context

**Language/Version**: C++17 (llama runner rebuild) and Python 3.10+ (ETDump parsing script, in the project's `uv`-managed `.venv`), matching `001`.

**Primary Dependencies**: `examples/models/llama/main.cpp` (`llama_main`, rebuilt with event tracing); `devtools/etdump` (`ETDumpGen`, flatcc schema — already used by `main.cpp`, just needs the build flag); the six `.pte` files and `results/shapes.json` from `001-minipc-baseline-benchmarks` (reused as-is, no re-export).

**Storage**: Flat files only — raw `.etdump` binaries, parsed per-invocation and aggregated JSON, and a rendered markdown report under this feature's `results/` directory. No database, no ETRecord files.

**Testing**: Manual reconciliation checks (aggregated event-time sum vs. phase wall-clock, per FR-005) and shape cross-checks against `001`'s `results/shapes.json`; no automated test suite for a one-shot analysis artifact.

**Target Platform**: Same `rocky-ryzen` MiniPC (AMD Radeon 780M, RDNA3 iGPU) as `001` — no new hardware.

**Project Type**: Internal measurement/analysis effort, companion to `001` — no new project, no frontend/backend split.

**Performance Goals**: None — this feature explains an existing measurement, it does not set a new target.

**Constraints**: Must reuse the existing six `.pte` exports unmodified (no re-export); profiling captures MUST NOT run concurrently with any other CPU/GPU-heavy process (same resource-contention lesson learned the hard way in `001`); decode profiling MUST use a short window, not the full 1024-step decode length, per the spec's Assumptions.

**Scale/Scope**: 6 configurations × 2 phases (prefill, decode-window) = 12 profiling captures, each producing an aggregated breakdown plus raw per-invocation companion data.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against `.specify/memory/constitution.md` (Vulkan Cooperative-Matrix (WMMA) GEMM Constitution, v1.1.0):

- **I. Correctness Before Performance** — PASS. This feature adds no shader or dispatch-logic changes; it only rebuilds the runner with an existing, already-shipping instrumentation flag (`EXECUTORCH_ENABLE_EVENT_TRACER`) and reuses already-validated `.pte` files. No new correctness surface.
- **II. Samsung RDNA3 iGPU Is the Target, Not a Fallback** — PASS. Stays entirely within `rocky-ryzen`'s already-endorsed scope as the local proxy platform; introduces no new hardware claims.
- **III. Explicit Eligibility Gating, Safe Fallback Always** — PASS (not applicable to add/relax — no new dispatch gate is introduced by this feature).
- **IV. Two-Tier, Statistically Sound Benchmarking** — PASS. This feature deepens the microbenchmark/e2e attribution the constitution asks for: it explains *where* the `001` baseline numbers come from, at real shapes, on real hardware. It explicitly reports profiling overhead (FR-006) rather than conflating a profiled run's timing with the un-profiled baseline, consistent with the constitution's "single untimed run is not evidence" spirit — here applied as "a profiled number is not silently the same as an un-profiled one."
- **V. Document Every Driver Workaround at the Point of Use** — N/A expected (pure instrumentation build flag, no shader changes); if enabling the event tracer surfaces a driver quirk on this hardware, it must be documented per this principle before landing.

No violations identified. Complexity Tracking table is not needed.

*Post-Phase-1 re-check (after research.md, data-model.md, contracts/, quickstart.md were produced):* no new dependencies or dispatch paths were introduced beyond the build-flag change already evaluated above; the decision to skip ETRecord (research.md Decision 2) further *reduces* complexity versus the originally-assumed approach. No new violations.

## Project Structure

### Documentation (this feature)

```text
specs/002-etdump-shader-profiling/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
└── contracts/           # Phase 1 output (/speckit-plan command)
```

### Source Code (repository root)

```text
cmake-out-vk-profiling/           # SEPARATE build dir from 001's cmake-out-vk
  # (EXECUTORCH_ENABLE_EVENT_TRACER + EXECUTORCH_BUILD_DEVTOOLS require a
  # rebuild; kept separate so 001's existing build/artifacts are untouched)

specs/002-etdump-shader-profiling/results/
├── etdumps/                       # raw .etdump binaries, one per (config, phase)
│   └── <model>_<scheme>_<phase>.etdump
├── raw/                           # parsed data per config
│   └── <model>_<scheme>.json      # both phases: raw invocations + aggregated entries + category rollup
└── profiling-report.md            # rendered summary, references 001's baseline-report.md

# No changes to backends/vulkan or examples/models/llama source — this
# feature only adds a build-flag combination and a standalone parsing script
# (scripts/ location decided in tasks.md), no product code changes.
```

**Structure Decision**: Single-project, analysis-only layout — no source code
changes to the Vulkan backend or the llama runner; only a differently-configured
build directory and a standalone parsing script. All new data lives under this
feature's own `specs/` directory, reusing `001`'s existing `.pte` files and
`results/shapes.json` in place rather than copying them.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations — this section is intentionally empty.
