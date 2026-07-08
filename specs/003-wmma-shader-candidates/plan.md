# Implementation Plan: WMMA-Optimizable Shader Candidates Report

**Branch**: `003-wmma-shader-candidates` | **Date**: 2026-07-04 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/003-wmma-shader-candidates/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Classify every shader in `002`'s per-config profiling data (already-aggregated kernel entries) into one of four WMMA-candidacy buckets, cite the specific existing/blocking/missing shader for each, and produce a single ranked "optimization candidates" report across all six configurations — primarily sorted by absolute time, with relative percentage-of-phase shown alongside (per the Clarifications session). This is a pure analysis layer: no new profiling, no shader code, no device access at all — it reads `002`'s existing JSON plus the actual dispatch source (`QuantizedLinear.cpp`, `SDPA.cpp`) as ground truth. Critically, verifying the classification logic against real code turned up **two independent blockers** for the prefill linear family, not one: the real model's linear-layer output tensors are rank-3 (`[1, M, K]`), which fails `can_use_q4gsw_coopmat()`'s `dim_of(output) > 2` "batched output" guard *before* the function even reaches its separate `storage_type_of(output) != kBuffer` check — either alone would already block coopmat, independent of `001`'s `ET_VK_FORCE_TILED_LINEAR` toggle.

## Technical Context

**Language/Version**: Python 3.10+ (in the project's `uv`-managed `.venv`), matching `001`/`002`. No C++ build or device access needed at all for this feature.

**Primary Dependencies**: `002-etdump-shader-profiling`'s `results/raw/<model>_<scheme>.json` files (the `aggregated` and `category_rollup` arrays, already computed); the actual dispatch source — `backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp` (`can_use_q4gsw_coopmat`, `pick_linear_qw_shader`, `pick_linear_dqa_qw_shader`) and `backends/vulkan/runtime/graph/ops/impl/SDPA.cpp` — read directly as the source of truth for classification reasons, not re-derived from data alone.

**Storage**: Flat files only — a classification JSON per configuration and one consolidated ranked markdown report, under this feature's own `results/` directory. No database, no re-profiling, no new `.pte`/`.etdump` artifacts.

**Testing**: Manual verification — every "exists but blocked" or "no WMMA implementation" classification must be checked against the cited source file:line before being accepted; total time summed per classification group must reconcile against `002`'s phase totals.

**Target Platform**: None required — this feature needs no MiniPC/GPU access; it is pure data + source-code analysis, runnable anywhere the repo and `002`'s results are checked out.

**Project Type**: Internal analysis/reporting layer, third in this workstream's series (`001` → `002` → `003`) — no new project, no code changes.

**Performance Goals**: None — this feature identifies optimization targets, it does not implement or benchmark any of them.

**Constraints**: MUST NOT re-profile or re-capture anything; every classification MUST cite a real, checked source location, not a guess; classification-(b) vs classification-(c) MUST be determined by actually reading the relevant dispatch function, not inferred solely from the profiling data's shape/kernel-name patterns.

**Scale/Scope**: 6 configurations × 2 phases × ~8-10 distinct kernel+shape entries per phase (per `002`) ≈ ~100-120 fine-grained Shader Classification rows, rolled up into a small number (single digits) of ranked Optimization Candidate groups by shared root cause.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against `.specify/memory/constitution.md` (Vulkan Cooperative-Matrix (WMMA) GEMM Constitution, v1.1.0):

- **I. Correctness Before Performance** — PASS (not applicable to relax or violate — no code changes at all in this feature).
- **II. Samsung RDNA3 iGPU Is the Target, Not a Fallback** — PASS. Analyzes the same `rocky-ryzen` proxy data already validated in `001`/`002`; introduces no new hardware claims.
- **III. Explicit Eligibility Gating, Safe Fallback Always** — PASS, and this feature actively *reinforces* the principle: it exists specifically to document today's eligibility gates (`can_use_q4gsw_coopmat`'s checks) faithfully and completely, including the newly-found second blocker, rather than presenting an incomplete picture.
- **IV. Two-Tier, Statistically Sound Benchmarking** — PASS. Builds strictly on `001`/`002`'s already-validated, statistically-sound measurements; invents no new unvalidated numbers.
- **V. Document Every Driver Workaround at the Point of Use** — N/A (no driver-facing code touched).

No violations identified. Complexity Tracking table is not needed.

*Post-Phase-1 re-check (after research.md, data-model.md, contracts/, quickstart.md were produced):* the two-blocker correction (found by reading the actual code before finalizing Phase 0) is now the authoritative version threaded through data-model.md and the contract; no new dependencies or violations were introduced in Phase 1.

## Project Structure

### Documentation (this feature)

```text
specs/003-wmma-shader-candidates/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md         # Phase 1 output (/speckit-plan command)
├── quickstart.md         # Phase 1 output (/speckit-plan command)
└── contracts/            # Phase 1 output (/speckit-plan command)
```

### Source Code (repository root)

```text
# No product/backend code touched by this feature at all -- pure analysis.

specs/003-wmma-shader-candidates/
├── scripts/
│   └── classify_shaders.py    # reads 002's results/raw/*.json, applies the
│                               # classification rule set (research.md
│                               # Decision 2), writes per-config classification
│                               # JSON + the consolidated ranked report
└── results/
    ├── classifications/
    │   └── <model>_<scheme>.json   # per-config Shader Classification rows
    └── wmma-candidates-report.md   # US3's ranked, consolidated deliverable

# Read-only references (not modified):
specs/002-etdump-shader-profiling/results/raw/*.json
backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp
backends/vulkan/runtime/graph/ops/impl/SDPA.cpp
```

**Structure Decision**: Single-project, analysis-only layout, matching `002`'s
shape (one parsing/classification script, one `results/` directory). No
source code changes anywhere in the repo; no rebuild or device access
required to run this feature end to end.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations — this section is intentionally empty.
