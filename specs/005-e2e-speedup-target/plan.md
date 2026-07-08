# Implementation Plan: End-to-End Speedup Target and Validation

**Branch**: `005-e2e-speedup-target` | **Date**: 2026-07-04 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/005-e2e-speedup-target/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Formalize this workstream's success target — at least 2x (100%) prefill tok/s improvement per (model, scheme) configuration, with combined e2e tok/s tracked but not judged pass/fail (per the Clarifications session, since decode is ~90% of e2e wall-clock time and has no identified fix) — as a machine-readable artifact tied to `001`'s exact baseline numbers, and build a validated outcome-comparison tool ready to run the moment a real optimization build exists. This feature does **not** implement any optimization and cannot produce real "after" numbers yet — its buildable, honest scope is: record the target now, and prove the comparison/verdict logic correct against synthetic self-test scenarios before it is ever trusted on real future data.

## Technical Context

**Language/Version**: Python 3.10+ (`uv`-managed `.venv`), matching every prior tool in this workstream.

**Primary Dependencies**: `001`'s existing `results/raw/<model>_<scheme>.json` schema (`e2e.prefill_tokens_per_sec`, `e2e.decode_tokens_per_sec`, `e2e.prefill_tokens`, `e2e.decode_tokens`, `e2e.num_runs`, `e2e.variance` — already produced for all six baseline configs). No new capture format is introduced; any future real re-measurement reuses this exact schema and `001`'s exact capture procedure.

**Storage**: Flat files only — `specs/005-e2e-speedup-target/results/speedup-target.json` (the formalized target, produced now) and `specs/005-e2e-speedup-target/results/selftest/` (synthetic before/after pairs plus the self-test outcome report, produced now to validate the tooling). The real `outcome-report.md` is **not** produced by this planning/implementation pass — there is no real "after" build yet.

**Testing**: A self-test against synthetic re-measurement scenarios (exactly 2x, exceeds, falls short, regresses, non-comparable-methodology) run against all six real baseline configs, proving every branch of the verdict logic fires correctly before it is ever pointed at real future data — the same "prove it on a known case first" discipline established in `002`/`003`.

**Target Platform**: None required to build this feature's actual deliverable (target JSON + validated tool). A **real** execution of User Story 2/3 requires the `rocky-ryzen` MiniPC and a build containing actual optimization work — neither exists yet, so that execution is explicitly out of this pass's scope and deferred to later, separate work.

**Project Type**: Internal tooling/methodology feature — no device access needed to build it, unlike `001`/`004`; similar in shape to `002`/`003`'s pure-analysis tooling, but its full lifecycle (Story 2/3) cannot complete until a future optimization build exists.

**Performance Goals**: N/A for this feature itself — it defines how *other* future work's performance will be judged, not its own.

**Constraints**: MUST NOT fabricate or simulate "after" numbers and present them as real; self-test data MUST be clearly and permanently marked synthetic so it is never mistaken for a real measurement; Story 2/3's real completion MUST remain explicitly gated on future optimization work existing.

**Scale/Scope**: 3 models × 2 quantization schemes (`4w`, `8da4w`) = 6 configurations, matching this workstream's established scope. The spec's "per model" language is satisfied by never averaging across models; reporting stays at the same (model, scheme) granularity every prior feature (`001`-`004`) already used, since `4w`/`8da4w` may see genuinely different coopmat behavior.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against `.specify/memory/constitution.md` (Vulkan Cooperative-Matrix (WMMA) GEMM Constitution, v1.1.0):

- **I. Correctness Before Performance** — PASS. No product code changes; this only defines a target and builds comparison tooling.
- **II. Samsung RDNA3 iGPU Is the Target, Not a Fallback** — PASS. Any future real re-measurement reuses the same `rocky-ryzen` proxy hardware already validated throughout this workstream.
- **III. Explicit Eligibility Gating, Safe Fallback Always** — PASS, and reinforced: the target explicitly documents *why* combined e2e isn't the pass/fail bar (decode's GEMV path has no eligibility path to coopmat at all yet, per `003`), rather than silently ignoring that gap.
- **IV. Two-Tier, Statistically Sound Benchmarking** — PASS, and directly enforced by FR-003: any future re-measurement MUST reuse `001`'s exact statistical methodology (repeated runs, steady-state reporting, no resource contention), not a new, less rigorous one invented for convenience.
- **V. Document Every Driver Workaround at the Point of Use** — N/A.

No violations identified. Complexity Tracking is not needed.

*Post-Phase-1 re-check*: Phase 1's data model and contract keep the self-test data physically separate from (and clearly labeled apart from) any future real outcome report, so there is no risk of the constitution's correctness-before-performance principle being violated by a synthetic number being mistaken for a real one.

## Project Structure

### Documentation (this feature)

```text
specs/005-e2e-speedup-target/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
└── contracts/           # Phase 1 output (/speckit-plan command)
```

### Source Code (repository root)

```text
# No product/backend code touched -- pure target-definition + comparison tooling.

specs/005-e2e-speedup-target/
├── scripts/
│   └── compute_outcome.py     # builds speedup-target.json from 001's baseline;
│                                # compares a baseline+after JSON pair and emits
│                                # a per-config verdict (met/exceeded/missed/
│                                # regressed/not_comparable) + tracked e2e change
└── results/
    ├── speedup-target.json     # Story 1's deliverable -- produced now
    ├── selftest/
    │   ├── synthetic_after_*.json   # 6 configs x 5 synthetic scenarios
    │   └── selftest-outcome-report.md
    └── outcome-report.md       # Story 3's REAL deliverable -- NOT produced by
                                  # this pass; requires a future optimization
                                  # build + real re-measurement to exist first

# Read-only references (not modified):
specs/001-minipc-baseline-benchmarks/results/raw/*.json
```

**Structure Decision**: Single-project, tooling-only layout matching `002`/`003`'s
shape. No source code changes anywhere in the product; no rebuild or device
access required to produce this feature's actual (buildable-now) deliverables.
The real `outcome-report.md` is deliberately absent from this pass's output —
producing it would require fabricating data that doesn't exist yet.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations — this section is intentionally empty.
