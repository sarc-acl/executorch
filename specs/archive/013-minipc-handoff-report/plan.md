# Implementation Plan: MiniPC RDNA3 Handoff Report

**Branch**: `013-minipc-handoff-report` | **Date**: 2026-07-05 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/013-minipc-handoff-report/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Consolidate everything concluded across specs `001`-`012` into one report,
so the `rocky-ryzen` MiniPC phase of this workstream can be handed off
cleanly before moving to Samsung/Xclipse hardware. Planning already
gathered the exact headline numbers from every prior spec's own results
file (research.md Decision 1) and the actual current repo state (branch
`quant-perf-optimization`, last commit `d8800fb02e`, 71 files of
uncommitted work spanning specs `007`-`013` and their underlying
production-code fixes -- research.md Decision 3). Per the user's own
Clarifications, the report is high-level (not an exhaustive script) and
does not address the gitignored `.pte` export files at all -- those are
re-exported independently on the new machine.

## Technical Context

**Language/Version**: Markdown only -- no code, no scripts. This is the
first feature in this workstream with zero Python/C++/GLSL surface.

**Primary Dependencies**: Every prior spec's own `results/` files (read
-only sources for the Consolidated Findings table) and `git status`/
`git log` (for the Repo Handoff State section).

**Storage**: A single flat report file,
`specs/013-minipc-handoff-report/results/handoff-report.md`.

**Testing**: N/A -- no code. Validation is spot-checking cited numbers
against their source files (FR-002) and confirming the repo-state
description against a fresh `git status`/`git log` (quickstart.md).

**Target Platform**: N/A for this feature's own execution (it produces a
document); the document's subject matter spans both `rocky-ryzen` (fully
covered) and Samsung/Xclipse (runbook only, not executed here).

**Project Type**: Documentation/consolidation -- no build targets, no new
production or test code, the lightest-weight feature in this workstream
alongside `011`.

**Performance Goals**: N/A.

**Constraints**:
- No commit/push is performed by this feature -- named as a prerequisite
  only (FR-004, per this repo's "commit only when explicitly asked"
  convention).
- No `.pte` export handling -- out of scope per Clarifications.
- No actual Android/`adb` build or Samsung/Xclipse device testing -- this
  session's environment has no access to that device (spec.md
  Assumptions); the runbook is a starting checklist, not a validated
  pipeline.
- Report content stays high-level per Clarifications -- reasonable
  defaults and pointers to existing `quickstart.md` files, not asserted
  untested commands.

**Scale/Scope**: Twelve prior features' worth of findings, consolidated
into one document; one repo-state snapshot; one runbook section with a
handful of methodology steps (export, build, dispatch-confirm, benchmark,
report).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against `.specify/memory/constitution.md` (v1.4.0):

- **I. Correctness Before Performance (NON-NEGOTIABLE)**: N/A -- this
  feature makes no performance claims of its own; it only cites already
  -validated numbers from prior features, each of which already satisfied
  this principle on its own.
- **II. Samsung RDNA3 iGPU Is the Target, Not a Fallback**: PASS, directly
  on-point -- this feature's own runbook section exists specifically
  because every finding to date is MiniPC-only, and explicitly refuses to
  assume any of it transfers to the real target device unchanged (FR-006).
- **III. Explicit Eligibility Gating, Safe Fallback Always**: N/A -- no
  dispatch/gating logic touched.
- **IV. Two-Tier, Statistically Sound Benchmarking**: PASS -- the
  Consolidated Findings table explicitly tags each headline by tier
  (microbenchmark vs. e2e, research.md Decision 1), preserving this
  distinction rather than flattening it away in summary.
- **V. Document Every Driver Workaround at the Point of Use**: N/A -- no
  driver workaround introduced by this feature.
- **VI. Verify With Tools, Never Assume**: PASS, applied during planning
  itself -- every headline number was re-read from its source file
  (research.md Decision 1), and the repo state was read via `git`
  directly (Decision 3), rather than recalled from conversation memory.

No violations identified. Complexity Tracking is not needed.

*Post-Phase-1 re-check*: Phase 1's data model keeps `source_file` as a
required field on every Consolidated Finding (data-model.md), so FR-002's
traceability requirement is structurally enforced, not just a writing
convention to remember.

## Project Structure

### Documentation (this feature)

```text
specs/013-minipc-handoff-report/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md              # Phase 2 output (/speckit-tasks, not this command)
```

No `contracts/` -- this feature has no external interface to document
(plan-template guidance: skip for purely internal deliverables).

### Source Code (repository root)

No production or test code -- this feature's only output is a report:

```text
specs/013-minipc-handoff-report/
└── results/
    └── handoff-report.md   # Consolidated Findings, open items, Repo
                              # Handoff State, Samsung/Xclipse Runbook

# Read-only references (not modified):
specs/001-minipc-baseline-benchmarks/ through specs/012-decode-wmma-feasibility/  # cited, not re-opened
```

**Structure Decision**: The lightest-weight structure in this
workstream -- a single report file, no scripts, no `contracts/`, no
build targets. Matches the feature's own nature: a consolidation
deliverable, not a measurement or implementation one.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations -- this section is intentionally empty.
