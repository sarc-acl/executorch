# Implementation Plan: M5 EVT1 Floating-Clock Speedup Table

**Branch**: `019-m5-floating-clock-speedup` | **Date**: 2026-07-06 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/019-m5-floating-clock-speedup/spec.md`

## Summary

Reproduce the existing 6-row pinned-clock speedup table (T-tiled baseline
vs full-stack optimal, 3 models x 2 schemes, `specs/015`/`018`) under
floating (unpinned) clocks. Reuses every PTE and dispatch-confirmation
result already established -- clock state doesn't affect which shader
dispatches, only how fast it runs. The one new methodological concern
this feature must handle correctly (not present in the pinned work):
constitution Principle VII's documented asymmetric throttle behavior
(tiled configs drop -19% to -27% run-to-run under sustained floating
load; coopmat configs stay flat, <4%) means a naive blended mean would
misstate the floating speedup ratio in coopmat's favor. Per-rep
reporting and an explicit cold-start-vs-steady-state comparison basis are
therefore required, not optional polish.

## Technical Context

**Language/Version**: N/A for new code -- reuses existing PTEs and
runner binaries, no source or export changes.

**Primary Dependencies**: Existing PTEs from `specs/015`/`018`'s
`.pte_out` (all 6 T-tiled baselines + 6 full-stack-optimal configs
already exported); this repo's already-built `llama_main` +
ETDump-enabled runner (no rebuild); the unpin equivalent of
`pin_freqs.sh` (write hardware min to `min_freq`, hardware max to
`max_freq`, per `.shared-context/instruction-for-ai/commands.md`).

**Storage**: No new `.pte` files. Raw per-rep floating-clock logs land in
`specs/019-m5-floating-clock-speedup/results/raw/`, following this
workstream's existing convention (`specs/015`'s `results/raw/`).

**Testing**: No automated test suite -- verification is the sysfs
readback confirming genuinely-floating clocks (FR-004) and the reused
dispatch-confirmation status from `specs/015`/`018` (FR-006), described
in `quickstart.md`.

**Target Platform**: Samsung M5 EVT1 (Exynos 2500 / Xclipse 970),
constitution Principle II's sole active target.

**Project Type**: Measurement/reporting only -- no source tree changes.

**Performance Goals**: N/A -- this feature measures existing
(already-optimized and already-baseline) code paths' floating-clock
throughput, it does not target a new performance number of its own.

**Constraints**: Must reuse the pinned work's exact PTEs/workload
(FR-001, FR-005) -- no new export; must report per-rep values, not a
blended mean, for any config showing meaningful run-to-run variation
(FR-002); every number must be labeled floating, never presented as or
alongside the pinned headline without that label (FR-003); must verify
genuinely-floating clock state via sysfs before trusting any capture
(FR-004).

**Scale/Scope**: 12 measurement points (3 models x 2 schemes x 2 config
types: T-tiled baseline, full-stack optimal), 3 reps each = 36 timed runs
total, plus one consolidated report update.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Check | Status |
|---|---|---|
| I. Correctness Before Performance | N/A -- no new shader/code; correctness of these code paths already established in `specs/015`/`018`. | PASS (N/A) |
| II. Samsung M5 EVT1 Sole Target | Measured exclusively on M5 EVT1, consistent with every other feature. | PASS |
| III. Explicit Eligibility Gating, Safe Fallback | N/A -- no new gating code; dispatch status reused, not re-derived. | PASS (N/A) |
| IV. Two-Tier, Statistically Sound Benchmarking | Tier-2 (model-level) only, matching the pinned baseline's own tier; FR-002's per-rep requirement is this feature's version of "statistically sound" given the known non-i.i.d. throttle behavior floating introduces. | PASS |
| V. Document Every Driver Workaround | N/A -- no new driver workaround. | PASS (N/A) |
| VI. Verify With Tools, Never Assume | FR-004 requires a sysfs readback to verify genuinely-floating clocks before trusting a capture -- directly applying this principle to the floating-specific failure mode (a "floating" run that's actually still capped). | PASS |
| VII. Clock Discipline | This feature exists BECAUSE of Principle VII -- floating runs explicitly permitted "whenever explicitly requested" (this is that request), FR-002/FR-003/FR-007 directly implement its per-rep-reporting and labeling requirements. | PASS |
| VIII. Verify Driver Before Every Coopmat Measurement | FR-005 requires re-verifying driver identity before measuring, per standing discipline. | PASS |
| IX. Never Disclose Samsung-Internal Specifics Upstream | N/A -- internal workspace report only, never proposed upstream. | PASS (N/A) |
| X. Consult `instruction-for-ai` Before Acting | Unpin procedure follows `.shared-context/instruction-for-ai/commands.md`'s documented floating-clock method (write hardware min/max to the frequency files), not an invented approach. | PASS |

No violations; Complexity Tracking not needed.

**Post-Phase-1 re-check**: `data-model.md`/`quickstart.md` introduced no
new gate risk -- the Floating Clock Measurement entity and quickstart's
validation checks stay within what Phase 0 already justified.
Constitution Check still PASSES across all ten principles.

## Project Structure

### Documentation (this feature)

```text
specs/019-m5-floating-clock-speedup/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit-tasks, not this command)
```

No `contracts/` -- no external interface of its own; reuses existing
runner binaries and PTEs, no new API/CLI/data format to document.

### Source Code (repository root)

```text
specs/019-m5-floating-clock-speedup/
├── results/
│   ├── raw/                          # NEW: per-rep floating-clock logs
│   ├── 1b-floating-results.md        # NEW
│   ├── 3b-floating-results.md        # NEW
│   ├── 8b-floating-results.md        # NEW
│   └── floating-vs-pinned-report.md  # NEW: consolidated 6-row table + caveat
```

No changes to `.pte_out/` or `specs/015`/`018`'s own files -- this
feature's results are a new, parallel report, not an edit to the pinned
one (the pinned table stays as the headline; floating sits alongside it,
per FR-003/User Story 4).

**Structure Decision**: New results directory under this feature's own
`specs/019.../`, mirroring `specs/015`'s `results/` shape but kept
separate from the pinned report rather than merged into it -- floating
and pinned are different measurement regimes per Principle VII, not two
rows of the same table.

## Complexity Tracking

*No violations -- table not needed.*
