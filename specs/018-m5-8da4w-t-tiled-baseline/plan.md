# Implementation Plan: M5 EVT1 8da4w T-tiled Baseline

**Branch**: `018-m5-8da4w-t-tiled-baseline` | **Date**: 2026-07-06 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/018-m5-8da4w-t-tiled-baseline/spec.md`

## Summary

`4w`'s speedup table has a real "vs T-tiled baseline" ratio for every
model; `8da4w`'s does not, because no T-tiled (texture-storage, default,
non-coopmat) `8da4w` PTE has ever been exported at the standard `ctx3072`
context length for any of the three target models. This feature exports
one per model, measures each at the standard 2048-prefill/1024-decode
workload with the same rigor already applied to every other number in
`specs/015` (pinned+verified clocks, 3-run mean+CoV, tool-verified tiled
dispatch), and folds the resulting ratios into the consolidated report.
No shader or production dispatch-logic code changes -- purely export +
measure + report, following this workstream's own established pattern
for this class of feature (e.g. `specs/015`, `specs/016`).

## Technical Context

**Language/Version**: N/A for new code -- this feature runs existing
export/build/measurement tooling, does not write new source.

**Primary Dependencies**: This repo's existing export pipeline
(`export_llm`, `backend.vulkan.storage_override` config key -- omitted
here, since T-tiled means the *default*, texture-storage behavior),
this repo's already-built `llama_main`/ETDump Android runner (no rebuild
needed -- this feature doesn't touch source), `.shared-context/scripts/analyze_etdump_shaders.py`
for dispatch verification.

**Storage**: New `.pte` files only -- `llama3_2_1b_8da4w_texture_ctx3072.pte`,
`llama3_2_3b_8da4w_texture_ctx3072.pte`, `llama3_1_8b_8da4w_texture_ctx3072.pte`,
landing in `/local/yanwen.xu/workspace/.pte_out` per constitution's Default
Scope rule (never `/tmp` or a scratch dir, per gotcha G4/G5).

**Testing**: No automated test suite -- verification is the same
tool-driven dispatch confirmation (Principle VI) and clock-pin
cross-check (Principle VII) already used throughout `specs/014-017`,
described in `quickstart.md`.

**Target Platform**: Samsung M5 EVT1 (Exynos 2500 / Xclipse 970),
constitution Principle II's sole active target -- same as every other
feature in this workstream.

**Project Type**: Measurement/reporting only -- no source tree changes.

**Performance Goals**: N/A -- this feature measures an existing
(unoptimized) code path's baseline throughput, it does not target a
performance number of its own.

**Constraints**: Must reuse the `4w` T-tiled baseline's exact methodology
(FR-006) so the two schemes' ratios are comparable; must not touch
shader/dispatch-logic source (Assumptions); must confirm tiled dispatch
via tooling, not assume it from the PTE's storage-type filename alone
(FR-004, directly informed by this workstream's own G6/Q11 history of
ETDump attribution being unreliable in some contexts).

**Scale/Scope**: 3 new PTE exports, 3 measurement runs (3-rep each = 9
timed runs total), plus dispatch-confirmation captures and report
updates across `specs/015`'s results files.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Check | Status |
|---|---|---|
| I. Correctness Before Performance | N/A for T-tiled itself (it's the pre-existing, already-correctness-tested tiled path, not a new shader) -- this feature makes no correctness claim about new code. | PASS (N/A) |
| II. Samsung M5 EVT1 Sole Target | Measured exclusively on M5 EVT1, consistent with every other feature. | PASS |
| III. Explicit Eligibility Gating, Safe Fallback | N/A -- no new gating code; T-tiled is the existing, always-available fallback path itself. | PASS (N/A) |
| IV. Two-Tier, Statistically Sound Benchmarking | This feature is tier-2 (model-level) only, matching the existing `4w` T-tiled baseline's own tier -- 3-run mean+CoV, separate dispatch-confirmation run per FR-002/FR-004. | PASS |
| V. Document Every Driver Workaround | N/A -- no new driver workaround introduced; reuses the already-documented `ET_VK_EXECUTE_NODE_THRESHOLD=16` 8B watchdog workaround if needed (Edge Cases). | PASS (N/A) |
| VI. Verify With Tools, Never Assume | FR-004/SC-004 explicitly require tool-verified tiled dispatch, not an assumption from the PTE's texture-storage filename -- directly applying this session's own G6 lesson. | PASS |
| VII. Clock Discipline | FR-002/FR-003 require pinned clocks, verified bound via GFLOP/s cross-check, same as every other tier-2 number in this workstream. | PASS |
| VIII. Verify Driver Before Every Coopmat Measurement | FR-003 requires re-verifying driver identity before measuring -- note this feature measures the *tiled* path specifically, but the same discipline applies since it's still a real hardware measurement on a shared device. | PASS |
| IX. Never Disclose Samsung-Internal Specifics Upstream | N/A -- this feature produces internal workspace reports (`specs/015`'s results files, this repo's own `.pte_out`), never proposed upstream. | PASS (N/A) |
| X. Consult `instruction-for-ai` Before Acting | Export follows `.shared-context/instruction-for-ai/export-pte.md`'s real mechanism (default/no `storage_override`, per gotcha G2's caution that the doc's `ET_VK_FORCE_BUFFER` env var doesn't exist) -- consult `.specify/memory/gotchas.md` first, per this workstream's own X amendment. | PASS |

No violations; Complexity Tracking not needed.

**Post-Phase-1 re-check**: `data-model.md`/`quickstart.md` introduced no
new gate risk -- the T-tiled Baseline Measurement entity and the
quickstart's validation checks stay within what Phase 0 already
justified. Constitution Check still PASSES across all ten principles.

## Project Structure

### Documentation (this feature)

```text
specs/018-m5-8da4w-t-tiled-baseline/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md         # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit-tasks, not this command)
```

No `contracts/` -- this feature has no external interface (API, CLI, data
format) of its own to document a contract for; it exports PTEs and runs
the existing runner binary with existing flags.

### Source Code (repository root)

```text
/local/yanwen.xu/workspace/.pte_out/
├── llama3_2_1b_8da4w_texture_ctx3072.pte   # NEW
├── llama3_2_3b_8da4w_texture_ctx3072.pte   # NEW
└── llama3_1_8b_8da4w_texture_ctx3072.pte   # NEW

specs/015-m5-e2e-wmma-validation/results/
├── 1b-results.md                          # MODIFIED: 8da4w row gets a real ratio
├── 3b-results.md                          # MODIFIED: same
├── 8b-results.md                          # MODIFIED: same
└── m5-e2e-validation-report.md            # MODIFIED: consolidated table, all 12 rows real
```

**Structure Decision**: No new source directories. Three new `.pte` files
in the workspace's canonical `.pte_out` (per constitution's Default
Scope rule), and targeted edits to `specs/015`'s existing results files
(this feature's numbers complete that spec's table, they don't warrant a
second copy of the same table living in `specs/018`).

## Complexity Tracking

*No violations -- table not needed.*
