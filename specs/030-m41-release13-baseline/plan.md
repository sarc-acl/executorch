# Implementation Plan: M41 Release/1.3 Baseline Clock & Quant-Mode Study

**Branch**: `030-m41-release13-baseline` | **Date**: 2026-07-14 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/030-m41-release13-baseline/spec.md`

## Summary

M41 (a secondary, non-target Exynos s5e9965/ERD9965-family Samsung device, per constitution
Principle II's Reference Hardware Inventory) already has a floating-clock, release/1.3-vanilla
4w-texture (T-tiled) baseline captured this session for all three target models — 1B/3B/8B, up to
3 reps each, one 8B rep crashed. This feature (a) preserves that dataset plus the device's driver
identity in a durable document, then (b) fills in the remaining 3 of 4 quant-mode × clock-mode
cells — 4w-pinned, 8da4w-floating, 8da4w-pinned — at 3 reps each, so all four combinations are
reported with per-rep numbers, CoV, and any crash's error signature, by end of day 2026-07-14. No
shader or dispatch-logic source changes — purely export-reuse + measure + report, following this
workstream's established pattern for this class of feature (`specs/018`, `specs/029`).

## Technical Context

**Language/Version**: N/A for new code — this feature runs the existing release/1.3 export/build
artifacts and the existing `llama_main_rel1.3` runner; it writes no new source.

**Primary Dependencies**: The `release-1.3/` worktree's already-built `llama_main_rel1.3` binary
(plain release/1.3, predates the WMMA coopmat port — no `ET_VK_DISABLE_COOPMAT`/
`ET_VK_EXECUTE_NODE_THRESHOLD` gate to rely on, per workspace-root `CLAUDE.md`'s worktree table);
the NFS run-kit (`/sarc-c/gpusw/users/yanwen.xu/android-run/{models,runners,assets}`) as the source
of already-exported PTEs/tokenizer/prompt; `pin_freqs.sh` for clock pinning.

**Storage**: No new `.pte` files — this feature reuses PTEs already exported and staged on NFS:
`llama3_{2_1b,2_3b,1_8b}_4w_texture_ctx3072.pte` (already on-device) and the `8da4w` siblings
(exist on NFS, not yet pushed to M41 as of plan time). Nothing is exported fresh.

**Testing**: No automated test suite — verification is a coherence check (already done for 4w this
session) plus the tool-driven checks this spec's clarifications added: a devfreq sysfs + in-graph
throughput cross-check before trusting any run as "pinned" (FR-009), and — per gotcha G11 below —
a `dmesg`/`/proc/meminfo` check on any `VK_ERROR_DEVICE_LOST` to distinguish a genuine GPU-watchdog
crash from a host-side OOM kill before attributing a cause.

**Target Platform**: M41 (Exynos s5e9965/ERD9965 family, `xgpusw-debug07`, serial
`000009b44fd4abd3`) — explicitly the constitution's "secondary quick-experiment Samsung device...
use for fast non-target-critical iteration, not as this workstream's validation target" (Reference
Hardware Inventory), consistent with this spec's FR-010/Clarification Q3 framing.

**Project Type**: Measurement/reporting only — no source tree changes.

**Performance Goals**: N/A — this feature measures an existing, unmodified code path's baseline
throughput; it does not target a number of its own, and per FR-010 it is never compared against
Samsung M5 EVT1 headline numbers.

**Constraints**: Must reuse the exact methodology already used for the collected 4w-floating data
(same runner, same 2048-prefill/1024-decode prompt, same 3-rep sampling) so all four tables are
internally comparable (FR-007/FR-008); must not halt the sweep on a crash (FR-006); must verify any
"pinned" label per FR-009 before trusting it; must attribute every `VK_ERROR_DEVICE_LOST` correctly
per gotcha G11 (host OOM vs. genuine watchdog) rather than defaulting to "thermal/watchdog" as this
session did before checking `dmesg`.

**Scale/Scope**: 3 quant-mode × clock-mode cells still to measure (4w-pinned, 8da4w-floating,
8da4w-pinned) × 3 models × 3 reps = 27 timed runs, plus the 9 already-collected 4w-floating runs
folded into the same document = 36 rep-slots total across 4 tables (spec SC-007).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Check | Status |
|---|---|---|
| I. Correctness Before Performance | N/A — no new/modified shader or dispatch code; this is the pre-existing, already-correctness-tested stock ExecuTorch tiled path on a branch that predates coopmat entirely. | PASS (N/A) |
| II. Samsung M5 EVT1 Sole Target | M41 is explicitly *not* the active target — this feature runs there deliberately, framed per FR-010/Clarification Q3 as a secondary/cross-device reference (the same treatment the constitution already gives the retired MiniPC), never compared against or substituted for M5 EVT1 headline numbers. Not a violation: the constitution names M41 by name as a legitimate secondary device for exactly this kind of non-target-critical iteration. | PASS |
| III. Explicit Eligibility Gating, Safe Fallback | N/A — no new gating code; T-tiled is the existing, always-available fallback path itself, and this branch has no coopmat gate to reason about at all. | PASS (N/A) |
| IV. Two-Tier, Statistically Sound Benchmarking | Tier-2 (model-level) only, 3-run sampling with CoV (FR-011) — matches this workstream's own two-tier discipline at the tier that applies (no tier-1 shader microbench exists for a plain tiled baseline). | PASS |
| V. Document Every Driver Workaround | N/A — no driver workaround introduced by this feature itself; if `ET_VK_EXECUTE_NODE_THRESHOLD` turns out to have any effect on `llama_main_rel1.3` (Phase 0 research), that finding is documented in `research.md`, not silently applied. | PASS (N/A) |
| VI. Verify With Tools, Never Assume | No coopmat kernel-dispatch claim is made (there is no coopmat on this branch), so the ETDump-dispatch-confirmation requirement doesn't apply — but this principle's spirit is honored via FR-009's throughput cross-check and gotcha G11's dmesg/meminfo crash-attribution check, both tool-driven, neither assumed. | PASS |
| VII. Clock Discipline | Directly implements this principle's remediation: FR-009 requires sysfs readback AND an in-graph throughput cross-check before any run is labeled "pinned" — exactly the check this principle's own rationale says was missing the first time this failure mode occurred. Floating results are labeled and carry the per-rep-not-blended-mean disclosure FR-007 requires. | PASS |
| VIII. Verify Driver Before Every Coopmat Measurement | This feature measures the tiled path (no coopmat), but FR-001 still requires re-verifying driver identity before measuring, applying the same shared-device discipline regardless of whether coopmat is in play. | PASS |
| IX. Never Disclose Samsung-Internal Specifics Upstream | N/A — this feature produces an internal workspace report only (`specs/030` results), never proposed upstream; device names/serials/driver hashes stay internal per this principle, consistent with how this spec already handles them (never destined for a public PR). | PASS (N/A) |
| X. Consult `instruction-for-ai` Before Acting | This plan's Technical Context and quickstart follow `.shared-context/instruction-for-ai/access-and-run/README.md` §4/§6 for the run/flash mechanics already used successfully this session, and gotcha G11/G12 for crash-attribution and node-threshold nuance rather than re-deriving either from scratch. | PASS |

No violations; Complexity Tracking not needed.

**Post-Phase-1 re-check**: `data-model.md` (the Run/Cell entities) and `quickstart.md` (the
sysfs+throughput pin-verification step, the dmesg/meminfo crash-attribution step) introduce no new
gate risk — both stay within what Phase 0 already justified. Constitution Check still PASSES
across all ten principles.

## Project Structure

### Documentation (this feature)

```text
specs/030-m41-release13-baseline/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md         # Phase 1 output
├── quickstart.md         # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit-tasks, not this command)
```

No `contracts/` — this feature has no external interface (API, CLI, data format) of its own; it
runs the existing runner binary with existing flags and writes a report.

### Source Code (repository root)

```text
/local/yanwen.xu/workspace/dev/executorch/specs/030-m41-release13-baseline/
└── results/
    └── m41-release13-baseline-report.md   # NEW — the 4-table deliverable (spec FR-007/SC-007)
```

No changes anywhere else in the source tree — no new `.pte` (all reused from existing NFS/`.pte_out`
staging), no runner rebuild, no shader/dispatch-logic edits.

**Structure Decision**: A single new `results/` doc under this feature's own spec directory (this
workstream's convention for a feature whose deliverable is a report, not a code change or a
correction to another spec's existing tables — c.f. `specs/029-release-version-4w-baseline`, which
also carries no `plan.md`/`tasks.md` and lives entirely in `spec.md` + `results/`). Unlike
`specs/018` (which folded its numbers into `specs/015`'s existing M5 tables), this feature's M41
numbers have no existing M5-scoped table to fold into — per FR-010 they are explicitly a separate,
secondary-device report, not a row in the M5 EVT1 results.

## Complexity Tracking

*No violations — table not needed.*
