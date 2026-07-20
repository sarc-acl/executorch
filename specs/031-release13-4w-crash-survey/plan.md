# Implementation Plan: Release/1.3 Vanilla 4w Crash Survey on M5 EVT1 (Floating Clocks)

**Branch**: `031-release13-4w-crash-survey` | **Date**: 2026-07-14 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/031-release13-4w-crash-survey/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Determine, with repeated evidence (3 reps each) rather than single anecdotes, which of Llama
1B/3B/8B at `4w` crash the M5 EVT1 board vs. complete normally when run on the vanilla
`release/1.3` worktree (no WMMA/coopmat fork additions) with GPU clocks floating. Deliver a single
end-of-day report table: prefill/decode tok/s + CoV for completed reps, explicit crash annotation
for failed ones. No source/shader changes — this is a data-collection feature executed entirely
via existing prebuilt runners (`llama_main_rel1.3`) and already-exported `.pte` files, driven over
`adb`/`fastboot`.

## Technical Context

**Language/Version**: N/A — no new source code; driven entirely by shell/`adb`/`fastboot` against
prebuilt binaries and existing `.pte` files.

**Primary Dependencies**: `adb`, `fastboot`, the vanilla `release-1.3/executorch` worktree's
prebuilt `llama_main_rel1.3` runner (already built and staged, see `specs/029`), the already-
exported `4w` and `8da4w` texture `ctx3072` PTEs for all three models (no new export needed —
all six already existed in `.pte_out/` from prior sessions). **Extension**: also
`release13-node-threshold/executorch`'s `llama_main_nodethresh` runner (rebuilt fresh mid-session
from that worktree's uncommitted `ComputeGraph.cpp` diff — a pure runtime patch adding an opt-in
`ET_VK_EXECUTE_NODE_THRESHOLD` env var, no AOT/export changes), used only on cells where vanilla
was empirically confirmed to crash.

**Storage**: Result artifacts are markdown + the raw per-rep numbers, written under this feature's
`results/` directory — no database, no new persistent runtime state.

**Testing**: Manual/tool-driven device runs; "pass" = the runner's own JSON stats line
(`prefill_token_per_sec`/`decode_token_per_sec`/`generated_tokens`) appears; "crash" = the device
drops off `adb` and re-enumerates as `S5E9975_LK_Bootloader`. No automated test suite — the
runner's coherence check (short low-token prompt) before each model's rep sequence is the only
correctness gate, per the constitution's Metrics Philosophy.

**Target Platform**: Android on M5 EVT1 (Samsung ERD9975 / S5E9975, Xclipse 970), Vulkan backend,
reached via `ssh yanwen.xu@sj1-dmckee-d01` + `adb -s 0000088f8e579c33`.

**Project Type**: N/A — benchmarking/data-collection feature, not a software module or service.

**Performance Goals**: N/A — this feature does not target a speedup; it characterizes existing
crash/normal behavior and reports observed throughput as-is.

**Constraints**: Deliverable by end of day (2026-07-14); must not leave the shared board in an
unrecoverable state (only a plain `fastboot reboot` recovery is in scope — no reflash/wipe without
separate explicit authorization); must not silently continue past an unrecognized driver hash
(Principle VIII) or an unrecovered device (per spec's Edge Cases).

**Scale/Scope**: originally 3 models × 3 reps = up to 9 benchmark attempts, `4w`-only,
floating-only. **Extended** to 3 models × 2 quant schemes × 2 clock policies × 3 reps = 12 cells,
36 target completed attempts on one shared device — actual total 46 attempts (34 completed + 12
crashed) once crash-retries are counted.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Applies? | Status |
|---|---|---|
| I. Correctness Before Performance | Partially — no shader/kernel change, so the correctness-test requirement is N/A; the Metrics Philosophy's "coherence check before benchmarking" still applies | **PASS** — a short low-token coherence check precedes each model's first rep (already practiced this session for 8B/3B; will repeat for 1B before its rep sequence in `tasks.md` Phase 3, T006) |
| II. Samsung M5 EVT1 Is the Only Active Target | Yes | **PASS** — M5 EVT1 (`0000088f8e579c33`) is the only device in scope; "M51" resolved to M5 EVT1 per spec Assumptions |
| III. Explicit Eligibility Gating | No | **N/A** — vanilla `release/1.3` has no coopmat dispatch path at all; nothing to gate |
| IV. Two-Tier, Statistically Sound Benchmarking | Partially | **PASS — deviation resolved by extension**: this is tier-2 (model-level) only; no tier-1 microbenchmark applies since no new kernel is involved. The Default Scope's "both 4w and 8da4w, six configs" was initially narrowed to `4w`-only (see Complexity Tracking history below) but the same-day extension added `8da4w`, so the full requirement is now met (and further exceeded — both clock policies, not just one). The "separate ETDump dispatch-confirmation run" sub-requirement remains **N/A**: it exists to confirm a WMMA/coopmat kernel dispatched, and neither `release-1.3` nor `release13-node-threshold` has such a kernel to confirm. |
| V. Document Every Driver Workaround at the Point of Use | No | **N/A** — no shader workaround being added |
| VI. Verify With Tools, Never Assume | Yes | **PASS** — this feature exists specifically to replace a one-rep anecdote with tool-verified, repeated evidence |
| VII. Clock Discipline | Yes | **PASS** — floating is explicitly requested (permitted per Principle VII's own text), always labeled as floating, never presented as a pinned headline; per-rep numbers are reported (spec FR-006) rather than a single blended mean, honoring the documented cold-start-vs-throttle spread |
| VIII. Verify the Driver Before Every Coopmat Measurement | Yes (generalized to "every measurement" for this feature) | **PASS** — spec FR-004/FR-007 require driver-hash verification before each model's first rep and re-verification after every crash recovery |
| IX. Never Disclose Samsung-Internal Specifics Upstream | N/A | This feature's output stays in this internal workspace (`dev/executorch/specs/`); nothing here is destined for the public `pytorch/executorch` repo |
| X. Consult `instruction-for-ai` Before Acting | Yes | **PASS** — this session already followed `access-and-run/README.md` (push/coherence-check/clock/driver-verify/logcat procedures) and `hardware/README.md` for this device before acting |

No unjustified violations. One deviation (4w-only scope) is recorded in Complexity Tracking below.

## Project Structure

### Documentation (this feature)

```text
specs/031-release13-4w-crash-survey/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── checklists/
│   └── requirements.md  # /speckit-specify quality checklist (already generated)
├── results/             # Phase 2/implementation output — per-rep raw numbers + headline table
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

No `contracts/` directory — this feature exposes no external interface (library API, CLI schema,
service endpoint) for another system to consume; it only produces a results document consumed by
the requesting engineer.

### Source Code (repository root)

No source code changes. This feature is executed entirely against the pre-existing, read-only
`release-1.3/executorch` worktree (see workspace-root `CLAUDE.md`'s worktree table) using its
already-built `llama_main_rel1.3` runner and already-exported `.pte` files — no new files under
`backends/`, `extension/`, or any build target. The only artifacts this feature creates live under
`specs/031-release13-4w-crash-survey/` (this plan, research, data model, quickstart, results) —
there is no `src/`/`tests/` tree to lay out.

**Structure Decision**: Docs-and-results-only structure (no `contracts/`, no source tree) — this
is a data-collection/reporting feature, not a software module. All artifacts live under this
feature's own `specs/031-release13-4w-crash-survey/` directory, consistent with how
`specs/029-release-version-4w-baseline` and `specs/030-m41-release13-baseline` (the two most
similar prior specs — also pure benchmarking studies with no code changes) organized their output.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because | Resolution |
|-----------|------------|-------------------------------------|------------|
| ~~Default Scope narrowed to `4w` only~~ (excludes `8da4w`, normally required alongside it for every model-level benchmark under this constitution) | The user's explicit mid-session refinement asked specifically for a 4w-only, same-day report table across 1B/3B/8B under floating clocks | Running the full 6-config (4w + 8da4w) matrix would roughly double the number of crash-recovery cycles needed that day, risking the end-of-day deadline for no requested benefit at the time | **RESOLVED same day** — the user explicitly asked for `8da4w` in a follow-up extension; the deviation no longer applies (kept here, struck through, for history — see `spec.md`'s Extension section) |

## Post-Design Constitution Re-check

Re-evaluated after Phase 0 (`research.md`) and Phase 1 (`data-model.md`, `quickstart.md`): no new
violations introduced. `research.md`'s methodology decisions (median/CoV convention, run order,
`fastboot reboot`-only recovery, driver/clock checkpoints) all resolve in favor of the same
principles cited in the Constitution Check above rather than against them — in particular, the
run-order decision (3B → 1B → 8B) does not compromise Principle II/VII/VIII since every model
still gets the same driver/clock verification checkpoints regardless of order. Gate: **PASS**.

## Post-Extension Constitution Re-check

Re-evaluated after the same-day extension (4w pinned gap-fill + full 8da4w matrix + threshold
policy change to 64-then-32-fallback). No new violations: Principle VII/VIII's per-cell
driver/clock verification discipline was maintained identically across all 46 attempts (34
completed + 12 crashed); Principle IV's Default Scope deviation (Complexity Tracking row above)
is now resolved rather than merely justified. One new methodology point worth flagging under
Principle VI (Verify With Tools, Never Assume): the extension explicitly avoided assuming a
threshold value that worked on one cell would work on another (e.g. `64` was independently
confirmed, not assumed, for each of 8B-floating-4w, 3B-pinned-4w, 3B-pinned-8da4w, and
8B-floating-8da4w; it was independently confirmed *insufficient* for 8B-pinned on both quant
schemes before falling back to `32`). Gate: **PASS**.
