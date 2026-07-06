---

description: "Task list for M5 EVT1 4w Linear Coopmat Retune"
---

# Tasks: M5 EVT1 `4w` Linear Coopmat Retune (fp16 Accumulate, Loop Flattening, Vectorized Dequant)

**Input**: Design documents from `/specs/014-m5-linear-coopmat-retune/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md (all present; no `contracts/`, per plan.md's Project Structure)

**Tests**: Not a separate automated suite — this feature's correctness
signal reuses (and, per FR-008, extends) the existing
`test_coopmat_linear_bench.cpp` coopmat correctness harness
(`research.md` Decision 2, revised), matching how prior specs in this
workstream validated shader changes inline rather than via a new test
phase.

**Organization**: Tasks are grouped by user story. Phase 3 (US1) requires
no device access and was this workstream's first-session deliverable
(already committed). Phase 4 is a new, hardware-independent prerequisite
surfaced by `/speckit-clarify` (FR-008) that both remaining user stories
need. Phases 5-6 (US2/US3) require Samsung M5 EVT1 hardware access,
unavailable as of the last session (`adb devices` returned empty) — they
are recorded as blocked per spec FR-006, not silently skipped.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files/resources, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Paths are relative to the repository root

## Path Conventions

- `backends/vulkan/runtime/graph/ops/glsl/linear_qw_coopmat.glsl` — the three code changes
- `backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_qw_coopmat.glsl`, `backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp` — the documentation-only change
- `backends/vulkan/test/custom_ops/test_coopmat_linear_bench.cpp` — FR-008's correctness-harness extension (new `kCorrectnessShapes`/`kRank3CorrectnessShapes` entries at production K)
- `specs/014-m5-linear-coopmat-retune/results/` — validation logs and disposition summary

---

## Phase 1: Setup

- [X] T001 Create `specs/014-m5-linear-coopmat-retune/results/` directory

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Confirm the working-tree diff is fully and only accounted for by the four described changes before committing anything

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T002 Diff `backends/vulkan/runtime/graph/ops/glsl/linear_qw_coopmat.glsl`, `backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_qw_coopmat.glsl`, and `backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp` against `HEAD` (`01fb136d6`) and confirm every hunk maps to exactly one of: fp16 accumulator, loop-shape flattening, vectorized dequant, or the documentation-only note — confirmed: `linear_qw_coopmat.glsl`'s hunks are the three interleaved shader changes (accumulator type at init/store sites, flattened `last`-conditioned loop replacing separate prologue/epilogue, vectorized `dequant_block`, plus the `bias_sh`/bias-tile type follow-through required by the accumulator change and stale-comment cleanup); `linear_dq8ca_qw_coopmat.glsl` and `QuantizedLinear.cpp` are comment-only
- [X] T003 [P] Confirm the unrelated `backends/cadence/utils/FACTO` submodule dirty state is out of scope for this feature (per spec Assumptions) and left untouched — confirmed not staged, not modified by any task below

**Checkpoint**: Foundation ready — every uncommitted hunk is attributed; nothing unexplained remains

---

## Phase 3: User Story 1 - Preserve and correctly attribute the existing uncommitted work (Priority: P1) 🎯 MVP

**Goal**: Commit the four changes with accurate, honestly-labeled per-change status so the real work is no longer at risk of being lost to an accidental `git stash`/`clean`/branch-switch.

**Independent Test**: `git show` the resulting commit(s) and confirm every hunk is explained by this spec's description of the four changes, with each change's current validation status stated in the commit message or an accompanying results file.

- [X] T004 [US1] Stage and commit `linear_qw_coopmat.glsl`, `linear_dq8ca_qw_coopmat.glsl`, and `QuantizedLinear.cpp` together (interleaved per `research.md` Decision 3), with a commit message naming all four changes and their not-yet-validated status for the three shader changes (depends on T002, T003)
- [X] T005 [US1] Create `specs/014-m5-linear-coopmat-retune/results/disposition-summary.md` seeded from `data-model.md`'s Retuned Shader Change table (all three shader changes `disposition: pending`, documentation change `disposition: keep`) (depends on T001)

**Checkpoint**: US1 complete — nothing from the original uncommitted working-tree state remains at risk; every change is committed and its status is recorded. **DONE** (commits `133044739`, `59a8e62df`).

---

## Phase 4: Correctness Harness Extension (FR-008) — shared prerequisite for User Stories 2 and 3

**Goal**: Close the gap `/speckit-clarify` found: `test_coopmat_linear_bench.cpp`'s existing `kCorrectnessShapes`/`kRank3CorrectnessShapes` only cover up to K=256, short of FR-003/FR-004's production-K (2048/4096+) requirement. Neither US2 nor US3 can honestly claim "passed the correctness check at production shapes" until this closes.

**Independent Test**: Build `test_coopmat_linear_bench.cpp` with the new cases added; the harness compiles and its correctness-only cases (`COOPMAT_BENCH_CORRECTNESS_ONLY=1`) run and report pass/fail for the new K=2048/4096 shapes, independent of whether any of US2/US3's shader changes have been evaluated yet.

**Status**: Not yet started — the source edit itself does not require M5 EVT1 device access (per `quickstart.md` step 2), only a build toolchain; only *running* the new cases needs the device.

- [ ] T006 [P] Author new `kCorrectnessShapes` entries in `backends/vulkan/test/custom_ops/test_coopmat_linear_bench.cpp` at production K (2048 and 4096 at minimum), coopmat-eligible (`M%64==0`, `N%64==0`, `K%32==0`), reusing the existing `make_deterministic_correctness_case` well-conditioned positive-data generation and `abs=0.5`/`rel=0.05` tolerance unchanged (data-model.md's Correctness Harness Extension entity; no device access required to author this)
- [ ] T007 [P] Author a matching new `kRank3CorrectnessShapes` entry at one of T006's production-K shapes (batch=1), same data/tolerance strategy, consistent with the existing rank-3 coverage added for specs `009` (no device access required to author this)
- [ ] T008 Obtain M5 EVT1 device access per `.shared-context/instruction-for-ai/devices-and-access.md`; re-verify driver identity per constitution Principle VIII (depends on: none — can run before or in parallel with T006/T007)
- [ ] T009 Build `test_coopmat_linear_bench` with T006/T007's new cases and run it with `COOPMAT_BENCH_CORRECTNESS_ONLY=1` on M5 EVT1 against the **pre-change** (`HEAD`-only, per `research.md` Decision 1) shader; confirm the new production-K cases compile, dispatch the coopmat kernel, and pass — this validates the harness extension itself, independent of any of this feature's three shader changes (depends on T006, T007, T008)

**Checkpoint**: Phase 4 complete when the harness reports pass/fail for production-K shapes against a known-good (pre-change) shader — only then can US2/US3 below produce a correctness verdict that actually means what FR-003/FR-004 require

---

## Phase 5: User Story 2 - Validate the two low-risk, same-math changes on M5 EVT1 (Priority: P2)

**Goal**: Confirm the loop-shape flattening and vectorized dequant are correctness-safe and measure their performance on the actual target device.

**Independent Test**: Build the post-change shader, pass the extended (Phase 4) INT4 coopmat correctness check at production shapes, and produce a kernel-dispatch-confirmed tier-1 timing compared against a fresh pre-change M5 EVT1 baseline.

**Status: BLOCKED — no M5 EVT1 device reachable as of the last session** (`adb devices` returned an empty list). Recorded per spec FR-006 rather than skipped silently. Additionally depends on Phase 4 completing (FR-008).

- [ ] T010 [US2] Build and run the pre-change (`HEAD`-only) `linear_qw_coopmat.glsl` tier-1 coopmat microbench on M5 EVT1 per `quickstart.md` step 1; record as the baseline in `results/us2-loop-vectorized-dequant-validation.md` (depends on T008)
- [ ] T011 [US2] Run the Phase-4-extended INT4 coopmat correctness check against the post-change shader at production K=2048/4096 (depends on Phase 4 (T009))
- [ ] T012 [US2] Run the tier-1 coopmat microbench against the post-change shader; confirm kernel dispatch and `spirv-dis`-verified `OpCooperativeMatrix*KHR` presence (`research.md` Decision 4); compare against T010's baseline (depends on T010, T011)
- [ ] T013 [US2] Update `results/disposition-summary.md`'s `loop_flattening` and `vectorized_dequant` rows with correctness/perf outcomes and final disposition (depends on T012)

**Checkpoint**: US2 complete when both same-math changes have a recorded disposition — independent of US3's outcome

---

## Phase 6: User Story 3 - Validate the fp16-accumulate experiment's correctness before any perf claim (Priority: P3)

**Goal**: Confirm or refute the fp16-accumulate change's numerical safety at real production K-dimensions before trusting any throughput number for it.

**Independent Test**: Run the fp16-accumulate variant against the Phase-4-extended correctness check at K=2048/4096; pass within the stated `abs=0.5`/`rel=0.05` tolerance, or fail explicitly and revert.

**Status: BLOCKED — no M5 EVT1 device reachable as of the last session**, same as Phase 5. Additionally depends on Phase 4 completing (FR-008).

- [ ] T014 [US3] Run the Phase-4-extended INT4 coopmat correctness check against the fp16-accumulate variant at production K=2048 and K=4096; record numerical divergence against the fp32-accumulate reference within the `abs=0.5`/`rel=0.05` tolerance stated in `data-model.md`'s `numerical_tolerance` field (depends on Phase 4 (T009))
- [ ] T015 [US3] If T014 passes: run the tier-1 coopmat microbench for the fp16-accumulate variant, confirm kernel dispatch + SPIR-V accumulator-type verification (`research.md` Decision 4), compare against T010's baseline (depends on T014, T010)
- [ ] T016 [US3] If T014 fails: revert the fp16-accumulate hunk only (accumulator type at init/store sites) in a new, separate commit, leaving `loop_flattening`/`vectorized_dequant` intact; record the failure shape and divergence magnitude (depends on T014)
- [ ] T017 [US3] Update `results/disposition-summary.md`'s `fp16_accumulate` row with the final outcome and disposition (`keep` if T015 completes, `revert` if T016 executes) (depends on T015 or T016)

**Checkpoint**: US3 complete when `fp16_accumulate` has a recorded disposition — independent of US2's outcome

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T018 Once all of US2/US3 are unblocked and complete, re-read `results/disposition-summary.md` as a whole and confirm every one of the three shader changes has a non-`pending` disposition with a stated reason (spec SC-002, SC-004) (depends on T013, T017)

---

## Dependencies & Execution Order

- **Phase 1 (Setup)** → **Phase 2 (Foundational)**: no dependencies, run first
- **Phase 3 (US1)**: depends on Phase 2; fully deliverable without hardware — **DONE**
- **Phase 4 (FR-008 harness extension)**: T006/T007 (authoring) have no hardware dependency and can start immediately; T008 (device access) can run in parallel; T009 (running the new cases) depends on all three. Phase 4 as a whole does NOT depend on Phase 3 (it doesn't touch the three shader changes), but blocks Phases 5 and 6's correctness steps.
- **Phase 5 (US2)** and **Phase 6 (US3)**: both depend on Phase 3 (the changes must be committed before validating them) and on Phase 4 completing (a correctness verdict against an unextended harness doesn't satisfy FR-003/FR-004) — but are otherwise **independent of each other** per spec Clarifications — neither blocks the other's completion or disposition
- **Phase 7 (Polish)**: depends on both Phase 5 and Phase 6 completing

## Parallel Execution Examples

- T002 and T003 (Phase 2) touch disjoint files and can run in parallel
- T006, T007, and T008 (Phase 4) touch disjoint concerns (test-shape authoring vs. device access) and can run in parallel; T009 waits for all three
- Once Phase 4 is complete, Phase 5 (US2) and Phase 6 (US3) can proceed on independent schedules — a correctness failure in US3 (T014) never blocks US2's own tasks

## Implementation Strategy

**MVP = User Story 1 only** (T001-T005): commits the existing work with
accurate attribution and status. This was achievable with zero device
access and is already done. **Next up = Phase 4** (T006-T009): also
partially achievable without device access (T006/T007) and unblocks both
remaining user stories at once — the highest-leverage next step, since
neither US2 nor US3 can produce a spec-compliant correctness verdict
without it. User Stories 2 and 3 remain independently completable
whenever M5 EVT1 access is next available; neither blocks the other's
disposition.
