---

description: "Task list for Re-Opening SUBGROUP_SIZE=32 in the 8da4w CoopMat Tile/Subgroup Sweep on M5 EVT1"

---

# Tasks: Re-Open SUBGROUP_SIZE=32 in the 8da4w CoopMat Tile/Subgroup Sweep on M5 EVT1

**Input**: Design documents from `/specs/026-8da4w-subgroup32-sweep/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/sweep-report-schema.md, quickstart.md

**Tests**: No dedicated unit-test tasks are included — this feature's correctness gate is the existing `COOPMAT_BENCH_CORRECTNESS_ONLY=1` harness for `dq8ca_q4gsw` (Constitution Principle I), reused as-is (now run across a broader shape set, not reimplemented); verification steps are folded into the implementation tasks below.

**Organization**: Tasks are grouped by user story (spec.md) to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- File paths below are relative to this repo (`dev/executorch`) unless prefixed `EXEC-WT/`, which means the **existing** `dbuf-int8-sweep` worktree (`023-8da4w-int8-dbuf-sweep-impl` branch) — reused deliberately per plan.md "Structure Decision" and research.md Decision 5, not a freshly-branched worktree.

## Path Conventions

- Analysis/orchestration scripts and all documentation: `specs/026-8da4w-subgroup32-sweep/` in this repo.
- Shader variant catalog and dispatch code (extended, not new): `EXEC-WT/backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_q4gsw_coopmat_tsweep.{glsl,yaml}` and `EXEC-WT/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp` — both already carry `025`'s uncommitted tsweep work plus this session's ad-hoc `sg32test` entry.
- Production shader (comment-only edit, Task T033): `EXEC-WT/backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_qw_coopmat.{glsl,yaml}`.
- Results: `specs/026-8da4w-subgroup32-sweep/results/`.

---

## Phase 1: Setup

**Purpose**: Create working directories and confirm the reused execution worktree is in the state this feature expects.

- [ ] T001 Create `specs/026-8da4w-subgroup32-sweep/scripts/` and `specs/026-8da4w-subgroup32-sweep/results/` (with a `results/raw/` subdirectory for correctness-harness logs) directories
- [ ] T002 Commit this feature's spec/plan/research/data-model/contracts/quickstart/tasks to `yanwen/dev-1.3`; confirm `EXEC-WT` (`dbuf-int8-sweep` worktree, `023-8da4w-int8-dbuf-sweep-impl` branch) is still checked out on that exact branch (never repoint it — workspace `CLAUDE.md` rule) and still has this session's uncommitted `linear_dq8ca_q4gsw_coopmat_tsweep.{glsl,yaml}` / `QuantizedLinear.cpp` / ad-hoc `sg32test` changes intact

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared infrastructure every user story needs. Must complete before any user story phase begins.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T003 [P] Extend `specs/025-8da4w-parameter-sweep/scripts/tile_constraints.py`'s logic into a new `specs/026-8da4w-subgroup32-sweep/scripts/tile_constraints.py`: same `wg_size`/`lds_bytes`/`accumulators_per_sg` formulas, but the `valid` boolean's `subgroup_size == 64` hard filter is replaced with `subgroup_size ∈ {32, 64}` (data-model.md `ConfigurationCandidate` Validation rules) — legality for `subgroup_size == 32` is now determined by real `compile_status` evidence from User Story 1, not rejected at enumeration time. Token generation extends to emit `s32` or `s64` correctly (`tsweep_t<M>x<N>k<K>g<SGX><SGY>s<sub>`). This module is shared by `enumerate_configs.py` (US2 prep) and `staged_search.py` (US3).
- [ ] T004 [P] Verify `EXEC-WT`'s Android build is current: `cmake-out-android-vk` has been installed (`--target install`, per research.md Decision 4 — this session already did this once; confirm it's not stale) and `cmake-out-android-vk/bench/test_coopmat_linear_bench` builds successfully against it; rebuild per quickstart.md Prerequisites if stale
- [ ] T005 [P] Confirm M5 EVT1 device access, driver identity, and clock pin on the board to be used first (`xgpusw-debug08`/`00000bf70c579c33`, this session's already-verified secondary board, or the primary `sj1-dmckee-d01`/`0000088f8e579c33`) per quickstart.md Prerequisites — record which board in every subsequent result (data-model.md `MeasurementResult.board`)

**Checkpoint**: Foundation ready — User Story 1 can begin immediately; User Stories 2/3 depend on User Story 1's legality findings.

---

## Phase 3: User Story 1 - Re-derive the legal search space with subgroup_size as a real axis (Priority: P1) 🎯 MVP

**Goal**: Attempt real on-device compilation/pipeline creation for subgroup=32 candidates across a representative spread of tile shapes (not the single shape this session's `sg32test` probe and `025`'s T014 each tested), and determine whether the documented Xclipse PAL compiler crash still applies, narrowly or broadly.

**Independent Test**: Compile and attempt pipeline creation for ≥5 tile-shape variants at `subgroup_size=32`; confirm the outcome (crash / no crash) is recorded per candidate/shape, not generalized from one data point.

### Implementation for User Story 1

- [ ] T006 [US1] Re-run the pre-flight driver/clock check (T005's procedure, freshly — Principle VIII requires re-verification before every coopmat measurement round, not reuse of a prior check)
- [ ] T007 [US1] Select ≥5 tile shapes spanning small/medium/large from `025`'s already-enumerated `subgroup_size=64` space (`specs/025-8da4w-parameter-sweep/results/configs.json`), including the shipped `128×64/K32/2×2` shape (already covered by this session's `sg32test` entry in `EXEC-WT/backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_q4gsw_coopmat_tsweep.yaml`) and `025`'s standing winner shape `128×32/K16/1×2`
- [ ] T008 [US1] For each shape selected in T007, add a `SUBGROUP_SIZE: 32` `shader_variants` entry to `EXEC-WT/backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_q4gsw_coopmat_tsweep.yaml` (reusing this session's `sg32test` entry for the shipped shape rather than duplicating it) and register each in `EXEC-WT/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp`'s `dq8ca_coopmat_variant()` — this feature's proper `tsweep_t...s32` tokens, superseding the session's ad-hoc literal `"sg32test"` allow-list entry (spec FR-012)
- [ ] T009 [US1] Rebuild `EXEC-WT`'s `vulkan_backend` (`--target install`) and `test_coopmat_linear_bench` (depends on T008); record any shape whose shader fails to compile or crashes `vkCreateComputePipelines` as `compile_status: compile_failed` with the specific error — this is a valid, reportable outcome, not a task failure (spec Edge Cases)
- [ ] T010 [US1] Run each surviving candidate from T009 via adb with `COOPMAT_BENCH_CORRECTNESS_ONLY=1` at the single small shape (`M=K=N=128`) as an initial pipeline-creation smoke check only (not the full correctness gate — that is User Story 2); write `specs/026-8da4w-subgroup32-sweep/results/subgroup32_legality.json` per contracts/sweep-report-schema.md §0, one entry per attempted shape with `compile_status`, `pipeline_creation_crashed`, `driver_hash`, `board`
- [ ] T011 [US1] Write the top-level summary line for `subgroup32_legality.json` stating explicitly, per Constitution Principle V, whether the historical crash reproduced at any shape, none, or all — this is the input to the Task T033 shader-comment update

**Checkpoint**: User Story 1 complete — legality of `subgroup_size=32` is now evidence-based across a spread of shapes, not assumed from a stale comment or generalized from one probe.

---

## Phase 4: User Story 2 - Correctness-gate subgroup=32 candidates across the full representative shape set (Priority: P1)

**Goal**: Re-derive the full legal space (both subgroup sizes) using User Story 1's compile-legality evidence, analytically shortlist it, and correctness-check every shortlisted candidate against the complete multi-shape representative set — not the single shape prior probes used.

**Independent Test**: Run `enumerate_configs.py` → `score_and_shortlist.py` → the correctness-matrix stage end-to-end and confirm `correctness_matrix.json` reports an identical shape-key set for every candidate, and that at least one candidate is flagged `all_shapes_pass: false` with named `failing_shapes` if the `M=256` regression this session found reproduces.

### Implementation for User Story 2

- [ ] T012 [US2] Implement `specs/026-8da4w-subgroup32-sweep/scripts/enumerate_configs.py`: iterate the same tile/grid parameter ranges `025` explored, times `subgroup_size ∈ {32, 64}` (research.md Decision 1); use `tile_constraints.py` (T003) to filter to `valid=true`; cross-reference `subgroup32_legality.json` (T010) so a `subgroup_size=32` candidate at a shape T009 found `compile_status: compile_failed` is excluded with that reason, not silently included; write `specs/026-8da4w-subgroup32-sweep/results/configs.json` per contracts/sweep-report-schema.md §1
- [ ] T013 [US2] Run `enumerate_configs.py` and verify: entries exist at both `subgroup_size: 32` and `subgroup_size: 64`; `025`'s winning token (`tsweep_t128x32k16g12s64`) and its `s32` counterpart are both present (the latter marked per its T010 compile outcome)
- [ ] T014 [US2] Implement `specs/026-8da4w-subgroup32-sweep/scripts/score_and_shortlist.py`: reuse `025`'s occupancy/register-pressure formula and calibration (both subgroup sizes share the same `lds_bytes`/`accumulators_per_sg` model — only `wg_size` differs by `subgroup_size`, per data-model.md); rank all candidates together; mark the top-ranked subset `shortlisted: true` up to `budget.json`'s `budget_cap`; force `shortlisted: true` for `025`'s winning token (`shortlist_reason: "anchor:025-winner"`) plus `025`'s own anchors (shipped-config, `4w`-winner) for continuity; write `specs/026-8da4w-subgroup32-sweep/results/shortlist.json` per contracts/sweep-report-schema.md §2
- [ ] T015 [US2] Run `score_and_shortlist.py` and verify: every `configs.json` candidate appears exactly once; `025`'s winning token is present with `shortlist_reason: "anchor:025-winner"`
- [ ] T016 [US2] Extend `EXEC-WT`'s `linear_dq8ca_q4gsw_coopmat_tsweep.yaml`/`QuantizedLinear.cpp` (building on T008's work) with one `shader_variants`/dispatch-token entry per candidate marked `shortlisted: true` in `shortlist.json` that isn't already present from User Story 1; rebuild (depends on T016)
- [ ] T017 [US2] For every shortlisted candidate with a compiling shader (from T009/T016), run the existing `test_coopmat_linear_bench` harness's full `COOPMAT_BENCH_CORRECTNESS_ONLY=1` multi-shape matrix via adb (not a single shape) — must include the `M=256,K=256,N=256` and `M=256,K=128,N={128,64}` shapes this session found failing at `subgroup_size=32` for the shipped tile shape, so a regression at those specific shapes cannot be missed for other candidates either
- [ ] T018 [US2] Parse T017's raw output into `specs/026-8da4w-subgroup32-sweep/results/correctness_matrix.json` per contracts/sweep-report-schema.md §3: one `CorrectnessResult` per candidate, `per_shape_results` map with an identical key set across all candidates, `all_shapes_pass` and `failing_shapes` derived per data-model.md's rule
- [ ] T019 [US2] Verify `correctness_matrix.json`: confirm no candidate is missing any shape key present in another candidate's entry (identical shape-key set contract); confirm any candidate reproducing this session's `M=256` failure pattern is correctly marked `all_shapes_pass: false` with those exact shapes named in `failing_shapes`
- [ ] T020 [US2] Retire this session's ad-hoc `"sg32test"` literal allow-list entry from `EXEC-WT/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp`'s `dq8ca_coopmat_variant()` now that its one covered shape/tile combination is subsumed by this feature's proper `tsweep_t...s32` tokens (T008/T016) and correctness-matrix coverage (T017-T019) — confirm the file still builds after removal (spec FR-012)

**Checkpoint**: User Story 2 complete — every shortlisted candidate has a per-shape correctness verdict; only `all_shapes_pass: true` candidates are eligible to proceed to User Story 3's performance search.

---

## Phase 5: User Story 3 - Search for and validate a subgroup=32 (or mixed) winner against the subgroup=64 baseline (Priority: P2)

**Goal**: Run a staged, budget-capped performance search over the correctness-surviving candidates from User Story 2, converge on one overall winner (new, or `025`'s standing winner), and report a definitive comparison.

**Independent Test**: Run the staged search over the `all_shapes_pass: true` subset of `shortlist.json` and confirm every survivor gets at least one measurement, weak candidates are eliminated early, top contenders get full statistical rigor, and the final report states a clear winner or an explicit "no improvement over `025`'s winner" outcome.

### Implementation for User Story 3

- [ ] T021 [US3] Implement `specs/026-8da4w-subgroup32-sweep/scripts/staged_search.py` Round 1 (`round1_gate`): for each candidate in `shortlist.json` with `shortlisted: true` AND a corresponding `correctness_matrix.json` entry with `all_shapes_pass: true` (the script MUST refuse to emit a `MeasurementResult` for any other candidate — contracts/sweep-report-schema.md §4), run via adb against the T016-rebuilt binary; re-verify driver hash/device availability/clock pin fresh before the round starts, aborting with a `halted: true` sentinel if it fails
- [ ] T022 [US3] Run Round 1 across all correctness-surviving shortlisted candidates; write `specs/026-8da4w-subgroup32-sweep/results/round1_results.json` and update `budget.json` (depends on T020, T021)
- [ ] T023 [US3] Implement `staged_search.py` Round 2 (`round2_full_shapes`): for the top-performing subset of Round 1 survivors, run at the harness's default rigor across the 6 representative shapes (`wq`+`w1_gate` × {1B,3B,8B}); re-run the pre-round driver/clock check first
- [ ] T024 [US3] Run Round 2; write `round2_results.json` and update `budget.json` (depends on T022, T023)
- [ ] T025 [US3] Implement `staged_search.py` Round 3 (`round3_confirmation`): for the top Round 2 survivors (which may include both `subgroup_size=32` and `subgroup_size=64` finalists), repeat the measurement across exactly 3 independent process invocations to compute `mean_gflops`/`stddev_gflops`/`run_count` with `cov < 0.05`; apply `025`'s documented tie-breaking rule if finalists are statistically indistinguishable; re-run the pre-round driver/clock check first
- [ ] T026 [US3] Run Round 3; write `round3_results.json` and update `budget.json` (depends on T024, T025)
- [ ] T027 [US3] Verify across all three rounds' `budget.json` snapshots that `configs_measured_on_hardware` never exceeded `budget_cap` at any checkpoint (spec FR-009/SC-005 enforcement)
- [ ] T028 [US3] Re-run `COOPMAT_BENCH_CORRECTNESS_ONLY=1` for the Round 3 winner across the FULL representative shape set one final time (not just the smoke-check shape) to confirm PASS immediately before reporting it as the winner — if it fails at any shape, drop this candidate, fall back to the next Round 3 finalist, and re-run this task
- [ ] T029 [US3] Inspect the winner's compiled SPIR-V (`spirv-dis` or equivalent) and confirm genuine int8 cooperative-matrix instructions (`OpCooperativeMatrixMulAddKHR`/coopmat-family) are present (Constitution Principle VI) — if the winner uses `subgroup_size=32`, additionally confirm the SPIR-V's declared subgroup size matches (not silently defaulted to 64 by the driver)
- [ ] T030 [US3] Implement the `staged_search.py --report-only` report generator: read `round3_results.json`, `correctness_matrix.json`, `subgroup32_legality.json`, `budget.json`, and `shortlist.json`, and produce `specs/026-8da4w-subgroup32-sweep/results/sweep-report.md` per contracts/sweep-report-schema.md §6 — `axis_disposition` stated up front, the correctness matrix, the speedup-vs-`025`-winner table, the carried-forward shipped-baseline/`4w`-winner comparisons, an explicit statement of the winner's `subgroup_size_used` (spec FR-007), and a "probe disposition" section stating whether T020's removal of the ad-hoc `sg32test` binding is the final state or whether it was instead retained with a reason (spec FR-012/SC-007 — this section MUST be populated even though T033's shader-comment diff doesn't exist until Phase 6; leave a literal placeholder line (e.g. "shader-comment diff: pending T033") in this initial write so T033 has an anchor to update rather than needing to guess where to insert its reference)
- [ ] T031 [US3] Implement the FR-008 "axis confirmed closed" branch in the report generator: if no `subgroup_size=32` candidate beats `025`'s standing winner on both correctness (all shapes) and performance, state this explicitly with `axis_disposition: "subgroup32_legal_but_no_improvement"` (or `"subgroup32_illegal_confirmed"` if User Story 1 found the crash reproduces everywhere attempted) and `recommendation: "keep_025_winner"`, rather than an inconclusive result
- [ ] T032 [US3] Add the pruning-audit appendix to `sweep-report.md` (or a direct link to `shortlist.json`/`correctness_matrix.json`) so any candidate's fate — shortlisted, anchor, excluded, compile-failed, correctness-failed (naming shapes), or eliminated — can be traced without re-running the search (spec SC-006)

**Checkpoint**: All three user stories complete. `sweep-report.md` is the decision-ready artifact answering the feature's original question — with proper sweep evidence this time, not a one-shot probe.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation, the Principle V documentation deliverable, and integration with this workstream's existing conventions.

- [ ] T033 Draft the proposed diff to `EXEC-WT/backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_qw_coopmat.glsl`/`.yaml`'s header comment (research.md Decision 6): replace the blanket "the Xclipse PAL compiler crashes ... at forced subgroup size 32" claim with the actual, shape-broad finding from T011/T019 (e.g. "does not crash on driver `f14c51b6f8`+ at any of N tested shapes, but subgroup=32 candidates are shape-dependently incorrect and/or slower than subgroup=64 — see `specs/026`"); save the diff to `specs/026-8da4w-subgroup32-sweep/results/shader-comment-update.diff` — this diff is a documentation record only (Constitution Principle V) and is never applied to the production shader by this feature itself, regardless of `sweep-report.md`'s `recommendation` value; whether/when to apply it is a separate follow-on decision (spec Assumptions), matching `025`'s own deferred-shipping precedent. Update `sweep-report.md`'s "probe disposition" section (written by T030) to replace its placeholder line with a real reference to this diff's path.
- [ ] T034 [P] Run `quickstart.md` end-to-end from a clean state and confirm every "Expected outcome" in it holds
- [ ] T035 [P] Add a one-line pointer from `specs/026-8da4w-subgroup32-sweep/checklists/requirements.md` Notes to the final `results/sweep-report.md` location
- [ ] T036 Confirm `EXEC-WT`'s uncommitted shader/dispatch changes (this feature's `tsweep_...s32` entries, minus the now-removed `sg32test` literal from T020) are either committed to `023-8da4w-int8-dbuf-sweep-impl` and pushed, or explicitly left as uncommitted experimental work by design (matching how `025`'s own `tsweep_...s64` entries already exist) — do not leave the state ambiguous given this worktree is reused across two features now

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately.
- **Foundational (Phase 2)**: Depends on Setup. BLOCKS all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational (specifically T004, T005 — this story is on-device from its first task).
- **User Story 2 (Phase 4)**: Depends on Foundational (T003) AND on User Story 1's output (`subgroup32_legality.json`, T010-T011).
- **User Story 3 (Phase 5)**: Depends on User Story 2's output (`correctness_matrix.json`, T018-T019).
- **Polish (Phase 6)**: Depends on all three user stories being complete.

### User Story Dependencies

- **User Story 1 (P1, MVP)**: Independently testable once Foundational T004/T005 are done. This is the true MVP — it stands alone as "here is the actual, shape-broad legality evidence for subgroup=32," even before any correctness-matrix or performance work exists — directly answering the narrower question `025`'s T014 and this session's own probe each left open.
- **User Story 2 (P1)**: Requires US1's `subgroup32_legality.json` as input (which candidates are even attemptable) — sequential by design.
- **User Story 3 (P2)**: Requires US2's `correctness_matrix.json` as input (which candidates are eligible for performance measurement) — likewise sequential.

### Parallel Opportunities

- T004 and T005 (Foundational) are independent of each other and can run in parallel; T003 is independent of both.
- T034 and T035 (Polish) are independent and can run in parallel.
- Within User Story 3, T021→T022→...→T026 are strictly sequential (each round depends on the prior), but T030 (report generator implementation) can be written in parallel with T021-T026 (different files) as long as it's not *run* until T026 completes.

---

## Parallel Example: Foundational Phase

```bash
# Launch all three foundational checks together:
Task: "Extend tile_constraints.py per T003"
Task: "Confirm EXEC-WT execution worktree build is current per T004"
Task: "Confirm M5 EVT1 device/driver/clock state per T005"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: `subgroup32_legality.json` exists and its summary states, with per-shape evidence across ≥5 tile shapes, whether the historical crash reproduces — this alone is a useful, reviewable artifact (settling the "is 32 even legal" question properly) even before any correctness-matrix or performance work begins.

### Incremental Delivery

1. Setup + Foundational → extended tile-constraint model and device/build readiness in place.
2. User Story 1 → subgroup=32 legality established with real, shape-broad on-device evidence (MVP: "here's whether 32 actually crashes, and where").
3. User Story 2 → full legal space re-derived with both subgroup sizes, shortlisted, and correctness-gated across the complete representative shape set — the step that would have caught this session's `M=256` regression if it had existed before.
4. User Story 3 → staged on-device search narrows correctness-surviving candidates to a confirmed top performer, validated and reported against `025`'s standing winner, closing the loop on whether the axis should stay open or closed.

### Notes

- This feature's three user stories are a **pipeline**, not independent parallel workstreams — US2 needs US1's legality output and US3 needs US2's correctness-matrix output. Sequencing them in priority order is the actual data dependency, not just a suggestion.
- Every task touching the execution worktree (T004, T006-T010, T016-T029) operates in the **reused** `EXEC-WT` worktree (`dbuf-int8-sweep`, `023-8da4w-int8-dbuf-sweep-impl` branch), not this `dev/executorch` worktree and not a freshly-branched one — see plan.md "Structure Decision" and research.md Decision 5, a deliberate deviation from `025`'s own precedent of branching fresh.
- T020's removal of the ad-hoc `sg32test` literal must happen only after T017-T019 confirm the broader correctness matrix actually covers that entry's one shape/tile combination — do not remove it as a bare cleanup step disconnected from that confirmation, or FR-012's "supersede, don't just delete" requirement is not actually met.
- Commit spec-kit documentation and script changes in this repo (`dev/executorch`, on a feature branch PR'd into `yanwen/dev-1.3` per workspace convention) per this workstream's existing small-commit convention; T036 explicitly resolves what happens to the execution worktree's own uncommitted state, since it is now shared history across two features (`025`'s original `tsweep_...s64` work and this feature's `...s32` extension) rather than a single feature's disposable scratch space.
