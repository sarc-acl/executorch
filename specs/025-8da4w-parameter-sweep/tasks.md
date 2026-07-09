---

description: "Task list for 8da4w (dq8ca/q4gsw) CoopMat Tile/Subgroup Parameter Sweep on M5 EVT1"

---

# Tasks: 8da4w (dq8ca/q4gsw) CoopMat Tile/Subgroup Parameter Sweep on M5 EVT1

**Input**: Design documents from `/specs/025-8da4w-parameter-sweep/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/sweep-report-schema.md, quickstart.md

**Tests**: No dedicated unit-test tasks are included — this feature's correctness gate is the existing `COOPMAT_BENCH_CORRECTNESS_ONLY=1` harness for `dq8ca_q4gsw` (Constitution Principle I), reused as-is rather than reimplemented; verification steps are folded into the implementation tasks below.

**Organization**: Tasks are grouped by user story (spec.md) to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- File paths below are relative to this repo (`dev/executorch`) unless prefixed `EXEC-WT/`, which means the dedicated execution worktree created per plan.md "Structure Decision" (research.md Decision 4) — never the existing `dev/` worktree folder itself.

## Path Conventions

- Analysis/orchestration scripts and all documentation: `specs/025-8da4w-parameter-sweep/` in this repo.
- Shader variant catalog and dispatch code: `EXEC-WT/backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_q4gsw_coopmat_tsweep.yaml` and `EXEC-WT/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp`.
- Existing `dbuf1-4` shader family (from `specs/023`): `EXEC-WT/backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_q4gsw_coopmat_dbuf{1..4}.{glsl,yaml}`.
- Results: `specs/025-8da4w-parameter-sweep/results/`.

---

## Phase 1: Setup

**Purpose**: Create the working directories and the dedicated execution worktree this feature's code changes and on-device measurement run in.

- [X] T001 Create `specs/025-8da4w-parameter-sweep/scripts/` and `specs/025-8da4w-parameter-sweep/results/` directories
- [X] T002 Commit this feature's spec/plan/research/data-model/contracts/quickstart/tasks to `yanwen/dev-1.3`, then create a new dedicated git worktree branched from that commit (research.md Decision 4) — reuse `specs/023-8da4w-int8-dbuf-sweep`'s execution worktree if it is still present and has a warm Android build; otherwise create a fresh one and bootstrap it per `.shared-context/instruction-for-ai/` (`./install_executorch.sh --minimal`)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared infrastructure every user story needs. Must complete before any user story phase begins.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T003 [P] Implement the shared tile-constraint validation module in `specs/025-8da4w-parameter-sweep/scripts/tile_constraints.py`: given `(wg_tile_m, wg_tile_n, wg_tile_k, sg_grid_x, sg_grid_y, subgroup_size)`, compute `wg_size`, `lds_bytes` (using `8da4w`'s own shared-memory layout — `Ash_int8`/`Bsh_int8` plus the `izp_sh`/`ifs_sh`/`wsum_sh`/`wsc_sh`/`bias_sh` broadcast arrays, per research.md Decision 1/2, NOT `4w`'s formula), `accumulators_per_sg` (this shader carries both an `int32` and an `fp32` accumulator array), and a `valid` boolean (`wg_size <= 1024`, MMA-alignment, `wg_tile_k` divides the INT4 group size, `lds_bytes <= 65536`, and `subgroup_size == 64` — any `subgroup_size == 32` input is rejected as an invalid, known-driver-crash configuration, never marked `valid`). Also generate the canonical `tsweep_t<M>x<N>k<K>g<SGX><SGY>s64` token string. This module is shared by `enumerate_configs.py` (US2) and `staged_search.py` (US3).
- [X] T004 [P] Verify `EXEC-WT` has a current Android build (`cmake-out-android-vk/lib/libvulkan_backend.a` and `cmake-out-android-vk/bench/test_coopmat_linear_bench` both present and newer than the worktree's source files, including the existing `dbuf1-4` shader family from `specs/023`); rebuild per `.shared-context/instruction-for-ai/` build docs if stale
- [X] T005 [P] Confirm M5 EVT1 device access, driver identity, and clock pin per quickstart.md Step 3 (driver hash matches `.shared-context/ACTIVE-STATUS.md`'s known-good value, no `llama`/`coopmat` process running, `pin_freqs.sh` reports the expected pinned clocks)

**Checkpoint**: Foundation ready — User Story 1 can begin immediately; User Stories 2/3 depend on User Story 1's loop-structure winner.

---

## Phase 3: User Story 1 - Re-confirm the dbuf2 loop-structure starting point (Priority: P1) 🎯 MVP

**Goal**: Re-measure all four `dbuf1-4` loop-structure variants at the currently-shipped `8da4w` tile/subgroup geometry on M5 EVT1, and determine which one is actually fastest before holding it fixed for the rest of this sweep.

**Independent Test**: Run all four `dbuf1-4` variants at the shipped geometry and confirm each produces a correctness-verified, coopmat-dispatch-confirmed timing number or an explicit failure reason; confirm the fastest one is recorded with an explicit statement of agreement/disagreement against the user-reported `dbuf2` claim.

### Implementation for User Story 1

- [X] T006 [US1] Re-run the pre-flight driver/clock check (T005's procedure, freshly — Principle VIII requires re-verification before every coopmat measurement round, not reuse of a prior check)
- [X] T007 [US1] For each of `EXEC-WT`'s existing `linear_dq8ca_q4gsw_coopmat_dbuf{1,2,3,4}` variants (from `specs/023`, unmodified), run `COOPMAT_BENCH_CORRECTNESS_ONLY=1` via `ET_VK_DQ8CA_COOPMAT_VARIANT=dbuf<N>` against the shipped 128×64/K32/2×2/s64 geometry and confirm PASS; for any that fails or crashes, record `failure_reason` and continue to the next variant (spec Edge Cases)
- [X] T008 [US1] For each `dbuf{1..4}` variant that passed T007, run the harness 3 times via adb at the 6 representative shapes (`wq`+`w1_gate` × {1B,3B,8B}), confirm dispatch via kernel-name capture (Principle VI — must show the coopmat kernel, not a tiled fallback), and compute `mean_us`/`cov`; write `specs/025-8da4w-parameter-sweep/results/dbuf_reconfirmation.json` per contracts/sweep-report-schema.md §0
- [X] T009 [US1] Determine the fastest variant (`argmin(mean_us)` among `correctness_status: pass` entries) and record, in `dbuf_reconfirmation.json`'s accompanying summary, an explicit statement of whether it matches the user-reported `dbuf2` claim (spec SC-001) — if it disagrees, this is the loop structure held fixed regardless, per spec Edge Cases

**Checkpoint**: User Story 1 complete — the loop structure for User Stories 2/3 is confirmed by fresh on-device measurement, not assumed from the prior claim.

---

## Phase 4: User Story 2 - Prune the tile/subgroup search space with zero device time (Priority: P1)

**Goal**: Re-derive the legal `8da4w` tile-shape × subgroup-grid × subgroup-size space (at the User Story 1 loop structure) and rank/shortlist it using only analytical, hardware-derived signals, with zero on-device measurement.

**Independent Test**: Run `enumerate_configs.py` then `score_and_shortlist.py` end-to-end and confirm `shortlist.json` has a materially-smaller-than-full shortlist marked `shortlisted: true` (including the shipped-config and, if legal, the `4w`-winner anchors), with zero adb/device interaction having occurred.

### Implementation for User Story 2

- [X] T010 [US2] Implement `specs/025-8da4w-parameter-sweep/scripts/enumerate_configs.py`: iterate the same tile/grid parameter ranges `022` explored (`wg_tile_m/n`, `wg_tile_k`, `sg_grid_x/y`), but with `subgroup_size` fixed at `64` only (research.md Decision 1 — no `32` candidates generated at all, not filtered post-hoc); use `tile_constraints.py` (T003) to filter to `valid=true` only; write `specs/025-8da4w-parameter-sweep/results/configs.json` per contracts/sweep-report-schema.md §1
- [X] T011 [US2] Run `enumerate_configs.py` and verify: every entry has `subgroup_size: 64`; the currently-shipped `tsweep_t128x64k32g22s64` config is present; no `...s32` token appears anywhere in the file
- [X] T012 [US2] Implement `specs/025-8da4w-parameter-sweep/scripts/score_and_shortlist.py`: for each candidate in `configs.json`, compute `occupancy_proxy`/`register_penalty`/`score` using the `8da4w`-specific `lds_bytes`/`accumulators_per_sg` from T003 and the `K`/`weight` recalibrated from `dbuf_reconfirmation.json`'s shipped-geometry measurement (research.md Decision 2 — not `022`'s `4w`-calibrated constants); rank all candidates by score descending; mark the top-ranked subset `shortlisted: true` up to `budget.json`'s `budget_cap` (`min(round(0.15*N), 30)`); force `shortlisted: true` for the shipped-config anchor regardless of rank, and for the `4w` 128×64/K16/2×2/s32 anchor only if it is a legal `8da4w` candidate (else record it in a top-level `excluded_anchors` array with a reason); write `specs/025-8da4w-parameter-sweep/results/shortlist.json` per contracts/sweep-report-schema.md §2
- [X] T013 [US2] Run `score_and_shortlist.py` and verify: `shortlist.json` has one entry per `configs.json` candidate (full ranking, per spec FR-009); the shipped-config anchor is present with `shortlist_reason: "anchor:shipped-config"`; the `4w`-winner anchor is either present with `shortlist_reason: "anchor:4w-winner"` or listed in `excluded_anchors` with a reason
- [X] T014 [US2] [P] As a bounded, non-search-budget-counted re-verification (research.md Decision 1, Alternatives), attempt to build and run a single `subgroup_size=32` variant of the tile/subgroup template at the shipped tile shape, to confirm the documented Xclipse PAL compile-crash workaround is still necessary on the current driver; record the outcome (crash reproduced, or newly-compiles) in `specs/025-8da4w-parameter-sweep/results/subgroup32-reverification.md` — if it newly compiles, flag this as a finding for a follow-up feature, do not fold `subgroup_size=32` candidates into this feature's search scope

**Checkpoint**: User Story 2 complete — shortlist produced and budget-capped, zero device time consumed beyond User Story 1's own measurements.

---

## Phase 5: User Story 3 - Find and validate the best-performing configuration (Priority: P2)

**Goal**: Measure the shortlisted `8da4w` tile/subgroup candidates on M5 EVT1 using a staged approach, converge on one validated winner, and report it against the shipped baseline and the `4w` winner.

**Independent Test**: Run the staged search over `shortlist.json` and confirm every shortlisted candidate receives at least one cheap measurement, only top performers proceed to full/confirmation rounds, `budget.json`'s `configs_measured_on_hardware` never exceeds `budget_cap`, and the final report names a validated winner (or explicitly states the shipped configuration stands).

### Implementation for User Story 3

- [X] T015 [US3] Extend `EXEC-WT/backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_q4gsw_coopmat_tsweep.yaml` (new file, built on the User Story 1 winning loop structure) with one `shader_variants` entry per candidate marked `shortlisted: true` in `shortlist.json`, following `022`'s `tsweep` entry format
- [X] T016 [US3] Extend `EXEC-WT/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp` with a new `dq8ca_coopmat_variant_tile()` + `kTokens[]` table, one token branch per shortlisted candidate, additive to (not replacing) `specs/023`'s existing `ET_VK_DQ8CA_COOPMAT_VARIANT` dbuf-selection env var (research.md Decision 3); default (both env vars unset) dispatch behavior unchanged (depends on T015)
- [X] T017 [US3] Rebuild `vulkan_backend` and `test_coopmat_linear_bench` in `EXEC-WT` (depends on T016); for any candidate whose shader fails to compile, mark `compile_status: compile_failed` in `shortlist.json`, remove its yaml/cpp entries, and rebuild again
- [X] T018 [US3] Implement `specs/025-8da4w-parameter-sweep/scripts/staged_search.py` Round 1 (`round1_gate`): for each shortlisted candidate with `compile_status: compiles`, run via adb against the rebuilt binary; before the round starts, re-verify driver hash/device availability/clock pin (fresh check, not reused from Foundational or User Story 1) and abort the round (writing a `halted: true` sentinel per contracts/sweep-report-schema.md §3) if it fails
- [X] T019 [US3] Run Round 1 across all compiling shortlisted candidates; write `specs/025-8da4w-parameter-sweep/results/round1_results.json` and update `budget.json` (depends on T017, T018); candidates failing to compile or failing correctness are marked `eliminated_at: true` and excluded from Round 2
- [X] T020 [US3] Implement `staged_search.py` Round 2 (`round2_full_shapes`): for the top-performing subset of Round 1 survivors, run at the harness's default rigor across the 6 representative shapes; re-run the pre-round driver/clock check first
- [X] T021 [US3] Run Round 2; write `round2_results.json` and update `budget.json` (depends on T019, T020)
- [X] T022 [US3] Implement `staged_search.py` Round 3 (`round3_confirmation`): for the top Round 2 survivors, repeat the measurement across exactly 3 independent process invocations to compute `mean_gflops`/`stddev_gflops`/`run_count` with `cov < 0.05` (spec Clarified 2026-07-09, Constitution Principle IV); apply a documented tie-breaking rule (prefer smaller `lds_bytes`, then smaller `accumulators_per_sg`, matching `022`'s precedent) if finalists are statistically indistinguishable; re-run the pre-round driver/clock check first
- [X] T023 [US3] Run Round 3; write `round3_results.json` and update `budget.json` (depends on T021, T022)
- [X] T024 [US3] Verify across all three rounds' `budget.json` snapshots that `configs_measured_on_hardware` never exceeded `budget_cap` at any checkpoint (spec FR-007/SC-002 enforcement)
- [X] T025 [US3] Run `COOPMAT_BENCH_CORRECTNESS_ONLY=1` for the Round 3 winner at the standard multi-tile validation shape via adb; confirm PASS (Constitution Principle I) — if it fails, drop this candidate, fall back to the next Round 3 finalist, and re-run this task
- [X] T026 [US3] Inspect the winner's compiled SPIR-V (`spirv-dis` or equivalent) and confirm the expected int8 cooperative-matrix instructions (`OpCooperativeMatrixMulAddKHR`/coopmat-family) are present (Constitution Principle VI)
- [X] T027 [US3] Implement the `staged_search.py --report-only` report generator: read `round3_results.json`, `dbuf_reconfirmation.json`, `budget.json`, and `shortlist.json`, and produce `specs/025-8da4w-parameter-sweep/results/sweep-report.md` per contracts/sweep-report-schema.md §5, including the loop-structure re-confirmation table (spec SC-001), the shipped-baseline comparison (spec FR-006/SC-002), and the `4w`-winner cross-shader comparison (spec FR-006/SC-004)
- [X] T028 [US3] Populate the report's search-cost section: compute `estimated_exhaustive_device_seconds` (`total_valid_universe` × average Round-2 per-candidate device time) and compare against actual `total_device_seconds` consumed, confirming ≥5x reduction (spec SC-006)
- [X] T029 [US3] Implement the FR-010 "no improvement" branch in the report generator: if the winner's `mean_gflops` does not exceed the shipped `8da4w` configuration's known throughput, state this explicitly in `sweep-report.md` and set `recommendation: keep_existing_winner` instead of naming a new winner
- [X] T030 [US3] Add the pruning-audit appendix to `sweep-report.md` (or a direct link to `shortlist.json`) so any candidate's fate — shortlisted, anchor, excluded, or eliminated — can be traced without re-running the search (spec FR-009/SC-005)

**Checkpoint**: All three user stories complete. `sweep-report.md` is the decision-ready artifact answering the feature's original question.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and integration with this workstream's existing documentation conventions.

- [X] T031 [P] Run `quickstart.md` end-to-end from a clean state and confirm every "Expected outcome" in it holds
- [X] T032 [P] Add a one-line pointer from `specs/025-8da4w-parameter-sweep/checklists/requirements.md` Notes to the final `results/sweep-report.md` location
- [X] T033 If `sweep-report.md` recommends `productionize_candidate`, add a short cross-reference note in `specs/024-8da4w-slower-than-4w/`'s working notes that this feature's optimal `8da4w` configuration is available as an input to that investigation (per plan.md Assumptions — this feature's result informs but does not close `024`)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately.
- **Foundational (Phase 2)**: Depends on Setup. BLOCKS all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational (specifically T004, T005 — this story is on-device from its first task).
- **User Story 2 (Phase 4)**: Depends on Foundational (T003) AND on User Story 1's output (`dbuf_reconfirmation.json`, T009).
- **User Story 3 (Phase 5)**: Depends on User Story 2's output (`shortlist.json`, T013).
- **Polish (Phase 6)**: Depends on all three user stories being complete.

### User Story Dependencies

- **User Story 1 (P1, MVP)**: Independently testable once Foundational T004/T005 are done. This is the true MVP — it stands alone as "here is which loop structure actually wins, verified fresh," even before any tile/subgroup work exists.
- **User Story 2 (P1)**: Requires US1's `dbuf_reconfirmation.json` as input (the loop structure it fixes for enumeration) — sequential by design, matching spec Assumptions on axis separability.
- **User Story 3 (P2)**: Requires US2's `shortlist.json` as input — likewise sequential.

### Parallel Opportunities

- T004 and T005 (Foundational) are independent of each other and can run in parallel; T003 is independent of both.
- T014 (User Story 2's bounded subgroup=32 re-verification) can run in parallel with T010-T013 — different files/binaries, and explicitly excluded from the search budget.
- T031 and T032 (Polish) are independent and can run in parallel.
- Within User Story 3, T015→T016→T017 are strictly sequential (each edits based on the previous), but T018 (script implementation) can be written in parallel with T015-T017 (different files) as long as it's not *run* until T017 completes.

---

## Parallel Example: Foundational Phase

```bash
# Launch all three foundational checks together:
Task: "Implement tile_constraints.py per T003"
Task: "Confirm execution worktree build is current per T004"
Task: "Confirm M5 EVT1 device/driver/clock state per T005"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: `dbuf_reconfirmation.json` exists and its summary states, with numeric evidence, whether `dbuf2` is actually the fastest loop structure at the shipped geometry — this alone is a useful, reviewable artifact (settling the loop-structure question with fresh measurement) even before any tile/subgroup work begins.

### Incremental Delivery

1. Setup + Foundational → tile-constraint model and device/build readiness in place.
2. User Story 1 → loop-structure winner confirmed by fresh measurement (MVP: "here's which loop shape actually wins, and whether it matches what was reported").
3. User Story 2 → legal `8da4w` tile/subgroup space re-derived and pruned to a budget-capped shortlist.
4. User Story 3 → staged on-device search narrows the shortlist to a confirmed top performer, validated and reported against both the shipped baseline and `4w`'s winner, closing the loop to a decision.

### Notes

- This feature's three user stories are a **pipeline**, not independent parallel workstreams — US2 needs US1's loop-structure output and US3 needs US2's shortlist. Sequencing them in priority order is the actual data dependency, not just a suggestion.
- Every task touching the execution worktree (T004, T006-T009, T014-T023, T025-T026) operates in the dedicated `EXEC-WT` worktree from T002, not this `dev/executorch` worktree — see plan.md "Structure Decision" and research.md Decision 4.
- Commit spec-kit documentation and script changes in this repo (`dev/executorch`, on a feature branch PR'd into `yanwen/dev-1.3` per workspace convention) per this workstream's existing small-commit convention; the execution worktree's shader/dispatch edits are uncommitted experimental work by design (matching how `dbuf1-4`/`tsweep_*` already exist for prior sweeps) unless/until User Story 3 recommends productionizing a winner, at which point porting the winning geometry into a real commit on `dev` is separate follow-on work, not part of this feature.
