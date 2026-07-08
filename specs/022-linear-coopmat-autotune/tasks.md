---

description: "Task list for Smart Autotuning for q4gsw CoopMat Tile Configuration on M5 EVT1"

---

# Tasks: Smart Autotuning for q4gsw CoopMat Tile Configuration on M5 EVT1

**Input**: Design documents from `/specs/022-linear-coopmat-autotune/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/autotune-report-schema.md, quickstart.md

**Tests**: No dedicated unit-test tasks are included — this feature's correctness gate is the existing `COOPMAT_BENCH_CORRECTNESS_ONLY=1` harness (Constitution Principle I), reused as-is rather than reimplemented; verification steps are folded into the implementation tasks below.

**Organization**: Tasks are grouped by user story (spec.md) to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- File paths below are relative to this repo (`quant-perf-optimization/executorch`) unless prefixed `EXEC-WT/`, which means the execution worktree `.artifacts/tsweep-256x256-smoketest/executorch` (see plan.md "Structure Decision").

## Path Conventions

- Analysis/orchestration scripts and all documentation: `specs/022-linear-coopmat-autotune/` in this repo.
- Shader variant catalog and dispatch code: `EXEC-WT/backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coopmat_tsweep.yaml` and `EXEC-WT/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp`.
- Results: `specs/022-linear-coopmat-autotune/results/`.

---

## Phase 1: Setup

**Purpose**: Create the working directories and capture the calibration data this feature's analytical model depends on.

- [X] T001 Create `specs/022-linear-coopmat-autotune/scripts/` and `specs/022-linear-coopmat-autotune/results/` directories
- [X] T002 [P] Record this session's 10 known real on-device measurements (the original 7-config sweep + the 3 configs tried this session: 256×256/K16/4×4, 128×64/K16/4×4 [compile-failed], 128×64/K64/4×4) as `specs/022-linear-coopmat-autotune/results/known-measurements.json`, matching the `MeasurementResult` shape in data-model.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared infrastructure every user story needs. Must complete before any user story phase begins.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T003 [P] Implement the shared tile-constraint validation module in `specs/022-linear-coopmat-autotune/scripts/tile_constraints.py`: given `(wg_tile_m, wg_tile_n, wg_tile_k, sg_grid_x, sg_grid_y, subgroup_size)`, compute `wg_size`, `lds_bytes`, `accumulators_per_sg`, and a `valid` boolean per the four constraints validated against real hardware this session (`WG_SIZE <= 1024`, MMA-alignment, positive-integer staging passes, `LDS <= 65536`); also generate the canonical `tsweep_t<M>x<N>k<K>g<SGX><SGY>s<sub>` token string. This module is shared by `enumerate_configs.py` (US1) and `staged_search.py` (US2). Verified against all 10 `known-measurements.json` entries (T002): 100% match on `valid`/`compile_status` and `token`.
- [X] T004 [P] Confirm the execution worktree `.artifacts/tsweep-256x256-smoketest/executorch` has a current Android build (`cmake-out-android-vk/lib/libvulkan_backend.a` and `cmake-out-android-vk/bench/test_coopmat_linear_bench` both present and newer than the worktree's source files); rebuild per `.shared-context/instruction-for-ai/build.md` §Android arm64 cross-build if stale
- [X] T005 [P] Confirm M5 EVT1 device access, driver identity, and clock pin per quickstart.md Step 3 (`md5sum /vendor/lib64/hw/vulkan.samsung.so`, no `llama`/`coopmat` process running, `pin_freqs.sh` reports 509000/2730000/663000)

**Checkpoint**: Foundation ready — User Story 1 can begin immediately; User Stories 2/3 can begin once US1's shortlist exists.

---

## Phase 3: User Story 1 - Prune the search space with zero device time (Priority: P1) 🎯 MVP

**Goal**: Rank and shortlist the 642 valid buffer-storage tile configurations using only hardware-derived analytical signals, with zero on-device measurement.

**Independent Test**: Run `enumerate_configs.py` then `score_and_shortlist.py` end-to-end and confirm `shortlist.json` has ~30-40 candidates marked `shortlisted: true` (including all known-measurement anchors), with zero adb/device interaction having occurred.

### Implementation for User Story 1

- [X] T006 [US1] Implement `specs/022-linear-coopmat-autotune/scripts/enumerate_configs.py`: iterate `wg_tile_m/n ∈ {16,32,64,128,256}`, `wg_tile_k ∈ {8,16,32,64,128}`, `sg_grid_x/y ∈ {1,2,4,8}`, `subgroup_size ∈ {32,64}`; use `tile_constraints.py` (T003) to filter to `valid=true` only; write `specs/022-linear-coopmat-autotune/results/configs.json` per contracts/autotune-report-schema.md §1
- [X] T007 [US1] Run `enumerate_configs.py` and verify: exactly 642 entries in `configs.json`; `tsweep_t128x128k16g42s32` (dbuf1-equivalent) and `tsweep_t128x64k16g22s32` (prior sweep winner) both present; `tsweep_t128x64k16g44s32` absent (matches this session's real compile failure). NOTE: original spec/plan/tasks text said "321" -- corrected to 642 during implementation (321 was an arithmetic error, an incorrect halving of an already storage-agnostic count); all derived thresholds (SC-001's cap) recomputed accordingly.
- [X] T008 [US1] Implement `specs/022-linear-coopmat-autotune/scripts/score_and_shortlist.py`: for each candidate in `configs.json`, compute `occupancy_proxy = min(65536/lds_bytes, 1024/wg_size)` and `register_penalty = 1 + max(0, accumulators_per_sg - 8) * 0.15` (research.md Decision 2), `score = occupancy_proxy / register_penalty`, rank all 642 by score descending, mark the top ~24-32 `shortlisted: true`, then force `shortlisted: true` for every previously-measured, compiling known config in `known-measurements.json` regardless of rank (research.md Decision 3, revised per the T009 calibration finding); write `specs/022-linear-coopmat-autotune/results/shortlist.json` per contracts/autotune-report-schema.md §2
- [X] T009 [US1] Calibrate the scoring model: score the 10 candidates in `known-measurements.json` (T002) using the same formula, and confirm the ranking is directionally consistent with their real measured throughput — documented in `specs/022-linear-coopmat-autotune/results/calibration-check.md`. OUTCOME: partial pass. The model correctly ranks the two known worst performers at the bottom, but does not reliably rank the single best performer (winner scored 3rd of 9), and would have wrongly dropped a real mid-pack performer (128×256/K16/4×2, real rank 4, scored 8th). Rather than retune the formula against only 10 points (overfitting risk), revised research.md Decision 3 to force-include all 9 compiling known configs into the shortlist regardless of score, not just 2 anchors.
- [X] T010 [US1] Run `score_and_shortlist.py` and verify: `shortlist.json` has 642 total entries (full ranking, per FR-008); all 9 known anchors present with `shortlist_reason` starting `anchor:`. FINAL: 34 shortlisted (25 top-rank + 9 anchors). A first run (37 shortlisted, 28 top-rank) surfaced a second scoring-model flaw beyond T009's finding: the unbounded occupancy proxy top-ranked 28 degenerate single-subgroup (SG_GRID=1x1, WG_SIZE<128) tiles that no real tested config resembles and this shader's double-buffered design can't exploit well. Added a `WG_SIZE >= 128` floor to research.md Decision 2 (candidates below it stay in the full ranking but are never top-ranked) and re-ran to get the final 34-candidate shortlist.

**Checkpoint**: User Story 1 complete — shortlist produced and calibration-checked, zero device time consumed. This alone already satisfies SC-001's ≤96 ceiling (shortlist size ≤34) even before any on-device elimination.

---

## Phase 4: User Story 2 - Find the best performer without measuring everything on the shortlist (Priority: P2)

**Goal**: Measure the shortlisted candidates on M5 EVT1 using a staged, successive-halving-style search so most device time is spent only on the most promising candidates.

**Independent Test**: Run the staged search over `shortlist.json` and confirm every shortlisted candidate receives a Round 1 measurement, only the top third proceeds to Round 2, only the top 3-5 proceed to Round 3, and `budget.json`'s `configs_measured_on_hardware` never exceeds 96.

### Implementation for User Story 2

- [X] T011 [US2] Extend `EXEC-WT/backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coopmat_tsweep.yaml` with one `shader_variants` entry per candidate marked `shortlisted: true` in `shortlist.json`, following the existing entry format (see file's current entries). Only the 25 `top-rank`-reason candidates needed new entries (the 9 anchors already had shader variants from earlier this session); `WEIGHT_STORAGE=texture2d` only per research.md Decision 9. Result: 43 total `shader_variants` entries, all unique, valid YAML.
- [X] T012 [US2] Extend `EXEC-WT/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp`'s `coopmat_variant_tile()` and `kTokens[]` with one token branch per shortlisted candidate, following the existing pattern (depends on T011). 34 total tokens (9 existing + 25 new).
- [X] T013 [US2] Rebuild `vulkan_backend` and `test_coopmat_linear_bench` in the execution worktree (`cmake --build cmake-out-android-vk --target install`, then the bench sub-project); for any candidate whose shader fails to compile, mark `compile_status: compile_failed` in `shortlist.json`, remove its yaml/cpp entries, and rebuild again (depends on T012). Result: all 25 new candidates compiled to SPIR-V successfully, zero compile failures this round.
- [X] T014 [US2] Implement `specs/022-linear-coopmat-autotune/scripts/staged_search.py` Round 1 (`round1_gate`): for each shortlisted candidate with `compile_status: compiles`, run via adb against the rebuilt binary; before the round starts, perform the driver-hash/device-availability/clock-pin check from research.md Decision 7 and abort the round (writing a `halted: true` sentinel per contracts/autotune-report-schema.md §3) if it fails. NOTE: the bench harness has no way to run "just one shape" -- a single `COOPMAT_BENCH_M=2048` invocation always runs the full 12-13 shape sweep plus the (unconditional) small-shape correctness matrix, so Round 1 gets correctness + full-shape perf data in one invocation; staging savings come from candidate-count reduction round to round, not per-invocation cost (documented in the script's module docstring). Parser unit-tested against real captured output from earlier this session before running on-device.
- [X] T015 [US2] Run Round 1 across all compiling shortlisted candidates; write `specs/022-linear-coopmat-autotune/results/round1_results.json` and update `budget.json`; candidates failing to compile or failing correctness are marked `eliminated_at: true` and excluded from Round 2 (depends on T013, T014). RESULT: all 25 correctness-passed; ranked 2518 (top, `t128x64k16g14s32` -- winner's tile, grid 1x4) down to 844 GFLOP/s (mean-of-shapes). Note: this round accidentally ran on the pre-quick-mode binary (already in flight when Decision 10's quick mode was added), so it got full 13-shape data rather than the intended quick 3-shape subset -- higher-fidelity than planned, just slower (~162s/candidate, ~68min total) than intended for an elimination-only pass.
- [X] T016 [US2] Implement `staged_search.py` Round 2 (`round2_full_shapes`): for the top third of Round 1 survivors (ranked by Round 1's single-shape GFLOP/s), run at the harness's default rigor; re-run the Decision 7 pre-round check first. Uses `COOPMAT_BENCH_QUICK=1` (Decision 10) since only 8 candidates remain -- the "full shapes at the very end round only" feedback is honored starting here (Round 1's full-shape data was an artifact of timing, not by design).
- [X] T017 [US2] Run Round 2; write `round2_results.json` and update `budget.json` (depends on T015, T016). RESULT: top 8 survivors re-measured, ranking matches Round 1 almost exactly (same order, ~3% run-to-run noise) -- strong signal, not a fluke. Top 5 (`t128x64k16g14s32`, `t64x128k16g41s32`, `t64x128k16g14s32`, `t128x64k16g41s32`, `t64x64k16g41s32`) advance to Round 3. Budget after Round 2: 25/96 configs measured on hardware.
- [X] T018 [US2] Implement `staged_search.py` Round 3 (`round3_confirmation`): for the top 3-5 Round 2 survivors, repeat the Round 2 measurement across ≥3 independent process invocations to compute `mean_gflops`/`stddev_gflops`/`run_count` (Constitution Principle IV); apply the tie-breaking rule from research.md Decision 6 (prefer smaller `lds_bytes`, then smaller `accumulators_per_sg`) if finalists are statistically indistinguishable; re-run the Decision 7 pre-round check first. Uses the full (non-quick) binary/shapes per Decision 8/10 -- final confirmation must match `jira-tile-sweep.md`'s methodology exactly.
- [ ] T019 [US2] Run Round 3; write `round3_results.json` and update `budget.json` (depends on T017, T018) -- IN PROGRESS as of this checkpoint.
- [ ] T020 [US2] Verify across all three rounds' `budget.json` snapshots that `configs_measured_on_hardware` never exceeded 96 at any checkpoint (SC-001 enforcement)

**Checkpoint**: User Story 2 complete — a top candidate (or explicit confirmation of the existing winner) has been identified with bounded, staged device measurement.

---

## Phase 5: User Story 3 - Validate and report the recommended configuration (Priority: P3)

**Goal**: Confirm the search's winning candidate with the full correctness check, and produce a decision-ready report comparing it to the production baseline and prior sweep winner.

**Independent Test**: Open `autotune-report.md` and confirm it names a winner with a passing correctness result and Round 3 mean/stddev, or explicitly states that no candidate beat the existing winner.

### Implementation for User Story 3

- [ ] T021 [US3] Run `COOPMAT_BENCH_CORRECTNESS_ONLY=1` for the Round 3 winner at the standard multi-tile validation shape (M=K=N=256) via adb against the execution worktree's rebuilt binary; confirm PASS (Constitution Principle I) — if it fails, drop this candidate, fall back to the next Round 3 finalist, and re-run this task
- [ ] T022 [US3] Inspect the winner's compiled SPIR-V (`spirv-dis` or equivalent, from `EXEC-WT/cmake-out-android-vk/vulkan_compute_shaders/<winner-token>_buffer_texture2d_half.spv`) and confirm the expected `OpCooperativeMatrixMulAddKHR`/coopmat-family instructions are present (Constitution Principle VI)
- [ ] T023 [US3] Implement the `staged_search.py --report-only` report generator: read `round3_results.json`, `known-measurements.json`, `budget.json`, and `shortlist.json`, and produce `specs/022-linear-coopmat-autotune/results/autotune-report.md` per contracts/autotune-report-schema.md §5, reusing the exact comparison-table format from `.shared-context/report-for-human/jira-tile-sweep.md`
- [ ] T024 [US3] Populate the report's SC-002 "search cost" section: compute `estimated_exhaustive_device_seconds` (642 × average Round-2 per-candidate device time) and compare against the actual `total_device_seconds` consumed, confirming ≥5x reduction
- [ ] T025 [US3] Implement the FR-009 "no improvement" branch in the report generator: if the winner's `mean_gflops` does not exceed the prior sweep winner's known throughput, state this explicitly in `autotune-report.md` and set `recommendation: keep_existing_winner` instead of naming a new winner
- [ ] T026 [US3] Add the pruning-audit appendix to `autotune-report.md` (or a direct link to `shortlist.json`) so any of the 642 candidates' fate — shortlisted, anchor, or excluded with its analytical score — can be traced without re-running the search (SC-005)

**Checkpoint**: All three user stories complete. `autotune-report.md` is the decision-ready artifact answering the feature's original question.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and integration with this workstream's existing documentation conventions.

- [ ] T027 [P] Run `quickstart.md` end-to-end from a clean state and confirm every "Expected outcome" in it holds
- [ ] T028 [P] Add a one-line pointer from `specs/022-linear-coopmat-autotune/checklists/requirements.md` Notes to the final `results/autotune-report.md` location
- [ ] T029 If `autotune-report.md` recommends `productionize_candidate`, add a short cross-reference note to `.shared-context/report-for-human/RESULTS-SUMMARY.md` per this workstream's "one canonical home per fact" convention (skip this task entirely if the recommendation is `keep_existing_winner`)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately.
- **Foundational (Phase 2)**: Depends on Setup. BLOCKS all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational (specifically T003). Does NOT depend on T004/T005 (no device interaction in US1).
- **User Story 2 (Phase 4)**: Depends on Foundational (T004, T005) AND on User Story 1's output (`shortlist.json`, T010).
- **User Story 3 (Phase 5)**: Depends on User Story 2's output (`round3_results.json`, T019).
- **Polish (Phase 6)**: Depends on all three user stories being complete.

### User Story Dependencies

- **User Story 1 (P1)**: Independently testable once Foundational T003 is done. This is the true MVP — it stands alone as "here is the shortlist and why," even before any device measurement exists.
- **User Story 2 (P2)**: Requires US1's `shortlist.json` as input — not independent of US1 (this is expected; the spec's priority ordering IS the dependency order for this feature, unlike a typical multi-feature app).
- **User Story 3 (P3)**: Requires US2's `round3_results.json` as input — likewise sequential by design.

### Parallel Opportunities

- T002 (Setup) can run in parallel with T001 once the directory exists.
- T003, T004, T005 (Foundational) are independent of each other and can run in parallel.
- T027 and T028 (Polish) are independent and can run in parallel.
- Within User Story 2, T011→T012→T013 are strictly sequential (each edits based on the previous), but T014 (script implementation) can be written in parallel with T011-T013 (different files) as long as it's not *run* until T013 completes.

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
2. Complete Phase 2: Foundational (at minimum T003 — T004/T005 aren't needed until US2)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: `shortlist.json` exists, is calibration-checked, and names both anchors — this alone is a useful, reviewable artifact (a ranked, justified pruning of 642 → ~30 configs) even before any device time is spent.

### Incremental Delivery

1. Setup + Foundational → shortlist scoring infrastructure ready.
2. User Story 1 → shortlist produced (MVP: "here's what's worth trying, and why").
3. User Story 2 → staged on-device search narrows the shortlist to a confirmed top performer within the device-time budget.
4. User Story 3 → the top performer is validated and reported against the production baseline, closing the loop to a decision.

### Notes

- This feature's three user stories are a **pipeline**, not independent parallel workstreams (unlike the template's default assumption) — US2 needs US1's output and US3 needs US2's output. Sequencing them in priority order is not just a suggestion here, it's the actual data dependency.
- Every task touching the execution worktree (T004, T011-T013, T015, T017, T019, T021-T022) operates in `.artifacts/tsweep-256x256-smoketest/executorch`, not this repo — see plan.md "Structure Decision."
- Commit spec-kit documentation and script changes in this repo per this workstream's existing small-commit convention; the execution worktree's shader/dispatch edits are uncommitted experimental work by design (matching how `dbuf1-4`/`tsweep_*` already exist there) unless/until User Story 3 recommends productionizing a winner, at which point porting the winning geometry into a real commit is separate follow-on work, not part of this feature.
