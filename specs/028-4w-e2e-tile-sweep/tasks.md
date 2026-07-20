---

description: "Task list for 4w Tile/Subgroup Sweep Ranked by End-to-End Throughput"

---

# Tasks: 4w Tile/Subgroup Sweep Ranked by End-to-End Throughput

**Input**: Design documents from `/specs/028-4w-e2e-tile-sweep/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/e2e-ranking-schema.md, quickstart.md

**Tests**: No dedicated unit-test tasks are included — this feature's correctness gate for the *shortlist* is already satisfied by `022`'s existing per-shape correctness data (Constitution Principle I), reused as a hard filter (spec FR-004); the one genuinely new correctness step is re-verifying the ported shader (Phase 2.5) against that same gate, folded into the port tasks below rather than a separate test suite.

**Organization**: Tasks are grouped by user story (spec.md) to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- File paths below are relative to this repo (`dev/executorch`) unless prefixed `EXEC-WT/`, which means the **new** execution worktree this feature creates (`4w-e2e-tile-sweep`, branch `028-4w-e2e-tile-sweep`, cut from `yanwen/dev-1.3`) — unlike `027`, there is no existing worktree to reuse here (plan.md Structure Decision, research.md Decision 4).

## Path Conventions

- Analysis/orchestration scripts and all documentation: `specs/028-4w-e2e-tile-sweep/` in this repo.
- Ported shader/dispatch (Phase 2.5): `EXEC-WT/backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coop.glsl` (reference, unmodified), `EXEC-WT/backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coopmat_tsweep.{glsl,yaml}` (new), `EXEC-WT/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp` (new dispatch token).
- Runner/PTEs: `EXEC-WT/cmake-out-android-vk/examples/models/llama/llama_main`, on-device `llama3_1_8b_4w_buffer_ctx3072.pte` (US1-3), `llama3_2_1b_4w_buffer_ctx3072.pte`/`llama3_2_3b_4w_buffer_ctx3072.pte` (US4).
- Results: `specs/028-4w-e2e-tile-sweep/results/`.

---

## Phase 1: Setup

**Purpose**: Create working directories and stand up the new execution worktree — unlike `027`, there is no existing worktree to reuse (research.md Decision 4).

- [X] T001 Create `specs/028-4w-e2e-tile-sweep/scripts/` and `specs/028-4w-e2e-tile-sweep/results/` directories
- [X] T002 Create the execution worktree: `git worktree add 4w-e2e-tile-sweep -b 028-4w-e2e-tile-sweep yanwen/dev-1.3` from `/local/yanwen.xu/workspace`; run `./install_executorch.sh --minimal && pip install -e . --no-build-isolation` in `EXEC-WT/executorch` (quickstart.md Prerequisites — fresh worktree needs this, per this workstream's standing feedback memory)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared infrastructure every user story needs. Must complete before any user story phase begins.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T003 [P] Implement `specs/028-4w-e2e-tile-sweep/scripts/build_prefilter_ranking.py`: read `specs/022-linear-coopmat-autotune/results/round2_results.json` (8 correctness-passing candidates) and cross-reference `results/round3_results.json` to set `microbenchmark_confirmed` on the one matching token; sort by `microbenchmark_gflops`/`mean_gflops` descending to assign `microbenchmark_rank`; set `shape_family: "8B"` and `model_used: "llama3_1_8b_4w_buffer_ctx3072.pte"` for every candidate (research.md Decision 1/2); mark all 8 `shortlisted: true`
- [X] T004 [P] Ensure `llama3_1_8b_4w_buffer_ctx3072.pte`, `llama3_2_1b_4w_buffer_ctx3072.pte`, and `llama3_2_3b_4w_buffer_ctx3072.pte` are staged on the M5 EVT1 board to be used (push from NFS if missing, per quickstart.md Prerequisites — all three already exist on NFS, no export needed)
- [X] T005 [P] Confirm M5 EVT1 device access, driver identity, and clock pin (quickstart.md Step 2) — record which board for every subsequent result (data-model.md `E2EMeasurement.board`)
- [X] T006 Run `build_prefilter_ranking.py`; verify `specs/028-4w-e2e-tile-sweep/results/prefilter_ranking.json` has exactly 8 `shortlisted: true` entries, all `correctness_all_shapes_pass: true` (depends on T003)

**Checkpoint**: Pre-filter ranking ready — but User Story 1 still cannot begin, because no dispatch mechanism exists yet to measure any candidate other than the current fixed default (Phase 2.5 below).

---

## Phase 2.5: Infra Port (Blocking Prerequisite Unique to This Feature)

**Purpose**: Re-derive `4w`'s tile/subgroup dispatch mechanism against `dev`'s current base (research.md Decision 0) — `027` had no equivalent phase, since `8da4w`'s infra was already committed. **No candidate beyond the current fixed dispatch can be measured until this phase passes.**

- [X] T007 Read the archived reference patch at `.archived-artifacts/tmp-origcm-2026-07-08/untracked-new-files/backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coopmat_tsweep.{glsl,yaml}` and `EXEC-WT/backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coop.glsl` (dev's current base) side by side; note every parameterization point (tile size, subgroup grid, subgroup size) the archived file introduces
- [X] T008 Create `EXEC-WT/backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coopmat_tsweep.{glsl,yaml}`: port the tile/subgroup-grid/subgroup-size parameterization from the archived reference (T007) onto `dev`'s current `linear_q4gsw_coop.glsl` structure, preserving any driver-workaround logic already present in the current base (Constitution Principle V) (depends on T007)
- [X] T009 In `EXEC-WT/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp`, add an `ET_VK_Q4GSW_COOPMAT_VARIANT` env-var dispatch token, copying `ET_VK_DQ8CA_COOPMAT_VARIANT`'s existing structure verbatim in the same file; unset/empty/unrecognized value MUST fall back to today's fixed dispatch unchanged (depends on T008)
- [X] T010 Add `shader_variants` entries to `linear_q4gsw_coopmat_tsweep.yaml` for all 8 shortlisted tokens from `prefilter_ranking.json` (T006); rebuild `EXEC-WT`'s `vulkan_backend` (`--target install`) and `llama_main`/`test_coopmat_linear_bench` (depends on T006, T009)
- [X] T011 Run `022`'s existing fp32-reference correctness check (`COOPMAT_BENCH_CORRECTNESS_ONLY=1 ./test_coopmat_linear_bench`) against all 8 shortlisted tokens through the ported shader; write `specs/028-4w-e2e-tile-sweep/results/port_verification.json` per contracts §-1 (depends on T010)
- [X] T012 **Gate check**: any token with `correctness_status: "fail"` in `port_verification.json` MUST be excluded from all downstream measurement, with the failure recorded as its exclusion reason in the eventual report (spec FR-004 extended to the port boundary) — if ALL 8 tokens fail, halt and re-derive the port (T008) rather than proceeding (depends on T011)

**Checkpoint**: The ported dispatch mechanism exists and every shortlisted candidate that survives correctness re-verification is now measurable end-to-end. User Story 1 can begin.

---

## Phase 3: User Story 1 - Rank the existing microbenchmark shortlist by real e2e throughput (Priority: P1) 🎯 MVP

**Goal**: Screen the correctness-surviving shortlist (+ baseline) with one 8B e2e run each; escalate only candidates within 10% of, or ahead of, baseline to a 3-run confirmation; determine whether any confirmed candidate actually beats baseline, and whether microbenchmark rank predicted the e2e outcome.

**Independent Test**: Run screening → escalation → confirmation on the shortlist and confirm every result states its model/shape explicitly, the escalation decision is deterministic from the screen ratio, and the microbenchmark-vs-e2e rank agreement is stated as an explicit finding.

### Implementation for User Story 1

- [X] T013 [US1] Implement `specs/028-4w-e2e-tile-sweep/scripts/run_e2e_screen.py` (measurement mode): for each `port_verification.json`-passing `Candidate` plus `BASELINE_TOKEN`, run one 2048-token-prefill e2e measurement (`model_stage: "8b_search"`) via adb against `EXEC-WT/cmake-out-android-vk/examples/models/llama/llama_main` on `llama3_1_8b_4w_buffer_ctx3072.pte` (`ET_VK_EXECUTE_NODE_THRESHOLD=16`, `p2048_exact.txt`, `num_bos=1`); before the first measurement, run a short-prompt coherence check (Principle VI) and set `coherence_checked: true`
- [X] T014 [US1] Run `run_e2e_screen.py` across all correctness-surviving shortlisted candidates + baseline; write `specs/028-4w-e2e-tile-sweep/results/screen_results.json` per contracts §1 (depends on T012, T013)
- [X] T015 [US1] Implement `run_e2e_screen.py --decide-only` mode: compute `screen_ratio` per candidate against baseline's screen result within `model_stage: "8b_search"`, and `escalated = (screen_ratio >= -0.10)` (research.md Decision 3); write `specs/028-4w-e2e-tile-sweep/results/escalation_decisions.json` per contracts §2
- [X] T016 [US1] Run the decide-only mode; verify every candidate has exactly one `escalation_decisions.json` entry (depends on T014, T015)
- [ ] T017 [US1] Implement `specs/028-4w-e2e-tile-sweep/scripts/run_e2e_confirm.py`: for every candidate with `escalated: true`, PLUS `BASELINE_TOKEN` (baseline is always confirmed), run 3 fresh e2e measurements (research.md Decision 3 — not reusing the screening run); re-verify driver hash/clock pin fresh before this round (Principle VII/VIII); write `specs/028-4w-e2e-tile-sweep/results/confirm_results.json` per contracts §3
  <!-- BLOCKED 2026-07-12: script implemented and 14/18 runs completed (baseline 3/3, tsweep_t128x64k16g14s32 3/3, tsweep_t64x128k16g41s32 3/3, tsweep_t128x64k16g41s32 3/3, tsweep_t64x64k16g21s64 2/3), then halted mid-round on a driver-hash mismatch (unrecognized 21e1251c432ec9c8314470ef63d03e3b, not the documented default f14c51b6f8/c9861e9906...) found on the shared M5 EVT1 board. Reflash-to-default was attempted but denied by the permission system (requires real user authorization, not a coordinator instruction) — per user decision, stopping here rather than reflashing/switching boards/measuring further. Missing: tsweep_t64x64k16g21s64 run 3, and all 3 runs for tsweep_t64x64k16g12s64. Do NOT resume by re-running run_e2e_confirm.py as-is until the driver is re-verified/reflashed to the documented default with real user sign-off — see results/STATUS.md for full detail. -->
- [ ] T018 [US1] Implement the `ConfirmationResult` summary step: for each confirmed candidate, compute `mean_prefill_tok_s`/`stddev_prefill_tok_s`/`cov` and compare against baseline's own confirmed mean; set `improvement_pct`; write `specs/028-4w-e2e-tile-sweep/results/confirmation_results.json` per contracts §3
- [ ] T019 [US1] Run T017/T018 across all escalated candidates + baseline (depends on T016, T017, T018)
- [ ] T020 [US1] Compute the microbenchmark-vs-e2e rank-agreement finding (spec SC-006, data-model.md `RankAgreementFinding`): compare each candidate's `microbenchmark_rank` (from `prefilter_ranking.json`) against its e2e screen-stage rank; classify `"agree"` / `"partially_agree"` / `"disagree"`
- [ ] T021 [US1] Determine the 8B `winner_token` from T019's results: `BASELINE_TOKEN` if no candidate's confirmed range clearly beats baseline's, or the winning candidate's token if exactly one does (ties broken by fastest confirmed `mean_prefill_tok_s`, documented not arbitrary)

**Checkpoint**: User Story 1 complete — either a real 8B e2e winner is confirmed, or the shipped baseline stands, with the rank-agreement finding recorded either way.

---

## Phase 4: User Story 2 - Smartly extend the search beyond the existing shortlist if the top candidates don't clearly win (Priority: P2)

**Goal**: Only if User Story 1 found no confirmed winner, select and measure a small, budgeted set of new tile/subgroup/subgroup-size candidates end-to-end.

**Independent Test**: Confirm this phase is skipped entirely (no new files, no new builds) if User Story 1 already found a winner; if it runs, confirm new candidates are chosen with a documented rationale, stay within the pre-declared budget cap, and go through the identical screen→escalate→confirm pipeline as User Story 1's original set.

### Implementation for User Story 2

- [ ] T022 [US2] **Gate check**: if T021's `winner_token != BASELINE_TOKEN`, SKIP all remaining tasks in this phase (spec FR-006) — do not build or measure anything further; proceed directly to Phase 5
- [ ] T023 [US2] If triggered: using `022`'s existing analytical scoring model (Round 1 zero-device-time proxies) plus T020's rank-agreement finding, select a small set of new tile/subgroup/subgroup-size candidates not already in `prefilter_ranking.json` (documented `selection_rationale`, spec FR-007); write `specs/028-4w-e2e-tile-sweep/results/extension_candidates.json` per data-model.md `SearchExtension`, with a small `budget_cap` (e.g. ≤5)
- [ ] T024 [US2] Add corresponding `shader_variants` entries to `EXEC-WT/backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coopmat_tsweep.yaml` for any genuinely new tile/grid/subgroup combination selected in T023; rebuild `EXEC-WT`'s `vulkan_backend` (`--target install`) and `llama_main`/`test_coopmat_linear_bench` (depends on T023)
- [ ] T025 [US2] Run `022`'s existing correctness gate (`COOPMAT_BENCH_CORRECTNESS_ONLY=1`) for every new candidate from T023; only candidates passing proceed (spec FR-004) (depends on T024)
- [ ] T026 [US2] Run the identical screen → escalate → confirm pipeline (T013-T019's scripts, re-invoked) for the correctness-surviving extension candidates, appending to (not replacing) the existing `screen_results.json`/`escalation_decisions.json`/`confirm_results.json`/`confirmation_results.json` (depends on T025)
- [ ] T027 [US2] Re-run T021's `winner_token` determination including the extension candidates' results (depends on T026)

**Checkpoint**: If triggered, User Story 2 either produces a confirmed winner from the extended set or confirms the baseline stands even after the extension.

---

## Phase 5: User Story 3 - Report a definitive e2e-ranked answer for 4w (Priority: P1)

**Goal**: Produce one final, unambiguous 8B answer stating the e2e winner (or that baseline stands), the rank-agreement finding, and full auditability of every candidate's disposition.

**Independent Test**: Read `final_8b_answer.json` and confirm it states exactly one winner, includes the rank-agreement finding, and every shortlist-eligible candidate not confirmed has a stated reason.

### Implementation for User Story 3

- [ ] T028 [US3] Implement `specs/028-4w-e2e-tile-sweep/scripts/build_report.py` (`--stage final-8b` mode): read `prefilter_ranking.json`, `port_verification.json`, `screen_results.json`, `escalation_decisions.json`, `confirmation_results.json`, and `extension_candidates.json` (if present); produce `specs/028-4w-e2e-tile-sweep/results/final_8b_answer.json` per contracts §4 — `winner_token` stated first, then `rank_agreement`, then `excluded_candidates` with a reason for every non-confirmed candidate (spec SC-005) and every port-correctness failure from T012
- [ ] T029 [US3] Populate the report's "search cost" section: total distinct candidates measured (screen + confirm, both user stories), compared against the target "far fewer than the full legal space" bar (spec SC-004)
- [ ] T030 [US3] Run `build_report.py --stage final-8b`; verify `final_8b_answer.json` satisfies every bullet in contracts §4

**Checkpoint**: The 8B question is answered definitively — either a specific winner with full evidence, or an explicit "baseline stands" statement.

---

## Phase 6: User Story 4 - Confirm the 8B-derived answer holds on 1B and 3B (Priority: P2)

**Goal**: Take User Story 3's exact final 8B config (winner or baseline) and measure it end-to-end on 1B and 3B, reporting per-size whether the 8B finding's direction holds, is neutral, or reverses (spec FR-012, Clarifications 2026-07-12).

**Independent Test**: Given `final_8b_answer.json`, confirm no new candidate is searched for 1B/3B — only the one config from User Story 3 is measured — and confirm the report states, per model size, whether the 8B finding's direction holds (spec Acceptance Scenarios 1-3).

### Implementation for User Story 4

- [ ] T031 [US4] Implement `specs/028-4w-e2e-tile-sweep/scripts/run_1b3b_confirmation.py`: read `final_8b_answer.json`'s `winner_token`; for each of `llama3_2_1b_4w_buffer_ctx3072.pte` and `llama3_2_3b_4w_buffer_ctx3072.pte`, run one e2e screening measurement (`model_stage: "1b3b_confirmation"`) for the winner config AND the baseline config on that model size (coherence-checked per new `model_used`, Principle VI)
- [ ] T032 [US4] Apply the same escalation bar (research.md Decision 3) per model size: if the winner's screen_ratio vs. that size's own baseline is within 10% of, or ahead of, baseline, run a 3-run confirmation for that size (reuse `run_e2e_confirm.py` with `--model-stage 1b3b_confirmation`) (depends on T031)
- [ ] T033 [US4] Compute each `CrossSizeFinding` (data-model.md): `direction` = `"holds"` if the 8B finding's sign (win/neutral/loss) matches this size's own confirmed-or-screened result, `"neutral"` if within noise of this size's own baseline, `"reverses"` if the opposite direction from the 8B finding; write `specs/028-4w-e2e-tile-sweep/results/cross_size_confirmation.json` per contracts §5 — exactly one entry per model size, even when the 8B answer is "baseline stands" (depends on T032)
- [ ] T034 [US4] Run `build_report.py` (`--stage full-report` mode): merge `final_8b_answer.json` and `cross_size_confirmation.json` into `specs/028-4w-e2e-tile-sweep/results/sweep-report.md` per contracts §6, stating the 8B answer, the rank-agreement finding, and the 1B/3B cross-size finding explicitly (spec SC-007) (depends on T028, T033)

**Checkpoint**: Feature complete. `sweep-report.md` is the decision-ready artifact answering "what is the current e2e winner for 4w" at 8B, with the 1B/3B generalization question also answered rather than left implicit.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and integration with this workstream's existing conventions.

- [ ] T035 [P] Run `quickstart.md` end-to-end from a clean state and confirm every "Expected outcome" in it holds
- [ ] T036 [P] Add a one-line pointer from `specs/028-4w-e2e-tile-sweep/checklists/requirements.md` Notes to the final `results/sweep-report.md` location
- [ ] T037 Commit the ported shader/dispatch code (Phase 2.5) and any User-Story-2 extension shader variants to `028-4w-e2e-tile-sweep` on `EXEC-WT`, and push to `origin` promptly per this workspace's "don't let feature-branch work rot uncommitted" convention — this is new committed history, unlike `027`'s reuse of already-pushed infra
- [ ] T038 Update `.shared-context/ACTIVE-STATUS.md` with this feature's 8B answer and 1B/3B cross-size finding, following the same convention used for `specs/027`'s shipped-result entry

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately.
- **Foundational (Phase 2)**: Depends on Setup. Produces the pre-filter ranking but does NOT unblock User Story 1 by itself.
- **Infra Port (Phase 2.5)**: Depends on Phase 1 (needs `EXEC-WT`); independent of Phase 2's `prefilter_ranking.json` content but needs its candidate token list (T006) to know which `shader_variants` entries to add (T010). BLOCKS all user stories — unique to this feature, `027` had no equivalent gate.
- **User Story 1 (Phase 3)**: Depends on Phase 2 (T006) AND Phase 2.5 (T012).
- **User Story 2 (Phase 4)**: Depends on User Story 1's `winner_token` (T021) — gated, may be entirely skipped (T022).
- **User Story 3 (Phase 5)**: Depends on User Story 1's outputs, and User Story 2's outputs if it ran.
- **User Story 4 (Phase 6)**: Depends on User Story 3's `final_8b_answer.json` (T028/T030) — cannot start until the 8B question has exactly one answer.
- **Polish (Phase 7)**: Depends on all prior phases being complete (including User Story 2's gate resolution either way, and User Story 4).

### User Story Dependencies

- **User Story 1 (P1, MVP)**: Testable once Foundational (T006) AND Infra Port (T012) are done — unlike `027`, the MVP here has two blocking prerequisite phases, not one.
- **User Story 2 (P2, conditional)**: Only runs if User Story 1 finds no winner — sequential and conditional by design (spec FR-006).
- **User Story 3 (P1)**: Requires User Story 1's (and, if triggered, User Story 2's) results as input.
- **User Story 4 (P2)**: Requires User Story 3's single definitive 8B answer as input — a pure confirmation pass, never an independent search (spec FR-012).

### Parallel Opportunities

- T004 and T005 (Foundational) are independent of each other and of T003.
- T002 (worktree setup) can run in parallel with T003 (pre-filter script has no worktree dependency).
- T035 and T036 (Polish) are independent and can run in parallel.
- Within User Story 1, T013→T014→T015→T016→T017→T018→T019 are strictly sequential, but T020 (rank-agreement analysis) only needs T014's screen results and can run in parallel with T017-T019's confirmation-stage work.
- Within User Story 4, the 1B and 3B measurements (T031-T033) are independent of each other per model size and can be parallelized across two device sessions if two boards are available, though a single board running them sequentially is equally valid.

---

## Parallel Example: Foundational Phase

```bash
# Launch independent foundational tasks together:
Task: "Implement build_prefilter_ranking.py per T003"
Task: "Stage llama3_{1_8b,2_1b,2_3b}_4w_buffer_ctx3072.pte per T004"
Task: "Confirm M5 EVT1 device/driver/clock state per T005"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 2.5: Infra Port — **cannot be skipped**, unlike `027`'s equivalent point in its pipeline, since no measurable variant dispatch exists until this lands
4. Complete Phase 3: User Story 1
5. **STOP and VALIDATE**: `confirmation_results.json` and the T020 rank-agreement finding
   exist — this alone may already answer the 8B question (a confirmed winner, or a
   confirmed "baseline stands"), in which case User Story 2 is skipped entirely (T022) and
   User Story 3's reporting, then User Story 4's 1B/3B confirmation, remain.

### Incremental Delivery

1. Setup + Foundational + Infra Port → pre-filter ranking ready AND the dispatch mechanism
   to actually measure it exists — this combined block is this feature's true prerequisite,
   larger than `027`'s equivalent since the port has no shortcut.
2. User Story 1 → adaptive screen→confirm pipeline run on the shortlist; likely resolves
   the feature's core 8B question outright.
3. User Story 2 (conditional) → only spends additional device time if User Story 1's
   answer was "baseline stands."
4. User Story 3 → produces the single decision-ready 8B answer.
5. User Story 4 → confirms that exact 8B answer generalizes (or doesn't) to 1B/3B, closing
   the specific gap the user's "also try on 1b and 3b" instruction raised.

### Notes

- Unlike `025`/`026`/`027`, this feature has a **mandatory, non-reusable infra-port phase**
  (Phase 2.5) before any user story can begin — the single biggest structural difference
  from its own template feature. Do not treat T007-T012 as optional polish; they are
  the actual reason this feature is more than "run 027's scripts on different data."
- Every task touching the execution worktree (T002, T008-T011, T013, T014, T017, T019,
  T024-T026, T031-T033, T037) operates in the **new** `EXEC-WT` worktree
  (`4w-e2e-tile-sweep`), not this `dev/executorch` worktree — see plan.md "Structure
  Decision" and research.md Decision 4.
- T013's `model_used`-vs-`shape_family` consistency check (inherited from `027`'s own
  anti-regression guard) applies equally to the new `model_stage` dimension User Story 4
  introduces — a 1B/3B measurement recorded under the wrong `model_stage` or `model_used`
  is the same class of bug `027` had to catch and fix for `026`, just with one more axis.
- T037 explicitly commits and pushes the new execution worktree's state, since — unlike
  `027`, which reused already-pushed infra — this feature's port work has nowhere else to
  live once the worktree is eventually retired.
