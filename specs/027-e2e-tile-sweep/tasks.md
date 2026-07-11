---

description: "Task list for 8da4w Tile/Subgroup Sweep Ranked by End-to-End Throughput"

---

# Tasks: 8da4w Tile/Subgroup Sweep Ranked by End-to-End Throughput

**Input**: Design documents from `/specs/027-e2e-tile-sweep/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/e2e-ranking-schema.md, quickstart.md

**Tests**: No dedicated unit-test tasks are included — this feature's correctness gate is already satisfied by `025`'s/`026`'s existing per-shape correctness data (Constitution Principle I), reused as a hard filter (spec FR-004) rather than reimplemented; verification steps are folded into the implementation tasks below.

**Organization**: Tasks are grouped by user story (spec.md) to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- File paths below are relative to this repo (`dev/executorch`) unless prefixed `EXEC-WT/`, which means the **existing** `dbuf-int8-sweep` worktree (`023-8da4w-int8-dbuf-sweep-impl` branch) — reused as-is, no new worktree (plan.md Structure Decision, research.md Decision 5).

## Path Conventions

- Analysis/orchestration scripts and all documentation: `specs/027-e2e-tile-sweep/` in this repo.
- Runner/PTEs (no changes needed for US1): `EXEC-WT/cmake-out-android-vk/examples/models/llama/llama_main`, on-device `llama3_1_8b_8da4w_buffer_ctx3072.pte`.
- Shader catalog (extended only if US2 triggers): `EXEC-WT/backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_q4gsw_coopmat_tsweep.{glsl,yaml}`, `EXEC-WT/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp`.
- Results: `specs/027-e2e-tile-sweep/results/`.

---

## Phase 1: Setup

**Purpose**: Create working directories and confirm the reused execution worktree has everything User Story 1 needs already built.

- [X] T001 Create `specs/027-e2e-tile-sweep/scripts/` and `specs/027-e2e-tile-sweep/results/` directories
- [X] T002 Confirm `EXEC-WT` (`dbuf-int8-sweep` worktree) is still on `023-8da4w-int8-dbuf-sweep-impl`, and that `cmake-out-android-vk/examples/models/llama/llama_main` and the full `linear_dq8ca_q4gsw_coopmat_tsweep` shader catalog (all `025`/`026` variants) are present and current; rebuild only if stale

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared infrastructure every user story needs. Must complete before any user story phase begins.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T003 [P] Implement `specs/027-e2e-tile-sweep/scripts/build_prefilter_ranking.py`: merge `specs/025-8da4w-parameter-sweep/results/round3_results.json` (25 subgroup=64 candidates) and `specs/026-8da4w-subgroup32-sweep/results/{round3_results.json,correctness_matrix.json}` (5 subgroup=32 candidates) into one `Candidate` list (data-model.md); filter to `correctness_all_shapes_pass: true` only; sort by `microbenchmark_gflops` descending to assign `microbenchmark_rank`; resolve `shape_family`/`model_used` per candidate (research.md Decision 2 — both sources are `"8B"` today, but the lookup must be a real per-candidate field, not a hardcoded constant); mark the top 8 `shortlisted: true`
- [X] T004 [P] Ensure `llama3_1_8b_8da4w_buffer_ctx3072.pte` is staged on the M5 EVT1 board to be used (push from NFS if missing, per quickstart.md Prerequisites)
- [X] T005 [P] Confirm M5 EVT1 device access, driver identity, and clock pin (quickstart.md Step 2) — record which board for every subsequent result (data-model.md `E2EMeasurement.board`)
- [X] T006 Run `build_prefilter_ranking.py`; verify `specs/027-e2e-tile-sweep/results/prefilter_ranking.json` has exactly 8 `shortlisted: true` entries (or fewer with an explicit note, per contracts §0), all `correctness_all_shapes_pass: true` (depends on T003)

**Checkpoint**: Foundation ready — User Story 1 can begin immediately.

---

## Phase 3: User Story 1 - Rank the existing microbenchmark shortlist by real e2e throughput (Priority: P1) 🎯 MVP

**Goal**: Screen the top-8 shortlist (+ baseline) with one e2e run each; escalate only candidates within 10% of, or ahead of, baseline to a 3-run confirmation; determine whether any confirmed candidate actually beats baseline, and whether microbenchmark rank predicted the e2e outcome.

**Independent Test**: Run screening → escalation → confirmation on the 8-candidate shortlist and confirm every result states its model/shape explicitly, the escalation decision is deterministic from the screen ratio, and the microbenchmark-vs-e2e rank agreement is stated as an explicit finding.

### Implementation for User Story 1

- [X] T007 [US1] Implement `specs/027-e2e-tile-sweep/scripts/run_e2e_screen.py` (measurement mode): for each shortlisted `Candidate` plus `BASELINE_TOKEN`, run one 2048-token-prefill e2e measurement via adb against `EXEC-WT/cmake-out-android-vk/examples/models/llama/llama_main` on the candidate's own `model_used` PTE (`ET_VK_EXECUTE_NODE_THRESHOLD=16`, `p2048_exact.txt`, `num_bos=1`); refuse to record a measurement if `model_used` doesn't match the candidate's `shape_family`-derived value (contracts §1 — the specific anti-regression check for this session's own 1B/8B mistake); before the first measurement under a new `model_used`, run a short-prompt coherence check (Principle VI) and set `coherence_checked: true`
- [X] T008 [US1] Run `run_e2e_screen.py` across all 8 shortlisted candidates + baseline; write `specs/027-e2e-tile-sweep/results/screen_results.json` per contracts §1 (depends on T006, T007)
- [X] T009 [US1] Implement `run_e2e_screen.py --decide-only` mode: compute `screen_ratio` per candidate against baseline's screen result, and `escalated = (screen_ratio >= -0.10)` (research.md Decision 3); write `specs/027-e2e-tile-sweep/results/escalation_decisions.json` per contracts §2
- [X] T010 [US1] Run the decide-only mode; verify every shortlisted candidate has exactly one `escalation_decisions.json` entry (depends on T008, T009)
- [X] T011 [US1] Implement `specs/027-e2e-tile-sweep/scripts/run_e2e_confirm.py`: for every candidate with `escalated: true` in `escalation_decisions.json`, PLUS `BASELINE_TOKEN` (baseline is always confirmed, data-model.md `ConfirmationResult` note), run 3 fresh e2e measurements (research.md Decision 4 — not reusing the screening run); re-verify driver hash/clock pin fresh before this round (Principle VII/VIII); write `specs/027-e2e-tile-sweep/results/confirm_results.json` per contracts §3
- [X] T012 [US1] Implement the `ConfirmationResult` summary step: for each confirmed candidate, compute `mean_prefill_tok_s`/`stddev_prefill_tok_s`/`cov` and compare against baseline's own confirmed mean±stddev range; set `beats_baseline = true` only if `improvement_pct > 0` AND the two 3-run ranges don't overlap (data-model.md derived rule — never asserted by hand); write `specs/027-e2e-tile-sweep/results/confirmation_results.json` per contracts §4
- [X] T013 [US1] Run T011/T012 across all escalated candidates + baseline (depends on T010, T011, T012)
- [X] T014 [US1] Compute the `microbenchmark_vs_e2e_rank_agreement` finding (spec SC-006): compare each candidate's `microbenchmark_rank` (from `prefilter_ranking.json`) against its e2e screen-stage rank; classify as `"agree"` (same top candidate or same overall order), `"partially_agree"` (top candidate differs but overall order is broadly similar), or `"disagree"` (materially different ordering, e.g. `026`'s microbenchmark winner ranking near the bottom of the e2e screen)
- [X] T015 [US1] Determine `FinalAnswer.winner_token` from T013's results: `BASELINE_TOKEN` if no candidate has `beats_baseline: true`, or the winning candidate's token if exactly one does (if more than one does, the fastest confirmed `mean_prefill_tok_s` wins — a documented tie-break, not an arbitrary choice)

**Checkpoint**: User Story 1 complete — either a real e2e winner is confirmed, or the shipped baseline stands, with the microbenchmark-vs-e2e rank-agreement finding recorded either way.

---

## Phase 4: User Story 2 - Smartly extend the search beyond the existing shortlist if the top candidates don't clearly win (Priority: P2)

**Goal**: Only if User Story 1 found no confirmed winner, select and measure a small, budgeted set of new tile/subgroup/subgroup-size candidates end-to-end.

**Independent Test**: Confirm this phase is skipped entirely (no new files, no new builds) if User Story 1 already found a winner; if it runs, confirm new candidates are chosen with a documented rationale, stay within the pre-declared budget cap, and go through the identical screen→escalate→confirm pipeline as User Story 1's original 8.

### Implementation for User Story 2

- [X] T016 [US2] **Gate check**: if `T015`'s `FinalAnswer.winner_token != BASELINE_TOKEN`, SKIP all remaining tasks in this phase (spec FR-006) — do not build or measure anything further; proceed directly to Phase 5
- [ ] T017 [US2] If triggered: using the existing `025`/`026` analytical scoring model plus T014's rank-agreement finding, select a small set of new tile/subgroup/subgroup-size candidates not already in `prefilter_ranking.json` (documented `selection_rationale`, spec FR-007); write `specs/027-e2e-tile-sweep/results/extension_candidates.json` per contracts §5 with a `budget_cap` (small, e.g. ≤5, matching this feature's overall "smartly" instruction)
- [ ] T018 [US2] Add corresponding `shader_variants` entries to `EXEC-WT/backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_q4gsw_coopmat_tsweep.yaml` for any genuinely new tile/grid/subgroup combination selected in T017 (skip if all selected candidates already have shader variants from `025`/`026`); rebuild `EXEC-WT`'s `vulkan_backend` (`--target install`) and `llama_main`/`test_coopmat_linear_bench` (depends on T017)
- [ ] T019 [US2] Run the existing small-shape correctness gate (`COOPMAT_BENCH_CORRECTNESS_ONLY=1`, full representative shape set — `025`/`026`'s convention) for every new candidate from T017; only candidates passing at every shape proceed (spec FR-004) (depends on T018)
- [ ] T020 [US2] Run the identical screen → escalate → confirm pipeline (T007-T013's scripts, re-invoked) for the correctness-surviving extension candidates from T019, appending to (not replacing) the existing `screen_results.json`/`escalation_decisions.json`/`confirm_results.json`/`confirmation_results.json` (depends on T019)
- [ ] T021 [US2] Re-run T015's `FinalAnswer.winner_token` determination including the extension candidates' results (depends on T020)

**Checkpoint**: If triggered, User Story 2 either produces a confirmed winner from the extended set or confirms the baseline stands even after the extension — either way, `FinalAnswer` is now based on the full search this feature actually performed.

---

## Phase 5: User Story 3 - Report a definitive e2e-ranked answer (Priority: P1)

**Goal**: Produce one final, unambiguous report stating the e2e winner (or that baseline stands), the rank-agreement finding, and full auditability of every candidate's disposition.

**Independent Test**: Read `sweep-report.md` and confirm it states exactly one `FinalAnswer`, includes the rank-agreement finding, and every shortlist-eligible candidate not measured has a stated reason.

### Implementation for User Story 3

- [X] T022 [US3] Implement `specs/027-e2e-tile-sweep/scripts/build_report.py`: read `prefilter_ranking.json`, `screen_results.json`, `escalation_decisions.json`, `confirmation_results.json`, and `extension_candidates.json` (if present); produce `specs/027-e2e-tile-sweep/results/sweep-report.md` per contracts §6 — `FinalAnswer` stated first, then the rank-agreement finding, screening/confirmation tables, and (if User Story 2 ran) the extension section
- [X] T023 [US3] Populate the report's "search cost" section: total distinct candidates measured (screen + confirm, both user stories), compared against the target "far fewer than the full legal space" bar (spec SC-004)
- [X] T024 [US3] Populate the skip-reasons appendix (spec SC-005): every correctness-passing `025`/`026` candidate NOT in the top-8 shortlist, with its `microbenchmark_rank` as the reason it was excluded — traceable directly from `prefilter_ranking.json` without re-running anything
- [X] T025 [US3] Run `build_report.py`; verify `sweep-report.md` satisfies every bullet in contracts §6 and the quickstart.md Success check

**Checkpoint**: Feature complete. `sweep-report.md` is the decision-ready artifact answering "what is the current e2e winner" with proper sweep evidence.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and integration with this workstream's existing conventions.

- [X] T026 [P] Run `quickstart.md` end-to-end from a clean state and confirm every "Expected outcome" in it holds
- [X] T027 [P] Add a one-line pointer from `specs/027-e2e-tile-sweep/checklists/requirements.md` Notes to the final `results/sweep-report.md` location
- [X] T028 If `EXEC-WT` gained any new committed-worthy state (US2's extension shader variants, if triggered), commit it to `023-8da4w-int8-dbuf-sweep-impl` per this workstream's "don't let feature-branch work rot uncommitted" convention (matching `026`'s own T036 precedent); if User Story 2 never triggered, state this explicitly rather than leaving it ambiguous

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately.
- **Foundational (Phase 2)**: Depends on Setup. BLOCKS all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational (T006's `prefilter_ranking.json`).
- **User Story 2 (Phase 4)**: Depends on User Story 1's `FinalAnswer` (T015) — gated, may be entirely skipped (T016).
- **User Story 3 (Phase 5)**: Depends on User Story 1's outputs, and User Story 2's outputs if it ran.
- **Polish (Phase 6)**: Depends on all prior phases being complete (including User Story 2's gate resolution either way).

### User Story Dependencies

- **User Story 1 (P1, MVP)**: Independently testable once Foundational T006 is done. This is the true MVP — it alone answers "does the existing top-8 microbenchmark shortlist contain a real e2e winner," which may fully resolve the feature's question without User Story 2 ever running.
- **User Story 2 (P2, conditional)**: Only runs if User Story 1 finds no winner — sequential and conditional by design (spec FR-006), not a parallel workstream.
- **User Story 3 (P1)**: Requires User Story 1's (and, if triggered, User Story 2's) results as input — sequential, the feature's actual deliverable.

### Parallel Opportunities

- T004 and T005 (Foundational) are independent of each other and of T003; T003 has no device dependency and can be written/run before device access is confirmed.
- T026 and T027 (Polish) are independent and can run in parallel.
- Within User Story 1, T007→T008→T009→T010→T011→T012→T013 are strictly sequential (each round depends on the prior's output), but T014 (rank-agreement analysis) only needs T008's screen results and can be computed in parallel with T009-T013's confirmation-stage work.

---

## Parallel Example: Foundational Phase

```bash
# Launch independent foundational tasks together:
Task: "Implement build_prefilter_ranking.py per T003"
Task: "Stage llama3_1_8b_8da4w_buffer_ctx3072.pte per T004"
Task: "Confirm M5 EVT1 device/driver/clock state per T005"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: `confirmation_results.json` and the T014 rank-agreement finding
   exist — this alone may already answer the feature's question (a confirmed winner, or a
   confirmed "baseline stands"), in which case User Story 2 is skipped entirely (T016) and
   only User Story 3's reporting remains.

### Incremental Delivery

1. Setup + Foundational → combined pre-filter ranking ready, zero device time spent yet.
2. User Story 1 → adaptive screen→confirm pipeline run on the top-8 shortlist; likely
   resolves the feature's core question outright (MVP: "here's the real e2e winner, or
   confirmation that baseline still wins").
3. User Story 2 (conditional) → only spends additional device time if User Story 1's
   answer was "baseline stands," extending the search a small, bounded amount.
4. User Story 3 → produces the single decision-ready report closing the loop.

### Notes

- Unlike `025`/`026`, this feature's three user stories are **not** a strict linear
  pipeline where every story always runs — User Story 2 is conditionally skipped
  (spec FR-006/Edge Cases), which is itself the concrete meaning of "(smartly)" once User
  Story 1's outcome is known.
- Every task touching the execution worktree (T002, T004, T007, T008, T011, T013, T018-T020)
  operates in the reused `EXEC-WT` worktree (`dbuf-int8-sweep`), not this `dev/executorch`
  worktree and not a freshly-branched one — see plan.md "Structure Decision" and
  research.md Decision 5.
- T007's `model_used`-vs-`shape_family` consistency check is this feature's single most
  important anti-regression guard — it is the automated version of the mistake a human
  reviewer caught by hand in this same session's `026` Tier-2 validation. Do not weaken or
  bypass it even under time pressure.
- Commit spec-kit documentation and script changes in this repo (`dev/executorch`, on a
  feature branch PR'd into `yanwen/dev-1.3` per workspace convention); T028 explicitly
  resolves the execution worktree's own state, since it is now shared history across three
  features (`025`, `026`, and this one).
