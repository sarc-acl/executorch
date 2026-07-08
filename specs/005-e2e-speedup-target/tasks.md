---

description: "Task list for End-to-End Speedup Target and Validation"
---

# Tasks: End-to-End Speedup Target and Validation

**Input**: Design documents from `/specs/005-e2e-speedup-target/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md (all present)

**Tests**: Not requested as an automated suite — this feature's correctness check IS
User Story 3's self-test (synthetic scenarios proving the verdict logic), which is
itself part of the task list below, not a separate test phase.

**Organization**: Tasks are grouped by user story. This is a pure tooling feature (no
device access, no product code changes) — Story 2's "real re-measurement" and Story
3's "real outcome report" cannot fully execute yet (no optimization build exists), so
their tasks build and validate the tooling via the self-test, per `research.md`
Decision 3, rather than fabricating real numbers.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Paths are relative to the repository root

## Path Conventions

Single tooling project under `specs/005-e2e-speedup-target/`:
- `scripts/compute_outcome.py` — the one script (target generation, comparison engine, self-test, report rendering)
- `results/speedup-target.json` — Story 1's real deliverable
- `results/selftest/` — synthetic scenarios + self-test report (Story 3's validation)
- `results/outcome-report.md` — Story 3's real deliverable, **not produced by this task list** (requires a future optimization build)

---

## Phase 1: Setup

- [X] T001 Create `specs/005-e2e-speedup-target/scripts/` and `specs/005-e2e-speedup-target/results/selftest/` directories
- [X] T002 Create `specs/005-e2e-speedup-target/scripts/compute_outcome.py` with an argparse skeleton: `--generate-target`, `--selftest`, and the real-invocation mode (`--target`, `--after-dir`, `--out`)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The comparison/verdict engine every user story depends on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T003 Implement target generation in `compute_outcome.py`: read all six `specs/001-minipc-baseline-benchmarks/results/raw/<model>_<scheme>.json` files, compute `target_prefill_tokens_per_sec = baseline.e2e.prefill_tokens_per_sec * 2.0` per config, write `speedup-target.json` matching `contracts/outcome-schema.md` (depends on T002)
- [X] T004 Implement the core verdict engine in `compute_outcome.py`: given one Speedup Target entry + one Re-Measurement entry (`data-model.md`), compute `observed_multiplier`, apply `research.md` Decision 4's thresholds (`<1.0`→regressed, `[1.0,2.0)`→missed, `~2.0` within `baseline_prefill_stdev`→met, `>2.0` beyond that band→exceeded), compute `combined_e2e_change_pct` as a tracked-only field that never feeds `verdict`, and return an Outcome record (depends on T002)

**Checkpoint**: Foundation ready — target generation and verdict computation both work

---

## Phase 3: User Story 1 - Define an unambiguous prefill success target (Priority: P1) 🎯 MVP

**Goal**: Produce the real, traceable Speedup Target from `001`'s existing baseline data.

**Independent Test**: Confirm `speedup-target.json` exists with 6 entries, each exactly 2x its cited baseline.

- [X] T005 [US1] Run `compute_outcome.py --generate-target` → `specs/005-e2e-speedup-target/results/speedup-target.json` (depends on T003)
- [X] T006 [US1] Verify all 6 entries: `target_prefill_tokens_per_sec` is exactly `baseline_prefill_tokens_per_sec * 2.0`, and `baseline_source` resolves to a real, existing `001` file for each (depends on T005)

**Checkpoint**: US1 complete — the target is formally recorded and traceable, before any optimization implementation begins (FR-002)

---

## Phase 4: User Story 2 - Re-measure end-to-end performance once optimization work exists (Priority: P2)

**Goal**: Make the tool ready to ingest a real future re-measurement the moment one exists, enforcing the same-methodology requirement (FR-003/FR-008).

**Independent Test**: Feed the tool a directory of (synthetic, for now) re-measurement JSON files and confirm it correctly loads, matches each to its target, and flags any methodology divergence — without needing a real optimization build to exist yet.

- [X] T007 [US2] Implement `--after-dir` loading in `compute_outcome.py`: read a directory of re-measurement JSON files named `<model>_<scheme>.json` (matching `001`'s convention), match each to its Speedup Target entry by (model, scheme) (depends on T004, T006)
- [X] T008 [US2] Implement the methodology-comparability check (FR-008): if an after-JSON's `e2e.prefill_tokens`/`e2e.decode_tokens` don't match the fixed 2048/1024 workload, or it sets `methodology_comparable: false` itself, propagate `verdict: not_comparable` with `observed_multiplier: null` and surface its `methodology_note` (depends on T007)

**Checkpoint**: US2 complete — the tool can ingest and validate a re-measurement; only a real optimization build's output is still needed, which is future work outside this feature

---

## Phase 5: User Story 3 - Report the outcome against the target, per model (Priority: P3)

**Goal**: Render the Outcome Report, and prove the entire pipeline correct via a clearly-labeled synthetic self-test before it is ever pointed at real data.

**Independent Test**: Run the self-test and confirm all five verdicts (`met`/`exceeded`/`missed`/`regressed`/`not_comparable`) appear correctly and are traceable to their constructed scenario.

- [X] T009 [US3] Implement the report renderer in `compute_outcome.py`: per-(model,scheme) section showing baseline, target, actual re-measured prefill number, `verdict`, and the tracked `combined_e2e_change_pct` — plus a top-level summary table scannable without opening per-config detail (depends on T004)
- [X] T010 [US3] Implement `--selftest` mode: construct 5 synthetic after-JSON scenarios (exactly-2x/met, above-2x/exceeded, below-2x/missed, below-baseline/regressed, mismatched-workload/not_comparable) across the 6 real configs, run them through the same T004/T007/T008/T009 code path, write to `results/selftest/` with every entry's `is_synthetic: true` and the report headed "SYNTHETIC SELF-TEST DATA — NOT A REAL MEASUREMENT" (depends on T007, T008, T009)
- [X] T011 [US3] Run `--selftest` and verify: all 5 verdicts appear and match their constructed scenario; `combined_e2e_change_pct` is shown but never determines `verdict` (a scenario with a large combined e2e change but a sub-2x prefill multiplier still shows `missed`); `not_comparable` entries show `observed_multiplier: null` and their `methodology_note` (depends on T010)
- [X] T012 [US3] Fix any bugs found during T011's verification (depends on T011)

**Checkpoint**: US3 complete — the full pipeline (target → re-measurement ingestion → verdict → report) is proven correct on synthetic data; the tool is ready to run for real the moment a real optimization build and re-measurement exist

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T013 [P] Confirm `results/selftest/` and the (not-yet-created) `results/outcome-report.md` path are structurally and visibly separated, so a future reader cannot mistake the self-test output for a real result
- [X] T014 Update `quickstart.md` with any corrections found during T006/T011 (if any were needed)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup. BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational (T003)
- **User Story 2 (Phase 4)**: Depends on Foundational (T004) and US1 (T006) — needs a verified target to match re-measurements against
- **User Story 3 (Phase 5)**: T009 depends only on Foundational (T004); T010-T012 depend on US2 (T007, T008) since the self-test exercises the full ingestion+comparison+report pipeline together
- **Polish (Phase 6)**: Depends on US3

### Within Each User Story

- US3: T009 (renderer) can start as soon as Foundational is done, in parallel with US2's T007/T008; T010 (self-test) needs all of T007-T009 done first

### Parallel Opportunities

- T009 [US3] can run in parallel with T007/T008 [US2] — different concerns (rendering vs. ingestion/validation) even though T010 later needs all three done

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (target generation + verdict engine)
3. Complete Phase 3: User Story 1 — the target is formally recorded, before any optimization work begins
4. **STOP and VALIDATE**: `speedup-target.json` is correct and traceable

### Incremental Delivery

1. Setup + Foundational → engine ready
2. US1 → target recorded (real deliverable, usable immediately)
3. US2 → ingestion/validation logic ready for a future real re-measurement
4. US3 → self-test proves the whole pipeline correct; real `outcome-report.md` is generated later, outside this task list, once real optimization work exists

---

## Notes

- No commits until the user explicitly asks, per repo convention.
- The real `results/outcome-report.md` is intentionally not a task here — producing it now would require fabricating data. `quickstart.md` step 4 documents exactly how to produce it later, once real work exists.
