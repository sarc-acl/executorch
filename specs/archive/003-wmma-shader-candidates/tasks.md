---

description: "Task list for WMMA-Optimizable Shader Candidates Report"
---

# Tasks: WMMA-Optimizable Shader Candidates Report

**Input**: Design documents from `/specs/003-wmma-shader-candidates/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md (all present)

**Tests**: Not requested — this feature's own "test" is manual verification against
cited source lines (see US1's verification task) rather than an automated test suite.

**Organization**: Tasks are grouped by user story to enable independent implementation
and testing of each story. This is a pure analysis feature (no device/build access,
no product code changes) — all "implementation" is one Python script plus its outputs.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Paths are relative to the repository root

## Path Conventions

Single analysis project under `specs/003-wmma-shader-candidates/`:
- `scripts/classify_shaders.py` — the one script (per-config classification + report generation)
- `results/classifications/<model>_<scheme>.json` — per-config output
- `results/wmma-candidates-report.md` — the consolidated deliverable

---

## Phase 1: Setup

**Purpose**: Project initialization

- [X] T001 Create `specs/003-wmma-shader-candidates/scripts/` and `specs/003-wmma-shader-candidates/results/classifications/` directories
- [X] T002 Create `specs/003-wmma-shader-candidates/scripts/classify_shaders.py` with an argparse skeleton: per-config mode (`--model`, `--scheme`, `--profiling-json`, `--out`) and report mode (`--generate-report`, `--classifications-dir`, `--out`)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The shared classification rule engine that every user story's per-config
classification depends on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T003 Implement the classification rule engine in `classify_shaders.py`: given one `002` aggregated entry (`kernel_name`, `shape`, `category`, `total_time_us`, `pct_of_phase`), return `(classification, blocking_reasons, existing_or_prospective_shader)` per `research.md` Decision 2's rule table — `category == "non-shader overhead"` → `d`; prefill linear family (categories attention-projection/feed-forward/output-projection, kernel name matches `gemm`/`_tiled`) → `b` with **both** cited blocking reasons (rank-3 output, `TEXTURE_3D` storage); decode linear family (same categories, `gemv`/`_coop`) → `c`; `category == "attention (sdpa)"` (either phase) → `c`; anything unmatched → `uncertain` per FR-008
- [X] T004 Implement per-config classification in `classify_shaders.py`: load one `002` `results/raw/<model>_<scheme>.json`, apply T003's engine to every `aggregated[]` entry in both phases, write the classification JSON matching `contracts/candidates-report-schema.md` (depends on T003)

**Checkpoint**: Foundation ready — per-config classification can now run

---

## Phase 3: User Story 1 - Classify every shader from the profiling report by WMMA candidacy (Priority: P1) 🎯 MVP

**Goal**: Prove the classification method end-to-end on one configuration, with every
`b`/`c` classification citing a specific, verified reason.

**Independent Test**: Run the classifier on `llama-3.2-1b_4w`'s existing `002` data
and manually verify every classification against `QuantizedLinear.cpp`/`SDPA.cpp`.

- [X] T005 [US1] Run `classify_shaders.py` on `llama-3.2-1b_4w` (both phases) → `specs/003-wmma-shader-candidates/results/classifications/llama-3.2-1b_4w.json` (depends on T004)
- [X] T006 [US1] Manually verify every `b`/`c` classification's `blocking_reasons` against the cited `QuantizedLinear.cpp`/`SDPA.cpp` lines (`quickstart.md` step 2): confirm the prefill linear family shows **exactly two** blocking reasons, decode linears are classified `c` (not `b`), `sdpa`-category entries are classified `c` in both phases, and no `classification: "a"` appears anywhere (depends on T005)
- [X] T007 [US1] Fix any classification-engine bugs found during T006's verification (depends on T006)

**Checkpoint**: US1 complete — one configuration is fully, correctly classified; the method is proven before scaling out

---

## Phase 4: User Story 2 - Extend the classification to all six configurations (Priority: P2)

**Goal**: Confirm the classification pattern holds (or find where it doesn't) across
all three models and both quantization schemes.

**Independent Test**: Run the classifier on the remaining five configurations and
confirm every shader in each has a complete classification.

- [X] T008 [P] [US2] Run `classify_shaders.py` on `llama-3.2-1b_8da4w` → `results/classifications/llama-3.2-1b_8da4w.json` (depends on T007)
- [X] T009 [P] [US2] Run `classify_shaders.py` on `llama-3.2-3b_4w` → `results/classifications/llama-3.2-3b_4w.json` (depends on T007)
- [X] T010 [P] [US2] Run `classify_shaders.py` on `llama-3.2-3b_8da4w` → `results/classifications/llama-3.2-3b_8da4w.json` (depends on T007)
- [X] T011 [P] [US2] Run `classify_shaders.py` on `llama-3.1-8b_4w` → `results/classifications/llama-3.1-8b_4w.json` (depends on T007)
- [X] T012 [P] [US2] Run `classify_shaders.py` on `llama-3.1-8b_8da4w` → `results/classifications/llama-3.1-8b_8da4w.json` (depends on T007)
- [X] T013 [US2] Cross-check all six configurations' classifications for consistency: confirm the same root causes get the same `classification`/`blocking_reasons` across every model/scheme, varying only in `total_time_us`/`pct_of_phase` (depends on T008, T009, T010, T011, T012)

**Checkpoint**: All six configurations classified; the pattern is confirmed consistent (or documented divergences are noted)

---

## Phase 5: User Story 3 - Produce a ranked WMMA-optimization candidates report (Priority: P3)

**Goal**: Roll the six configurations' classifications up into one actionable, ranked
report.

**Independent Test**: Generate the report from the completed classifications alone
(no re-classification, no re-profiling) and verify its ranking and grouping.

- [X] T014 [US3] Implement Optimization Candidate Group construction in `classify_shaders.py`: group all `b`/`c` Shader Classification rows across all six configs by shared root cause (`research.md` Decision 3's four groups), summing `total_time_us` per group for the primary sort key, retaining per-config breakdown (depends on T013)
- [X] T015 [US3] Implement `--generate-report` mode in `classify_shaders.py`: render the two-section ranked markdown report ("existing implementation blocked" / "no WMMA implementation exists"), each group sorted by `total_time_us_summed` descending with a per-config breakdown table and full `blocking_reasons`, per `contracts/candidates-report-schema.md` (depends on T014)
- [X] T016 [US3] Run `classify_shaders.py --generate-report` → `specs/003-wmma-shader-candidates/results/wmma-candidates-report.md` (depends on T015)
- [X] T017 [US3] Self-review the report against `quickstart.md` step 5 and spec.md's SC-003/SC-004/SC-005: confirm the top-ranked group in each section is identifiable without reading raw data, every group states fix-vs-new-authoring clearly, and the report states no `classification: "a"` entries exist anywhere (FR-009) (depends on T016)

**Checkpoint**: Ranked report complete and self-reviewed

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T018 [P] Cross-check every Optimization Candidate Group's `total_time_us_summed` against `002`'s `profiling-report.md` category percentages as a sanity check — no group's absolute time should contradict `002`'s already-published category breakdown
- [X] T019 Update `specs/003-wmma-shader-candidates/quickstart.md` with any corrections found during T006/T013/T017 (if any were needed)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup. BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational
- **User Story 2 (Phase 4)**: Depends on US1 (T007) — reuses the same proven engine; not run before US1 is verified, since re-running an unverified engine five more times would just multiply any bug
- **User Story 3 (Phase 5)**: Depends on US2 (T013) — needs all six configurations' classifications to group/rank across the full matrix
- **Polish (Phase 6)**: Depends on US3

### Within Each User Story

- US2: T008-T012 (the five remaining per-config runs) are parallel (different output files, same proven engine); T013 (cross-check) runs after all five complete
- US3: T014 (grouping) before T015 (report rendering) before T016 (run) before T017 (self-review) — strictly sequential, each depends on the prior step's output

### Parallel Opportunities

- T008-T012 (five per-config classification runs in US2) can run in parallel — different output files, no shared state
- T018 (Polish sanity-check) has no dependency on T019 and could run alongside it

---

## Parallel Example: User Story 2

```bash
# Launch the five remaining per-config classifications together:
Task: "Run classify_shaders.py on llama-3.2-1b_8da4w -> results/classifications/llama-3.2-1b_8da4w.json"
Task: "Run classify_shaders.py on llama-3.2-3b_4w -> results/classifications/llama-3.2-3b_4w.json"
Task: "Run classify_shaders.py on llama-3.2-3b_8da4w -> results/classifications/llama-3.2-3b_8da4w.json"
Task: "Run classify_shaders.py on llama-3.1-8b_4w -> results/classifications/llama-3.1-8b_4w.json"
Task: "Run classify_shaders.py on llama-3.1-8b_8da4w -> results/classifications/llama-3.1-8b_8da4w.json"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (the classification engine)
3. Complete Phase 3: User Story 1 — one configuration, fully verified against source
4. **STOP and VALIDATE**: every `b`/`c` classification traces to a real, cited line of code

### Incremental Delivery

1. Setup + Foundational → engine ready
2. US1 → one config classified and verified → proves the method
3. US2 → all six configs classified → proves the pattern holds everywhere
4. US3 → ranked report → the actual deliverable the user asked for

---

## Notes

- No test-writing tasks: this feature's correctness check is manual verification
  against cited source lines (T006), not an automated test suite — appropriate for
  a one-shot analysis report, not ongoing product code.
- Every task after T007 reuses T003's engine unchanged — if a config's classification
  ever needs a *different* rule than what T003 encodes, that is itself a finding to
  surface in T013's cross-check, not something to special-case silently.
- No commits until the user explicitly asks, per repo convention.
