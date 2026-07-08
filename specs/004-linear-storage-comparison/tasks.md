---

description: "Task list for Linear Shader Storage-Type Baseline Study"
---

# Tasks: Linear Shader Storage-Type Baseline Study (Texture3D vs. Buffer)

**Input**: Design documents from `/specs/004-linear-storage-comparison/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md (all present)

**Tests**: Not requested as a separate automated suite — this feature's correctness
checks ARE tasks in the list below (kernel-name verification, cross-check against
`001`, reproducibility spot-check), matching how `001` validated its own baseline.

**Organization**: Tasks are grouped by user story. Unlike `002`/`003`/`005`, this is
**real device work** — it needs an actual build and GPU capture on the `rocky-ryzen`
MiniPC, following `001`'s exact resource-contention discipline.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Paths are relative to the repository root

## Path Conventions

- `backends/vulkan/test/custom_ops/test_llama_baseline_bench.cpp` — modified in place
- `specs/004-linear-storage-comparison/scripts/compare_storage.py` — new analysis script
- `specs/004-linear-storage-comparison/results/raw/storage_bench_raw.log` — captured data
- `specs/004-linear-storage-comparison/results/storage-comparison-report.md` — the deliverable

---

## Phase 1: Setup

- [X] T001 Create `specs/004-linear-storage-comparison/scripts/` and `specs/004-linear-storage-comparison/results/raw/` directories
- [X] T002 Verify nothing else CPU/GPU-heavy is running (`ps aux`/`free -h`) before any capture — same discipline established since `001`'s mid-implementation correction

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The modified, built harness binary every user story's capture depends on

**⚠️ CRITICAL**: No user story capture can begin until this phase is complete

- [X] T003 Add the storage-type axis in `backends/vulkan/test/custom_ops/test_llama_baseline_bench.cpp`: `kStorageTypes = {{"texture3d", utils::kTexture3D}, {"buffer", utils::kBuffer}}`, thread `storage` through `LinearConfig` and `make_case()` (replacing the hardcoded `const utils::StorageType storage = utils::kTexture3D;`), extend `generate_cases()`'s cross-product and the results-printing loop to include it (research.md Decision 1)
- [X] T004 Add a `storage` column to the `RESULT,...` CSV line (right after `<regime>`, per `contracts/storage-comparison-schema.md`) and update the file's header comment to document dual-storage coverage and that `ET_VK_FORCE_TILED_LINEAR=1` is strictly required for every prefill case in this harness (research.md Decision 2, Constitution Principle V) — not just a formality, since this harness's 2D tensors and already-aligned shapes mean `Buffer` storage alone would satisfy every remaining coopmat-eligibility condition (depends on T003)
- [X] T005 Rebuild `test_llama_baseline_bench` (depends on T004) — corrected build dir: `cmake-out-vk/backends/vulkan/test/custom_ops` (`001`'s original separate `custom_ops` sub-project per the constitution's recipe), not `cmake-out-vk-profiling` as originally planned — that dir never had this target configured

**Checkpoint**: Foundation ready — the binary can produce 192 dual-storage cases

---

## Phase 3: User Story 1 - Isolate the storage-type effect on prefill linear performance (Priority: P1) 🎯 MVP

**Goal**: Get a real, uncontaminated Texture3D-vs-Buffer comparison for every prefill case.

**Independent Test**: Capture prefill cases at both storage types, confirm every `buffer`-storage prefill row's kernel is still tiled (never `*_coopmat`), and compute a real-vs-noise verdict per case.

- [X] T006 [US1] Capture: `ET_VK_FORCE_TILED_LINEAR=1 ./cmake-out-vk/backends/vulkan/test/custom_ops/test_llama_baseline_bench > specs/004-linear-storage-comparison/results/raw/storage_bench_raw.log 2>&1`, confirmed exactly 192 `RESULT,` lines, 0 "coopmat" matches (depends on T005, T002)
- [X] T007 [US1] Verify every prefill row's kernel name is tiled-family, never contains `coopmat` (depends on T006) — **real bug found and fixed**: initial verification showed ops sharing an identical (K,N) shape (e.g. `wq`/`wo`, `wk`/`wv`) reporting inconsistent, request-independent kernel-name suffixes. Root cause: `execute_test_cases()` groups cases by a `ReferenceKey` that excludes `storage_type`, reordering `results` relative to `generate_cases()`'s nested-loop order — a positional `results[idx++]` read silently mislabels rows. Fixed by looking up each result by the name `BenchmarkResult` is seeded with (`g_case_configs`, added to the harness) instead of assuming index correspondence. Re-captured after the fix: every op now consistently reports the kernel matching its actually-requested storage, 0 coopmat matches confirmed
- [X] T008 [US1] Implement the core comparison engine in `specs/004-linear-storage-comparison/scripts/compare_storage.py`: parse `RESULT` lines, pair `texture3d`/`buffer` rows per (model, scheme, regime, op), compute `relative_diff_pct`, apply the non-overlapping `mean ± 2·stdev` significance rule (research.md Decision 3) (depends on T007)
- [X] T009 [US1] Run the engine on the prefill subset and manually verify 2-3 cases' significance calls by hand against the raw numbers (depends on T008)

**Checkpoint**: US1 complete — prefill's storage-type effect is measured and verified uncontaminated

---

## Phase 4: User Story 2 - Extend the comparison to decode-regime linear performance (Priority: P2)

**Goal**: Get the same comparison for decode, confirming it's presented purely as a storage-cost measurement (not a coopmat-eligibility claim).

**Independent Test**: Confirm every decode row's kernel is `_coop`-family regardless of storage type, and that the engine produces the same kind of per-case verdict for decode as it did for prefill.

- [X] T010 [US2] Verify every decode row's kernel name is `_coop`-family for both storage types (expected and correct per FR-007 — decode's dispatch never depends on storage; this is confirming the expectation, not searching for a bug) (depends on T006)
- [X] T011 [US2] Confirm `compare_storage.py` (T008's engine) produces correct per-case output for decode rows with no special-casing needed — run and inspect (depends on T008, T010)

**Checkpoint**: US2 complete — both regimes are measured and correctly attributed

---

## Phase 5: User Story 3 - Produce a consolidated storage-type comparison report (Priority: P3)

**Goal**: Turn the per-case measurements into the actionable, easy-to-find verdict.

**Independent Test**: Generate the report and confirm the top-line prefill/decode verdicts are findable without reading the full case table, and are consistent with the underlying data.

- [X] T012 [US3] Implement the Decision 4 cross-check in `compare_storage.py`: compare this feature's own `texture3d` numbers against `001`'s already-published `results/raw/<model>_<scheme>.json` `microbench` entries for the same cases, within the same significance band (depends on T008) — **the cross-check did exactly its job**: it surfaced 37/96 cases diverging from `001`'s published data. Investigated and confirmed (not assumed): these are `001`'s own pre-existing `wq`/`wo` and `wk`/`wv` mislabeling bug (same root cause as T007's finding, latent in `001`'s original single-storage capture too, just invisible there since it never printed two visibly-different storage labels side by side). The report explains this inline rather than presenting unexplained "diverged" entries
- [X] T013 [US3] Implement the report renderer: top-level prefill/decode verdicts, full 96-row case table, "infeasible/contaminated" section (present even if empty), cross-check section — per `contracts/storage-comparison-schema.md` (depends on T008)
- [X] T014 [US3] Run `compare_storage.py` end to end → `specs/004-linear-storage-comparison/results/storage-comparison-report.md` (depends on T009, T011, T012, T013)
- [X] T015 [US3] Self-review the report against SC-001 through SC-004 (depends on T014) — **calibration bug found and fixed**: the initial verdict logic declared a regime "costly" from as few as 2/48 (4%) of cases showing a real effect, mischaracterizing a near-universal null result as a directional finding. Fixed to require a majority of cases before an overall costly/beneficial verdict; otherwise reports "effectively free for the majority, with N isolated exceptions" and names them (e.g. decode: free for 35/48, with `wv` showing a recurring +20-51% cost across 3 models — a real, named exception, not lost in an average)

**Checkpoint**: US3 complete — the consolidated report answers the question that motivated this whole study

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T016 Reproducibility spot-check: re-capture one case (e.g. one prefill op at both storage types) and confirm the kernel names and approximate timings match the original capture, matching `001`'s established reproducibility discipline
- [X] T017 [P] Update `quickstart.md` with any corrections found during T006-T016 (if any were needed)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup. BLOCKS all user stories (the binary must exist and be correct before any capture)
- **User Story 1 (Phase 3)**: Depends on Foundational
- **User Story 2 (Phase 4)**: Depends on Foundational; T010 only needs T006's capture (shares it with US1, doesn't require a separate one)
- **User Story 3 (Phase 5)**: Depends on US1's engine (T008) and needs both US1 (T009) and US2 (T011) done before the final report run (T014)
- **Polish (Phase 6)**: Depends on US3

### Within Each User Story

- US1: capture (T006) → kernel verification (T007) → engine (T008) → manual spot-check (T009), strictly sequential since each depends on the prior step's output
- US3: T012/T013 can be implemented in parallel (different concerns: cross-check vs. rendering) but both depend on T008's engine; T014 needs everything before it

### Parallel Opportunities

- T012 and T013 (US3) can be implemented in parallel — different functions in the same script, no shared state, though both need T008 done first
- T017 (Polish) has no dependency on T016 and could run alongside it

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (modify harness, build)
3. Complete Phase 3: User Story 1 — real, uncontaminated prefill comparison
4. **STOP and VALIDATE**: every prefill `buffer` row's kernel is confirmed tiled, not coopmat

### Incremental Delivery

1. Setup + Foundational → binary ready, 192 cases producible
2. US1 → prefill comparison, verified uncontaminated (the foundational measurement everything else builds on, per spec.md's "Why this priority")
3. US2 → decode comparison, correctly attributed as storage-only (not a coopmat claim)
4. US3 → the actual deliverable: one report with a clear go/no-go verdict

---

## Notes

- No commits until the user explicitly asks, per repo convention.
- T007's kernel-name check is a hard gate, not a nice-to-have: if it fails, the fix is to re-verify the `ET_VK_FORCE_TILED_LINEAR=1` env var actually reached the process (not a build issue), then redo the capture — never proceed with contaminated data.
