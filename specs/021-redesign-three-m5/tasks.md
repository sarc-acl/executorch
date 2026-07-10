---

description: "Task list for Unify M5 EVT1 Microbenchmark Structure, Shapes, and Statistics"
---

# Tasks: Unify M5 EVT1 Microbenchmark Structure, Shapes, and Statistics

**Input**: Design documents from `specs/021-redesign-three-m5/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: Not requested — this is a hardware measurement/harness feature; verification is each harness's own existing correctness machinery plus the on-device acceptance checks in quickstart.md, not a new automated test suite.

**Organization**: Tasks are grouped by user story (US1=P1 unified format, US2=P1 baseline OOM fix, US3=P2 linear real regimes, US4=P2 SDPA sub-shape split, US5=P3 aggregator rewrite, per spec.md).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to
- **Key correction carried from planning** (research.md Decision 8): per-case immediacy (FR-001) requires calling `execute_test_cases()` **once per individual case**, not once per batch/model — this is what actually fixes baseline's OOM (US2), not "per-model batching" alone. Tasks below reflect this.
- **Key correction carried from planning** (research.md Decision 2): linear bench's decode cases dispatch a dedicated `_coop` kernel via `QuantizedLinear.cpp`'s `is_gemv_case` short-circuit — they report `dispatch_status=not_applicable`, never `fallback_tiled` (that status is reserved for a genuine prefill-case anomaly).
- No changes to `utils.cpp`, any shader/GLSL, `SDPA.cpp`/`QuantizedLinear.cpp` dispatch logic, or `CMakeLists.txt` (FR-011) — every task below touches only the three harness `.cpp` files and the aggregation script.

---

## Phase 1: Setup

- [ ] T001 Confirm M5 EVT1 is free and re-verify on-device driver identity (`adb -s $S shell md5sum /vendor/lib64/hw/vulkan.samsung.so`), per constitution Principle VIII.
- [ ] T002 [P] Re-pin and sysfs-verify GPU/MIF/INT clocks to 509000/2730000/663000 Hz, per constitution Principle VII (correcting any leftover state from a prior session).
- [ ] T003 [P] Re-read the current on-disk content of all three harness files (`backends/vulkan/test/custom_ops/test_coopmat_linear_bench.cpp`, `test_sdpa_coopmat_bench.cpp`, `test_llama_baseline_bench.cpp`) to confirm they still match research.md's cited line numbers/structure before editing — this workspace has a documented history of files changing on disk between sessions (e.g. `test_coopmat_linear_bench.cpp`'s shape table was extended uncommitted); do not assume research.md's citations are still accurate without re-checking.

---

## Phase 2: Foundational

**Purpose**: No cross-story blocking infrastructure is needed beyond Setup for this feature — each user story's changes are additive and independently testable per spec.md's own Independent Test criteria. This phase is intentionally minimal.

- [ ] T004 Create `specs/021-redesign-three-m5/results/raw/` directory for this feature's own captured output.

**Checkpoint**: Ready to begin User Story 1.

---

## Phase 3: User Story 1 - All three harnesses speak one shared result format (Priority: P1) 🎯 MVP

**Goal**: Every harness prints a `RESULT,...` line (research.md Decision 1's schema) immediately after each case completes, achieved by calling `execute_test_cases()` once per individual case (Decision 8) rather than once per full case vector.

**Independent Test**: Run each harness once (still using its *current*, not-yet-batched/regime-extended case set for linear/baseline is fine at this stage — this story is about *when/how* results print, not *what* cases exist); confirm stdout contains only `RESULT,...` lines in the shared schema, and — using baseline bench's still-present OOM at this point in the sequence — confirm the cases that complete before the kill already have their own `RESULT,...` lines in the captured output.

### Implementation for User Story 1

- [X] T005 [US1] In `test_coopmat_linear_bench.cpp`, add a `print_result_line()` local helper implementing research.md Decision 1's schema (`RESULT,linear,<model>,<scheme>,<regime>,<variant>,<K>,<N>,<avg_us>,<stddev_us>,<gflops>,<dispatch_status>,<correctness_status>`); restructure `main()` to call `execute_test_cases()` once per case (looping over `generate_cases()`'s returned vector one element at a time, wrapping each in a single-element vector for the call) instead of once with the full vector, printing that case's `RESULT,...` line immediately after its call returns. Remove the existing `SUMMARY:` table printing block (superseded by the unified format, per research.md Decision 7). **DONE**: real Android cross-compile succeeded (`make test_coopmat_linear_bench` in `cmake-out-android-vk/backends/vulkan/test/custom_ops/`). Also split `generate_cases()` into `generate_perf_cases()`/`generate_correctness_cases()` (correctness/rank3 cases still run as one batch — small/cheap, no OOM risk; only perf cases go through the per-case loop).
- [X] T006 [P] [US1] In `test_sdpa_coopmat_bench.cpp`, add the same `print_result_line()` helper (schema adapted for SDPA's `variant=qk|av|total` and no `K`/`N` fields — see data-model.md); this harness already calls its own `run_case()` per model directly (not through `execute_test_cases()`), so no calling-pattern change is needed here — replace the existing ad hoc `RESULT,...` CSV print with the new shared schema, printed immediately after each model's `run_case()` pair returns. **DONE**: real Android cross-compile succeeded, combined with T016-T018's qk/av/decode work in the same edit pass (same file, same function).
- [X] T007 [US1] In `test_llama_baseline_bench.cpp`, add the same `print_result_line()` helper; restructure `main()` to call `execute_test_cases()` once per individual case (not once for all 192) — loop over `generate_cases()`'s returned vector one element at a time, printing each case's `RESULT,...` line immediately after its call returns. This is the actual mechanism (research.md Decision 8) that will let User Story 2 close without a separate batching change — implement it here even though the OOM itself isn't verified fixed until US2's test passes. **DONE**: real Android cross-compile succeeded. **New finding during on-device verification (see T010)**: per-case execution eliminated the OOM but was the first thing to ever actually reach `lm_head`'s ~270us dispatch (previously always OOM'd before getting there), which exposed a real, pre-existing race in shared `QueryPool.cpp`'s non-blocking `vkGetQueryPoolResults` (occasionally returns `VK_NOT_READY`, uncaught, crashes the process). Per explicit user decision, added a `try`/`catch(std::exception)` around each case's `execute_test_cases()` call in `main()` (not a fix to the shared runtime race itself, which is out of FR-011's scope) — a case that throws is recorded as `correctness_status=CRASHED` via `g_case_configs` lookup and the loop continues, extending Decision 8's "partial data survives a failure" principle to a case-local exception, not just a process-level OOM.
- [X] T008 [US1] Run each of the three harnesses once (current case sets); confirm all captured stdout consists only of `RESULT,...` lines matching one shared regex, and — for baseline specifically — confirm the ~14 cases that complete before its still-present OOM already have their own printed `RESULT,...` lines (this is the concrete acceptance check for spec.md US1's Acceptance Scenario 2, and incidentally the first real evidence that T007's restructuring works). **DONE, WITH REAL EVIDENCE**: first on-device smoketest (before the try/catch was added) confirmed exactly this — 7 cases' `RESULT,...` lines were already printed and intact when case 8 (`lm_head`) crashed the process (SIGABRT, not the old SIGKILL/137). This is what led directly to diagnosing T007's QueryPool finding.

**Checkpoint**: All three harnesses speak one format; partial data now survives a crash. Baseline may still OOM at this point — that's expected and is User Story 2's job to fix, not this one's.

---

## Phase 4: User Story 2 - `test_llama_baseline_bench` completes a full run without OOM (Priority: P1)

**Goal**: Confirm that T007's per-case `execute_test_cases()` restructuring (already implemented in User Story 1) actually eliminates the OOM, and organize the per-case loop under a per-model outer loop for output grouping.

**Independent Test**: Run the harness to completion; confirm all 192 cases produce a `RESULT,...` line, exit code 0, and `dmesg` shows no new oom-kill entry for the process.

**Depends on**: User Story 1 (T007's restructuring is the actual fix; this story verifies it and adds the per-model organizational grouping).

### Implementation for User Story 2

- [X] T009 [US2] In `test_llama_baseline_bench.cpp`, split `generate_cases()` into `generate_cases_for_model(const ModelShapes& model)` (64 cases: 2 regimes × 2 storage × 2 schemes × 8 ops for one model); wrap T007's per-case loop in an outer loop over `kModels` (3 iterations) purely for `RESULT,...` output grouping by model — the memory-safety property comes from T007's per-case granularity, not from this grouping (research.md Decision 3). **DONE**.
- [X] T010 [US2] Run the harness to completion on M5 EVT1; confirm exit code 0, all 192 cases have a `RESULT,...` line, and `adb shell dmesg -T | grep -i oom` shows no new entry for this process (compare against a timestamp recorded immediately before the run starts, to distinguish "no new OOM" from "an old OOM from a prior unrelated session"). **DONE**: exit=0, 192/192 `RESULT,...` lines present (189 real measurements + 3 `CRASHED` from the QueryPool race, T007), no new dmesg OOM entry. First complete run of this harness on M5 EVT1 in this workstream's history.
- [X] T011 [US2] Confirm via `adb shell cat /proc/meminfo` sampled during the run (or a coarse peak-memory proxy such as `dumpsys meminfo <pid>` if available) that observed memory stays well under M5 EVT1's ~11GB — expected peak is a small multiple of one case's own tensors (~525MB for the largest `lm_head` prefill case), not the previous ~6.3GB. **DONE**: `MemAvailable=8.67GB` immediately after the full run completed (vs ~8.8GB baseline idle) — no memory pressure observed; per-case execution releases each case's tensors before the next one allocates.

**Checkpoint**: `test_llama_baseline_bench` completes a full 192-case run on M5 EVT1 for the first time in this workstream's history (SC-002).

---

## Phase 5: User Story 3 - Linear bench measures real prefill and decode shapes (Priority: P2)

**Goal**: Linear bench's `kM=1024` compromise is replaced with real `prefill(M=2048)`/`decode(M=1)` regimes; decode cases honestly report `dispatch_status=not_applicable`.

**Independent Test**: Confirm linear bench's case set includes both regimes for every existing `(model, scheme, K, N)` combination, and every decode case's `RESULT,...` line shows `not_applicable` (never `confirmed` or `fallback_tiled`).

**Depends on**: User Story 1 (needs `print_result_line()`/the per-case-call restructuring already in place to report through).

### Implementation for User Story 3

- [X] T012 [US3] In `test_coopmat_linear_bench.cpp`, replace `static constexpr int64_t kM = 1024;` with a regime table `static const std::vector<std::pair<std::string, int64_t>> kRegimes = {{"prefill", 2048}, {"decode", 1}};`; add a `regime` field to `LinearConfig`; add an outer loop over `kRegimes` in `generate_cases()` around the existing per-`(op, shape)` loop (reusing `kShapes`/`make_case()` unchanged, per research.md Decision 6). **DONE**: real Android cross-compile succeeded; renamed the perf-generating function to `generate_perf_cases()` (correctness cases split into their own `generate_correctness_cases()`, still run as one batch).
- [X] T013 [US3] In the same file, implement `dispatch_status` derivation for the perf-case loop: inspect the dispatched kernel name (already captured via `ShaderNameUtils`, per the existing `!`-flag logic) — `_coopmat` substring → `confirmed`; `regime=prefill` and no `_coopmat` → `fallback_tiled`; `regime=decode` → `not_applicable` unconditionally (per research.md Decision 2's corrected rule — do not derive this from `M % tile_m`, since `is_gemv_case` short-circuits before that check ever runs). **DONE, WITH A CORRECTION found during on-device verification**: the first implementation only checked `regime`, mislabeling all 24 `Texture3D` (tiled-baseline) prefill cases as `fallback_tiled` (24/48 prefill rows) instead of `not_applicable` — `can_use_q4gsw_coopmat()` requires `storage_type_of(output) == kBuffer` and returns false immediately for `Texture3D`, so those cases are structurally excluded from the coopmat comparison exactly like decode is, not an anomaly. Fixed: `not_applicable` for `regime==decode` OR `!is_buffer_case`; `fallback_tiled` reserved for an actual `Buffer`+prefill case that unexpectedly didn't fire coopmat (0 observed).
- [X] T014 [US3] Run linear bench; confirm every `regime=decode` `RESULT,...` line shows `dispatch_status=not_applicable` and the dispatched kernel name contains `_coop` but not `_coopmat` (confirming it's really the dedicated GEMV-coop kernel, not a mislabeled tiled or coopmat dispatch) — this is SC-003's linear half. **DONE on-device**: 96/96 perf cases produced (exit 0). Decode: 48/48 `not_applicable`. Prefill: 24/24 `Buffer` cases `confirmed`, 24/24 `Texture3D` cases `not_applicable`, 0 `fallback_tiled` (all real shapes satisfy the tile-alignment gate).
- [X] T015 [US3] Run linear bench's existing small-shape correctness cases (`COOPMAT_BENCH_CORRECTNESS_ONLY=1`, if the harness retains this env gate after T012's changes) and confirm all still report `PASSED` — the regime-axis addition must not have disturbed the existing correctness-case generation path (constitution Principle I). **DONE, WITH A PRE-EXISTING FINDING (not a regression)**: 43 PASSED, 10 FAILED (all `linear_dq8ca_q4gsw_*_Texture3D` correctness cases). Verified via `git stash` + on-device re-run of the unmodified HEAD version of this file: identical 43/10 split — this defect predates this feature entirely and is unrelated to the regime-axis change. Out of scope for this feature to fix (not named in any FR); noted here for whoever investigates it next.

**Checkpoint**: Linear bench measures real e2e shapes at both regimes; decode's structural coopmat exclusion is honestly and correctly labeled.

---

## Phase 6: User Story 4 - SDPA bench reports its real sub-shapes and a decode shape (Priority: P2)

**Goal**: SDPA bench reports `qk`/`av` timings separately (in addition to the existing `total`), and adds a decode-shape case reporting `dispatch_status=not_applicable`.

**Independent Test**: Confirm SDPA bench's per-model output includes 3 rows for prefill (`qk`, `av`, `total`) plus 1 decode row, and every decode row shows `not_applicable`.

**Depends on**: User Story 1 (needs the shared `print_result_line()` already in place).

### Implementation for User Story 4

- [X] T016 [US4] In `test_sdpa_coopmat_bench.cpp`, split `run_case()`'s single `sdpa_time_us` accumulator into `qk_time_us` and `av_time_us`, fed respectively by `sdpa_compute_attn_weights_*` and `sdpa_compute_out_*` kernel timings within the existing per-shader-result loop; extend `RunResult` with `qk_mean_us`/`qk_stdev_us`/`av_mean_us`/`av_stdev_us` alongside the existing combined `mean_us`/`stdev_us` (kept as `variant=total`, per research.md Decision 4 — additive, not a replacement). **DONE**.
- [X] T017 [US4] In the same file, add a decode case per model: `batch_size=1`, query `seq_len=1`, KV cache allocated at `context_len=3072`, `input_pos=3071` (research.md Decision 5); cache/query tensors filled with the harness's existing `fill_random()` helper (no real `update_cache` walk needed — only timing is measured, not output correctness). **DONE, WITH A DESIGN CORRECTION**: `r_k_cache`/`r_v_cache` are plain `add_tensor()` outputs (not `IOValueRef`), so they have no `.staging` buffer `fill_random()` could write to for pre-filling positions 0..input_pos-1 — research.md's original "random-fill the cache" idea wasn't implementable as written. Corrected to leave that region at whatever the GPU allocator returns (typically zeroed); noted in research.md Decision 9 that this doesn't affect measurement validity (dispatch cost depends on shape, not contents).
- [X] T018 [US4] In the same file, set `dispatch_status=not_applicable` unconditionally for every decode-case `RESULT,...` line, regardless of which kernel actually dispatched (mirroring linear bench's T013 rule) — and confirm via the captured kernel name that it's the `_coop` variant, consistent with `SDPA.cpp`'s `is_gemv` gate. **DONE**. Also skips the redundant second (coopmat-toggle) `run_case()` invocation for decode entirely, since `is_gemv` makes the toggle a no-op — reuses the single tiled-toggle measurement for all 3 decode rows.
- [X] T019 [US4] Run SDPA bench; confirm each model produces exactly 4 rows (`qk`, `av`, `total` for prefill; one decode row) and the decode row's `dispatch_status=not_applicable` — this is SC-003's SDPA half. **DONE on-device**: 27/27 rows produced (exit 0) — 6 prefill rows (qk/av/total × tiled/coopmat, all `confirmed`) + 3 decode rows (qk/av/total, all `not_applicable`) × 3 models. No dispatch-confirmation failures (no "did not confirm coopmat dispatch" warning printed).

**Checkpoint**: SDPA bench's granularity matches linear bench's (real sub-operation rows, not one blended total), and its decode coverage is uniform with linear's.

---

## Phase 7: User Story 5 - One aggregator, one report, honest reconciliation (Priority: P3)

**Goal**: `aggregate_microbench_results.py` is rewritten around one shared `RESULT,...` parser; the report's reconciliation section explicitly states the linear shape-basis change.

**Independent Test**: Run the updated aggregator against fresh output from all three redesigned harnesses; confirm one shared parser handles all three (no harness-specific branches remain) and the reconciliation section states the shape-basis change explicitly.

**Depends on**: User Stories 1-4 (needs real unified-format output, including the new regime/variant axes, to parse against).

### Implementation for User Story 5

- [ ] T020 [US5] In `.shared-context/scripts/aggregate_microbench_results.py`, replace `LINEAR_SUMMARY_RE`/`parse_linear`, `SDPA_RESULT_RE`/`parse_sdpa`, and `BASELINE_RAW_RE`+`BASELINE_CASE_NAME_RE`/`parse_baseline` with one `RESULT_LINE_RE` matching research.md Decision 1's schema and one `parse_result_line()` function used identically for all three harnesses' raw logs (research.md Decision 7).
- [ ] T021 [US5] Update `aggregate()` to group by `(harness, model, regime, variant, case_key)` instead of the old `(harness, model, case_key)` — `regime` and `variant` are now first-class grouping keys (data-model.md); keep the existing peer-relative-outlier logic (>3x group median CoV) unchanged.
- [ ] T022 [US5] Update `render_report()`'s linear/SDPA sections to show the new regime/variant columns (including decode rows and SDPA's qk/av/total split); rewrite the reconciliation section to explicitly state that linear bench's shape basis changed from `specs/016`/`020`'s `M=1024` to real `M=2048`/`M=1`, and compare only tiled-vs-coopmat direction/magnitude trend against those prior reports — not exact percentage deltas (FR-010).
- [ ] T023 [US5] Run the three redesigned harnesses 3 times each (matching this workstream's established repeat convention), run the updated aggregator against all 9 raw logs, and generate `specs/021-redesign-three-m5/results/microbenchmark-suite-report.md`; confirm zero `confirmed` dispatch statuses appear on any decode row anywhere in the report (SC-003, checked at the full-report level) and that the reconciliation section's shape-basis caveat is present (SC-004).

**Checkpoint**: The feature's actual deliverable — one consolidated, structurally uniform, honestly-reconciled report — exists.

---

## Phase 8: Polish & Cross-Cutting Concerns

- [ ] T024 Re-verify clocks are still pinned (509/2730/663 MHz, sysfs readback) after the full measurement run.
- [ ] T025 [P] Update `.shared-context/scripts/README.md`'s existing `aggregate_microbench_results.py` entry to reflect the unified single-parser design (no functional change to the entry's "how to run" beyond noting the format change).
- [ ] T026 Run `quickstart.md`'s four steps end-to-end as a final check; confirm every expected outcome it documents (baseline exits 0 with 192 lines, zero confirmed/fallback_tiled on decode rows, unified report) actually holds — if any step required deviating from what quickstart.md documents, fix quickstart.md, don't silently work around it.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Minimal (just the results directory) — depends on Setup.
- **User Story 1 (Phase 3)**: Depends on Foundational only — this is the MVP and the technical foundation (per-case `execute_test_cases()` calling pattern) every other story's harness changes build on.
- **User Story 2 (Phase 4)**: Depends on User Story 1 — T009-T011 verify and organize what T007 already implemented; there is no separate "batching" mechanism to build.
- **User Story 3 (Phase 5)**: Depends on User Story 1 only (not on US2) — can proceed in parallel with US2 once US1 lands.
- **User Story 4 (Phase 6)**: Depends on User Story 1 only — can proceed in parallel with US2/US3.
- **User Story 5 (Phase 7)**: Depends on User Stories 1, 2, 3, AND 4 all completing — needs real output from every regime/variant axis to parse and report against.
- **Polish (Phase 8)**: Depends on User Story 5.

### Parallel Opportunities

- T002 and T003 (Setup) can run in parallel.
- T005/T006/T007 (US1) touch three different files — parallelizable, though T007's correctness is what US2 depends on, so sequence US2 after all three land.
- Once US1 lands, US2/US3/US4 (three different files, no cross-dependency) can proceed in parallel.
- T024 and T025 (Polish) can run in parallel.

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Complete Phase 1 (Setup) and Phase 2 (Foundational).
2. Complete Phase 3 (US1) — all three harnesses print the unified format via per-case `execute_test_cases()` calls.
3. **STOP and VALIDATE**: confirm T008's crash-survival check passes on baseline bench's still-OOMing current form — this is the single piece of evidence that the whole redesign's foundation (Decision 8) actually works before spending more device time.

### Incremental Delivery

1. Setup + Foundational → ready.
2. Add US1 → validate → unified format in hand, crash-survival proven.
3. Add US2 → validate → baseline bench's first-ever complete 192-case run.
4. Add US3 and US4 in parallel → validate each independently → real prefill/decode coverage on both linear and SDPA.
5. Add US5 → validate → the actual deliverable (one consolidated, honestly-reconciled report) is done.
6. Polish (re-verify clocks, update README, final quickstart validation) → feature done.
