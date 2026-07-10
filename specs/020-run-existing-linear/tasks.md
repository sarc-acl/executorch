---

description: "Task list for M5 EVT1 Full Microbenchmark Suite -- Stable Results Report"
---

# Tasks: M5 EVT1 Full Microbenchmark Suite — Stable Results Report

**Input**: Design documents from `specs/020-run-existing-linear/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: Not requested — this is a hardware measurement feature; verification is each harness's own dispatch/correctness output plus the cross-invocation stability check, not a code test suite.

**Organization**: Tasks are grouped by user story (US1=P1 run-with-verified-preconditions, US2=P1 stability check, US3=P2 consolidated report, per spec.md).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to
- All device paths use the env block from `.shared-context/instruction-for-ai/README.md` §Conventions (`$S`/`$D`/`$SC`)
- No source changes to any of the three harnesses or `CMakeLists.txt` — all three are confirmed already built and already registered (research.md Decision 1)

---

## Phase 1: Setup

**Purpose**: Confirm the shared device is usable and every binary this feature needs is current, before touching clock state.

- [X] T001 Confirm M5 EVT1 is free (`adb -s $S get-state`, `adb -s $S shell ps -A | grep -i test_`) and re-verify on-device driver identity (`adb -s $S shell md5sum /vendor/lib64/hw/vulkan.samsung.so`), per constitution Principle VIII. **DONE**: device idle, driver `c9861e9906d0...` -> `f14c51b6f8` (current default, known-good).
- [X] T002 [P] Confirm local binary freshness for all three harnesses: `stat -c "%Y %n"` each of `backends/vulkan/test/custom_ops/test_{coopmat_linear_bench,sdpa_coopmat_bench,llama_baseline_bench}.cpp` against `cmake-out-android-vk/backends/vulkan/test/custom_ops/test_{coopmat_linear_bench,sdpa_coopmat_bench,llama_baseline_bench}`; rebuild any binary whose source mtime is newer (per `.shared-context/instruction-for-ai/build.md`, Principle X) — as of this session's own check all three are current, but re-verify, don't assume it's still true. **DONE**: all 3 binaries newer than their sources, no rebuild needed.
- [X] T003 [P] Create `specs/020-run-existing-linear/results/raw/` directory. **DONE**.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Correct the leftover floating-clock state from the stopped `specs/019` session and confirm every binary is staged on-device — this applies once per session and blocks every measurement.

**⚠️ CRITICAL**: Complete before starting any user story. Do not proceed past T006 until clocks are sysfs-confirmed pinned.

- [X] T004 Re-pin GPU/MIF/INT clocks to 509000/2730000/663000 Hz (the workspace's `pin_freqs.sh` or equivalent), correcting the floating state left over from `specs/019`. **DONE**.
- [X] T005 Verify the pin actually bound via sysfs readback (`/sys/kernel/gpu/{min,max}_freq`, the MIF/INT devfreq nodes) — all six values must equal the pinned triple, not the hardware full range (Principle VII). Stop and re-issue the write if any value doesn't match. **DONE**: 509000/509000/2730000/2730000/663000/663000, all match.
- [X] T006 Check on-device staging state via `adb -s $S shell ls -la $D/ | grep -E "test_coopmat_linear_bench|test_sdpa_coopmat_bench|test_llama_baseline_bench"` (research.md Decision 5): confirm `test_coopmat_linear_bench_016`'s on-device size matches the current local build exactly; push `test_sdpa_coopmat_bench` and `test_llama_baseline_bench` to `$D/test_sdpa_coopmat_bench_020` and `$D/test_llama_baseline_bench_020` respectively (neither is currently staged), `chmod 755` both. **DONE**: linear_bench_016 size-matched (56440208), sdpa+baseline pushed and size-verified (55862256/56648752).

**Checkpoint**: Pinned clock state sysfs-confirmed, all three binaries confirmed staged and current — per-harness measurement can begin.

---

## Phase 3: User Story 1 - Run all three microbenchmarks with verified preconditions (Priority: P1) 🎯 MVP

**Goal**: One verified-precondition invocation of each of the three harnesses, raw output captured — including `test_llama_baseline_bench`'s first-ever M5 EVT1 run.

**Independent Test**: `results/raw/{linear,sdpa,baseline}_rep1.log` all exist, each with exit code 0 and a complete summary table; `baseline_rep1.log` in particular is genuinely new evidence (no prior M5 EVT1 run of this harness exists anywhere in this workstream).

### Implementation for User Story 1

- [X] T007 [US1] Run `test_coopmat_linear_bench_016` rep 1: `adb -s $S shell "cd $D && ./test_coopmat_linear_bench_016" > specs/020-run-existing-linear/results/raw/linear_rep1.log`. **DONE**: exit 0, complete 24-row summary table, all coopmat kernels fired.
- [X] T008 [P] [US1] Run `test_sdpa_coopmat_bench_020` rep 1: `adb -s $S shell "cd $D && ./test_sdpa_coopmat_bench_020" > specs/020-run-existing-linear/results/raw/sdpa_rep1.log`. **DONE**: exit 0, all 3 models `confirmed` dispatch (81.6%/81.8%/75.1% speedup).
- [X] T009 [P] [US1] Run `test_llama_baseline_bench_020` rep 1: `adb -s $S shell "cd $D && ./test_llama_baseline_bench_020" > specs/020-run-existing-linear/results/raw/baseline_rep1.log`. **DONE, WITH A REAL FINDING**: exit 137 (SIGKILL) after 14/192 cases -- kernel OOM-killer confirmed via `dmesg` (`Out of memory: Killed process ... test_llama_base ... anon-rss:6710980kB`). Root cause: `utils.cpp:1704-1705`'s `execute_test_cases()` materializes ALL 192 cases' tensors upfront before executing any; 12 `lm_head` prefill cases each hold a `[2048,128256]` fp16 tensor (~525MB), ~6.3GB total, matching the observed anon-rss almost exactly. This is `test_llama_baseline_bench`'s first-ever M5 EVT1 run -- never previously surfaced. Per user decision (AskUserQuestion), NOT worked around: harness stays untouched (matches this feature's no-source-changes scope), the crash and partial 14-case data are reported as-is in the final report (FR-010).
- [X] T010 [US1] Spot-check all three rep-1 logs: confirm each process exited 0, each harness's own summary table is present and complete (not truncated by a crash), and every case's dispatch-confirmation column (linear's `!`-flag, SDPA's `confirmed`/`NOT CONFIRMED`) and correctness column (PASSED/FAILED/SKIPPED) are populated — not garbage or missing. **DONE**: linear/SDPA clean; baseline_bench anomaly documented in T009, carried into US2/US3 rather than silently dropped.

**Checkpoint**: One real, verified-precondition M5 EVT1 result exists for all three harnesses — `test_llama_baseline_bench`'s data gap is closed. This alone is a valid, demonstrable increment even before US2/US3.

---

## Phase 4: User Story 2 - Confirm results are stable (Priority: P1)

**Goal**: Each harness invoked 3 separate times total; every case's cross-invocation spread computed and any peer-relative outlier flagged.

**Independent Test**: `aggregate_microbench_results.py` runs against all 9 raw logs (3 harnesses × 3 reps) without error, producing a StabilityVerdict (mean, CoV%, outlier flag) for every case defined in data-model.md, with the peer-relative outlier rule (research.md Decision 3 — no fixed cutoff) actually implemented, not a hardcoded threshold.

**Depends on**: User Story 1 (reuses rep 1's raw logs; only reps 2-3 are new here).

### Implementation for User Story 2

- [X] T011 [US2] Run `test_coopmat_linear_bench_016` reps 2 and 3, same command as T007 with `_rep2`/`_rep3` output filenames. **DONE**.
- [X] T012 [P] [US2] Run `test_sdpa_coopmat_bench_020` reps 2 and 3, same command as T008 with `_rep2`/`_rep3` output filenames. **DONE**.
- [X] T013 [P] [US2] Run `test_llama_baseline_bench_020` reps 2 and 3, same command as T009 with `_rep2`/`_rep3` output filenames. **DONE**: both reps also OOM-killed at the same 14/192 cases -- deterministic, confirms rep1 wasn't a fluke.
- [X] T014 [US2] Write `.shared-context/scripts/aggregate_microbench_results.py`: parsers for the linear/baseline harnesses' `SUMMARY:` table lines and the SDPA harness's `RESULT,...` CSV lines, keyed per data-model.md's per-harness `case_key` definitions; each CaseResult retains `tiled_value`/`tiled_stddev`/`coopmat_value`/`coopmat_stddev`/`dispatch_confirmed`/`correctness_status` from the harness's own output (FR-007) — never a bare mean. **DONE**. Note: baseline never reaches its `RESULT,...` CSV block (OOM before `execute_test_cases()` returns), so its parser reads the raw per-case dispatch line directly instead -- documented in the script's own comments.
- [X] T015 [US2] In the same script, implement the StabilityVerdict computation: group CaseResults by `(harness, model, case_key)` across their 3 reps, compute `mean`/`cov_pct`, and implement the peer-relative outlier rule from research.md Decision 3 (flag a case only when its `cov_pct` is a clear outlier relative to the other cases' `cov_pct` within the same harness+scheme grouping — no fixed numeric cutoff). **DONE**: >3x group-median-CoV rule, applied to the worse of tiled/coopmat CoV per case.
- [X] T016 [US2] Run the script against all 9 raw logs; confirm every case defined in data-model.md produces exactly one StabilityVerdict (SC-001), and that `dispatch_confirmed=false` or `correctness_status=FAILED` cases are marked distinctly, never folded into a normal-looking average (FR-003/FR-010). **DONE**: 24 linear + 3 SDPA + 14 baseline verdicts, all confirmed/flagged correctly (caught and fixed a real regex bug during this step -- op names with embedded underscores like `w1_gate` were mis-parsed until the model/scheme/regime/storage fields were tightened to `[^_]+`).

**Checkpoint**: Every reportable case across all three harnesses now has real cross-invocation stability evidence, not a single-sample number.

---

## Phase 5: User Story 3 - Produce one consolidated, plain-language report (Priority: P2)

**Goal**: One report file covering all three harnesses, readable without consulting raw logs or GLSL source, reconciled against `specs/016`'s prior linear/SDPA numbers.

**Independent Test**: `specs/020-run-existing-linear/results/microbenchmark-suite-report.md` exists with one section per harness (per-model/per-scheme tables + plain-language summary), an explicit reconciliation-vs-`specs/016` statement, and every anomaly from US2 named in the body.

**Depends on**: User Story 2 (needs StabilityVerdicts to render).

### Implementation for User Story 3

- [X] T017 [US3] Extend `aggregate_microbench_results.py` with a `--out` report-rendering step: one Markdown section per harness (linear, SDPA, baseline), each with a per-model/per-scheme table showing tiled/coopmat values, CoV%, dispatch/correctness status, and a one-sentence plain-language summary of the win/loss direction and magnitude. **DONE**.
- [X] T018 [US3] Add a `--compare-against specs/016-m5-linear-sdpa-microbench/results/` option: for the linear and SDPA sections only, state explicitly whether this feature's new numbers are consistent with `specs/016`'s prior single-invocation numbers (same order of magnitude, same win/loss direction), and by how much they differ if at all (FR-009). `test_llama_baseline_bench` has no prior M5 number to reconcile against — state that explicitly rather than fabricating a comparison. **DONE**: all 24 linear + 3 SDPA deltas within 1.5pp of specs/016, all marked YES-consistent; baseline explicitly states it has no prior number.
- [X] T019 [US3] Add an explicit anomaly section to the rendered report: every StabilityVerdict flagged as an outlier in US2, every `dispatch_confirmed=false` coopmat-eligible case, every `correctness_status=FAILED`, and any non-zero exit code from US1/US2's invocations — named by harness/model/case, not just a count (FR-010). **DONE**: baseline OOM + 6 named linear outliers (notably, all 6 are `8da4w` cases -- a real pattern surfaced by this check, not just noise).
- [X] T020 [US3] Generate the final report: `.../aggregate_microbench_results.py ... --out specs/020-run-existing-linear/results/microbenchmark-suite-report.md`; read it back and confirm it satisfies SC-002/SC-003/SC-004 (every coopmat claim backed by `dispatch_confirmed`, readable without GLSL knowledge, zero anomalies silently dropped). **DONE**: `specs/020-run-existing-linear/results/microbenchmark-suite-report.md` generated and read back; SC-002 (24/24 + 3/3 confirmed dispatch stated explicitly), SC-003 (plain-language summaries per section), SC-004 (anomaly section names every outlier + the OOM) all satisfied.

**Checkpoint**: The feature's actual deliverable exists and is internally self-consistent with US1/US2's raw evidence.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Leave the device and the workspace's tooling registry in a clean state.

- [X] T021 Re-verify clocks are still pinned (509/2730/663 MHz, sysfs readback) after the full run — 9 invocations across 3 harnesses is enough wall-clock time that a state change (reboot, another session) should be re-checked, not assumed unchanged from T005. **DONE**: all 6 values still 509000/2730000/663000.
- [X] T022 [P] Add `aggregate_microbench_results.py` to `.shared-context/scripts/README.md`'s catalog table, following the same convention as `analyze_etdump_shaders.py`/`run_m5_full_sweep.py`'s existing entries. **DONE**.
- [X] T023 Run `quickstart.md`'s four steps end-to-end as a final check; confirm the report exists, every section is populated, and no step required deviating from what quickstart.md actually documents (if it did, quickstart.md itself needs fixing, not silently worked around). **DONE**: all 4 steps were actually followed during T001-T020's real execution (stage/verify binaries, verify pinned clocks, 3x run each harness, aggregate+report); no deviation from what quickstart.md documents -- it needs no fix.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup — blocks all user stories. Do not proceed past T006 until clocks are sysfs-confirmed pinned and all three binaries are confirmed staged.
- **User Story 1 (Phase 3)**: Depends on Foundational only.
- **User Story 2 (Phase 4)**: Depends on User Story 1 (reuses its rep-1 raw logs) — not purely independent of US1, but still independently testable/demonstrable as its own increment (the stability evidence) once US1's data exists.
- **User Story 3 (Phase 5)**: Depends on User Story 2 (needs StabilityVerdicts).
- **Polish (Phase 6)**: Depends on User Story 3.

### Parallel Opportunities

- T002 and T003 (Setup) can run in parallel — different concerns, no file overlap.
- Within Phase 3/4, the linear/SDPA/baseline invocation tasks (T007-T009, T011-T013) are marked `[P]` where they write to different files with no cross-dependency — but all three ultimately contend for the same single physical M5 EVT1 device, so in practice they run sequentially on real hardware even though nothing in their task definition forces that ordering.
- T021 and T022 (Polish) can run in parallel.

---

## Parallel Example: User Story 1

```bash
# Launch all three rep-1 invocations for User Story 1 (logically independent, contend for the same device in practice):
Task: "Run test_coopmat_linear_bench_016 rep 1 -> results/raw/linear_rep1.log"
Task: "Run test_sdpa_coopmat_bench_020 rep 1 -> results/raw/sdpa_rep1.log"
Task: "Run test_llama_baseline_bench_020 rep 1 -> results/raw/baseline_rep1.log"
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Complete Phase 1 (Setup) and Phase 2 (Foundational) — including the pinned-clock sysfs confirmation and on-device staging check.
2. Complete Phase 3 (US1 — one verified invocation of each harness).
3. **STOP and VALIDATE**: confirm all three rep-1 logs are complete and well-formed, especially `baseline_rep1.log` — this alone closes this workstream's "baseline bench never run on M5" gap.
4. This proves every precondition (driver, clocks, staging) is correctly handled before committing to 2 more rounds of device time.

### Incremental Delivery

1. Setup + Foundational → device ready, all three binaries staged, clocks pinned.
2. Add US1 (one verified run each) → validate → first-ever M5 baseline-bench data in hand, fresh linear/SDPA data in hand.
3. Add US2 (2 more reps each + aggregation/stability) → validate → real stability evidence exists for the first time at the microbenchmark tier.
4. Add US3 (consolidated report) → validate → the actual deliverable (one readable report, reconciled against `specs/016`) is done.
5. Polish (re-verify clocks, register the new script, final quickstart validation) → feature done, workspace tooling stays discoverable.
