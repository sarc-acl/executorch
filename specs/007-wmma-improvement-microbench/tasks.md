---

description: "Task list for WMMA Coopmat Improvement Microbenchmark"
---

# Tasks: WMMA Coopmat Improvement Microbenchmark

**Input**: Design documents from `/specs/007-wmma-improvement-microbench/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md (all present)

**Tests**: Not requested as a separate automated suite — this feature's
correctness signal (kernel-dispatch check + SPIR-V instruction-presence
check + citing existing generic-shape correctness tests, per Clarification
Q1) IS the verification, matching how `001`/`004`/`006` validated their own
work inline rather than via a separate test phase.

**Organization**: Tasks are grouped by user story. This is **real device
work** (capture on `rocky-ryzen`), like `001`/`004`/`006`, but unlike any
prior feature in this workstream, it requires **zero production code
changes** — every mechanism (the harness, the env-var toggle, the
Buffer-storage tiled baseline) already exists from prior features.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files/resources, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Paths are relative to the repository root

## Path Conventions

- `backends/vulkan/test/custom_ops/test_llama_baseline_bench.cpp` — existing harness, run twice, **not modified**
- `cmake-out-vk/vulkan_compute_shaders/*coopmat*_buffer_buffer_half.spv` — existing compiled SPIR-V, inspected read-only
- `specs/007-wmma-improvement-microbench/scripts/compare_wmma.py` — new analysis script
- `specs/007-wmma-improvement-microbench/results/` — new capture log, SPIR-V dumps, and the report

---

## Phase 1: Setup

- [X] T001 Create `specs/007-wmma-improvement-microbench/scripts/`, `specs/007-wmma-improvement-microbench/results/raw/`, and `specs/007-wmma-improvement-microbench/results/spirv/` directories

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Confirm every mechanism this feature reuses is actually present and current — there is no code to write, only readiness to verify

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T002 [P] Confirm `cmake-out-vk/backends/vulkan/test/custom_ops/test_llama_baseline_bench` exists and is up to date with the committed source (rebuild via the `custom_ops` sub-build per the constitution's Reference Build Recipe if stale) — this is the binary this feature runs twice, unmodified — **rebuilt**: binary was stale (source mtime newer, from the earlier clang-format-only fix), rebuilt clean via `cmake --build cmake-out-vk/backends/vulkan/test/custom_ops --target test_llama_baseline_bench`
- [X] T003 [P] Confirm `spirv-dis` is on `PATH` (or note its full path) and confirm both `cmake-out-vk/vulkan_compute_shaders/linear_q4gsw_coopmat_buffer_buffer_half.spv` and `.../linear_dq8ca_q4gsw_coopmat_buffer_buffer_half.spv` exist (rebuild if missing) — confirmed at `~/vulkansdk/1.4.341.1/x86_64/bin/spirv-dis`; both `.spv` files present
- [X] T004 [P] Confirm `specs/004-linear-storage-comparison/results/raw/storage_bench_raw.log` and `specs/003-wmma-shader-candidates/results/classifications/*.json` are present and readable — these are read-only inputs this feature never re-captures — confirmed, all 6 classification files + raw log present

**Checkpoint**: Foundation ready — the WMMA-dispatch capture path, the SPIR-V inspection path, and both upstream datasets are all confirmed present

---

## Phase 3: User Story 1 - Prove the comparison on one configuration (Priority: P1) 🎯 MVP

**Goal**: Prove the WMMA-vs-tiled comparison mechanism works end-to-end,
with full dispatch + SPIR-V + correctness evidence, on one representative
configuration before spending device time on the full matrix.

**Independent Test**: Measure `llama-3.2-1b`/`4w`/`wq`'s WMMA dispatch time,
confirm it actually dispatched the coopmat kernel (not a tiled fallback),
confirm that kernel's compiled SPIR-V contains genuine cooperative-matrix
instructions, and confirm the reported speedup number is directly
traceable to that one row plus `004`'s matching tiled-baseline row.

- [X] T005 [US1] Capture one configuration: run `test_llama_baseline_bench` WITHOUT `ET_VK_FORCE_TILED_LINEAR` set, save full output, extract the `llama-3.2-1b`/`4w`/`wq`/`prefill`/`buffer` `RESULT` row (depends on T002) — first capture found a wiring gap (kernel=`q4gsw_linear_gemm__tin__w_4x8_nc_buffer_half`, tiled); **re-captured after Decision 8's fix**: kernel=`linear_q4gsw_coopmat_buffer_texture2d_half` (genuine coopmat)
- [X] T006 [US1] Verify FR-004's dispatch check for that one row: confirm its `kernel` field contains `coopmat`, not a tiled/coop-fallback name (depends on T005) — first pass caught a real dispatch-wiring gap (FR-004 doing exactly its job, see research.md Decision 8); **after the fix, re-verified: confirmed**, kernel contains `coopmat`
- [X] T007 [P] [US1] Run `spirv-dis` against `linear_q4gsw_coopmat_buffer_buffer_half.spv` → `specs/007-wmma-improvement-microbench/results/spirv/linear_q4gsw_coopmat_buffer_buffer_half.dis.txt`; confirm `OpCooperativeMatrixLoadKHR`/`OpCooperativeMatrixMulAddKHR` are present (research.md Decision 4) (depends on T003; independent of T005) — confirmed present (12x Load, 16x MulAdd, 8x Store); **also re-verified against the actually-dispatched `..._buffer_texture2d_half` variant** (12x Load, 16x MulAdd) after Decision 8's fix made it reachable
- [X] T008 [P] [US1] Confirm `test_coopmat_linear_bench.cpp`'s existing `kCorrectnessShapes` covers the `linear_q4gsw` kernel family (research.md Decision 7) — cite the specific test loop/shapes, don't just assume (independent of T005) — first pass found this coverage was hollow (same root cause as T006, see research.md Decision 8); **after the fix, re-ran `test_coopmat_linear_bench` and `test_q4gsw_linear` directly**: 72/72 + all `linear_q4gsw` cases PASSED against the real coopmat kernel, with a genuine ~5x speedup visible in `test_coopmat_linear_bench`'s own summary table
- [X] T009 [US1] Compute this one row's speedup % against `004`'s matching Buffer-storage tiled-baseline row for `llama-3.2-1b`/`4w`/`wq`/prefill; confirm the number is directly traceable to the two measured times (depends on T004, T005, T006) — **unblocked after Decision 8's wiring fix + re-capture**: tiled=4929.136us (004), WMMA=1937.958us (re-capture, kernel=`linear_q4gsw_coopmat_buffer_texture2d_half`), speedup = **+60.7%**

**Checkpoint**: US1 complete — the mechanism is proven on one real
configuration, with dispatch, SPIR-V, and correctness-citation evidence all
confirmed present

---

## Phase 4: User Story 2 - Measure every WMMA-candidate configuration (Priority: P2)

**Goal**: Extend the proven mechanism to all six configurations and every
in-scope op, covering this workstream's full established scope.

**Independent Test**: Run the full capture and confirm every one of the 42
in-scope (model, scheme, op) combinations produces a directly comparable
tiled-vs-WMMA pair, independent of the single configuration proven in US1.

- [X] T010 [US2] Run `test_llama_baseline_bench` to completion (same no-env-var invocation as T005) capturing the full 192-row `RESULT` catalog → `specs/007-wmma-improvement-microbench/results/raw/wmma_bench_raw.log` (depends on T009) — 192 rows captured against the post-Decision-8 binary
- [X] T011 [US2] For every one of the 42 in-scope rows (all ops except `lm_head`, per research.md Decision 3; `regime=prefill`, `storage=buffer`), verify `dispatch_status` via the kernel-name check (FR-004) — record any `fallback` case explicitly with its actual kernel name, do not drop it (depends on T010) — all 42 (and in fact all 48, including `lm_head`) confirmed dispatching a genuine `coopmat` kernel; zero fallback cases
- [X] T012 [US2] Confirm `lm_head` and every decode-regime row are identified as excluded from the in-scope set, with the stated reason (FR-006, research.md Decision 3) — never silently dropped from the configuration count (depends on T010) — confirmed excluded per Decision 3/FR-006, handled by `compare_wmma.py`'s `IN_SCOPE_OPS` list and Excluded/Out-of-Scope section

**Checkpoint**: US2 complete — every one of the 42 in-scope configurations
has a captured WMMA measurement (or an explicit fallback reason)

---

## Phase 5: User Story 3 - Report the improvement, in full and at a glance (Priority: P3)

**Goal**: Turn the measurements into an answer: how much does the existing
WMMA/coopmat shader actually speed up its candidate operations?

**Independent Test**: Generate the report and confirm each row's speedup,
significance, and correctness-verification status traces directly to its
own two measured times, with an overall time-weighted figure computed from
the full set.

- [X] T013 [US3] Implement `specs/007-wmma-improvement-microbench/scripts/compare_wmma.py`: load `004`'s Buffer tiled-baseline rows + this feature's new WMMA rows; compute per-row speedup %, significance (non-overlapping `mean ± 2·stdev` bands, matching `004`'s rule), `correctness_verified` (dispatch + SPIR-V + existing-test citation, per FR-007), and the time-weighted overall figure (research.md Decision 6, revised by its addendum to weight by each op's own share of tiled-baseline time rather than `003`'s `pct_of_phase`, which can't be split for same-shape sibling ops) (depends on T011)
- [X] T014 [US3] Implement the report renderer: overall time-weighted figure at the top, 42-row case table, Excluded/Out-of-Scope section (always present, even if empty), correctness-verification summary — per `contracts/wmma-improvement-report-schema.md` (depends on T013) — **also added a per-scheme breakdown before the blended figure**, found necessary during T015/T016: `4w` and `8da4w` move in opposite directions consistently, so the single blended number alone would misrepresent the result
- [X] T015 [US3] Run end to end → `specs/007-wmma-improvement-microbench/results/wmma-improvement-report.md` (depends on T012, T014) — done; `4w` +60.6%, `8da4w` -15.2%, blended +22.7%
- [X] T016 [US3] Self-review against SC-001 through SC-005: confirm every configuration appears (measured or explicitly excluded), no result is missing its iteration count/stdev, correctness-verified vs. unverified rows are distinguishable, and the overall figure is explicitly stated as time-weighted, not a bare average (depends on T015) — all five pass; SC-003 required adding an explicit, data-verified iteration-count statement (5 runs, confirmed uniform across all rows, not assumed) since the table alone only showed stdev

**Checkpoint**: US3 complete — the report answers whether (and by how
much) WMMA helps, with full per-configuration traceability

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T017 Reproducibility spot-check: re-run one in-scope configuration's WMMA capture and confirm it matches the original within noise, matching `001`'s established reproducibility discipline (a single extra run, per `001/tasks.md` T038's precedent) — re-ran `llama-3.2-1b`/`4w`/`wq`: 1836.5±8.7us vs original 1938.0±72.5us, `mean±2·stdev` bands overlap ([1819.2,1853.8] within [1793.0,2082.9]) — reproducible within noise; same kernel dispatched both times
- [X] T018 [P] Update `quickstart.md` with any corrections found during T005-T017 (if any were needed) — updated: (1) `compare_wmma.py`'s actual CLI args (dropped unused `--candidates-dir`/`--spirv-dir`), (2) should also document the Decision-8 wiring fix as a one-time prerequisite for anyone re-running this study from a pre-fix checkout

---

## ✅ RESOLVED: US1 found a production dispatch-wiring gap, not a data point (2026-07-04)

**Resolution**: fixed per user decision ("fix the wiring first, then measure both") — see `research.md` Decision 8 for the full root cause, fix, and safety verification. T009-T018 completed against the corrected binary. Result: `4w` is genuinely ~60% faster with WMMA; `8da4w` is ~15% slower (a real regression, not noise) — see `results/wmma-improvement-report.md`. Original STOP note preserved below for the investigation trail.

T005/T006 measured `llama-3.2-1b`/`4w`/`wq` and found `ET_VK_FORCE_TILED_LINEAR`-unset dispatch still produced the **tiled** kernel, not coopmat. This is not shape misalignment (verified: `M=2048`, `K=2048`, `N=2048` satisfy `4w`'s `(128, 64, 16)` tile requirement) and not noise — **every** `4w` row in the full T005 capture output (all 6 configs x 7 ops, i.e. every row with `scheme=4w`) shows the same tiled kernel, while **every** `8da4w` row shows a genuine `coopmat` kernel. A clean scheme-level split, not per-shape randomness.

Root-caused via direct code tracing (constitution Principle VI), not assumed:

- The real 4w export path (`op_registry.py:455`) and every existing prototyping harness (`test_llama_baseline_bench.cpp`, `test_coopmat_linear_bench.cpp`) construct the op name `et_vk.linear_q4gsw.default`. `VK_GET_OP_FN` resolves this to **`Q4gswLinear.cpp`** (`VK_REGISTER_OP(et_vk.linear_q4gsw.default, q4gsw_linear)`) — a file last touched by an unrelated PR (#20055, activation-transpose preprocessing), with **zero coopmat awareness**: its kernel names are always `q4gsw_linear_gemm__*` or `q4gsw_linear_gemv_coop__*`.
- The coopmat-capable weight-only dispatch logic this workstream built (`can_use_q4gsw_coopmat`, `kQ4gswCoopmatDims`, `add_linear_qw_node`'s `nbits==4` branch, and the `linear_q4gsw_coopmat_*.spv` shaders -- confirmed via `spirv-dis` in T007 to contain real `OpCooperativeMatrixMulAddKHR` instructions) lives in `QuantizedLinear.cpp`, but is **only reachable via `linear_q8csw`'s registration** (`et_vk.linear_q8csw.default`) — and `linear_q8csw`'s own C++ body hardcodes `weight_quant_config(8, kPerChannel, {K})`. It can never take the `nbits==4` branch either.
- **No op registered anywhere today calls the coopmat-capable weight-only path with a 4-bit weight config.** The `4w` scheme's coopmat machinery is fully built, compiles to genuine WMMA instructions, but is unreachable dead code from every current entry point -- the real model included.
- Knock-on effect: `test_coopmat_linear_bench.cpp`'s own "coopmat vs tiled" correctness/perf comparison for `op="linear_q4gsw"` resolves through the same `Q4gswLinear.cpp` path on **both** sides it labels "tiled" and "coopmat" -- meaning that comparison, and any correctness claim resting on it, has likely never actually exercised the real coopmat kernel for this scheme.
- The `8da4w` scheme (`et_vk.linear_dq8ca_q4gsw.default`, registered directly in `QuantizedLinear.cpp`) is unaffected and dispatching coopmat correctly, confirmed in this same capture.

**T009-T018 are paused pending user direction** -- computing a "4w WMMA speedup" number right now would report 0%/tiled-vs-tiled as if it were a measurement, when it's actually an unreachable-code finding that needs a decision (fix the wiring? scope this study to `8da4w` only and report `4w` as blocked? something else?), not a number.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup. BLOCKS all user stories — there is no code to write here, only readiness to confirm, but nothing downstream may start until it's confirmed
- **User Story 1 (Phase 3)**: Depends on Foundational
- **User Story 2 (Phase 4)**: Depends on US1 (T009) — extends the proven single-configuration mechanism to the rest
- **User Story 3 (Phase 5)**: Depends on US2 (T011, T012)
- **Polish (Phase 6)**: Depends on US3

### Within Each User Story

- US1: capture (T005) → dispatch check (T006) → speedup computation (T009), strictly sequential; SPIR-V check (T007) and correctness-citation check (T008) can proceed independently of the capture itself
- US2: T010 (full capture) is a single GPU-bound run; T011/T012 are analysis passes over its output, sequential after it (no new GPU work)

### Parallel Opportunities

- T002, T003, T004 (Foundational readiness checks) can all run in parallel — independent resources, no shared state
- T007 and T008 (US1) can run in parallel with each other and with T005 — they verify properties of the already-compiled shader and the already-existing correctness test, not of the new capture
- T018 (Polish) has no dependency on T017 and could run alongside it
- **Never** parallelize T005/T010/T017 with each other or with any other GPU-bound task — they share the MiniPC's one GPU, matching this workstream's established discipline (`001`/`002`/`006`)

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1 — one configuration, full dispatch/SPIR-V/correctness evidence proven
4. **STOP and VALIDATE**: the mechanism works end-to-end before spending device time on the other 41 rows

### Incremental Delivery

1. Setup + Foundational → readiness confirmed, no code written
2. US1 → proven on one real configuration, with full verification evidence
3. US2 → all 42 in-scope configurations measured (or explicitly excluded)
4. US3 → the actual answer: how much does WMMA help, time-weighted by real impact?

---

## Notes

- No commits until the user explicitly asks, per repo convention.
- This feature was planned as zero-production-code-change (no safety-
  verification hard gate like `006`'s T004 was anticipated) — **US1 itself
  turned out to BE the hard gate**: T005/T006 found `4w` coopmat was
  unreachable from any registered op, a real production dispatch-wiring
  gap, not a measurement question. Fixed per user decision (research.md
  Decision 8), safety-verified via `test_q4gsw_linear` (72/72 passed) and
  `test_coopmat_linear_bench` (`linear_q4gsw` cases all passed, ~5x speedup
  visible in its own summary) before trusting any of the other 41 rows.
  This is exactly the discipline `001`'s reasoning for a hard gate
  anticipates — it just arrived from a different direction than planned.
- `lm_head` is deliberately excluded from every task above (research.md
  Decision 3) — do not add it back without revisiting that decision.
- RGA is not installed on this machine (plan.md Constraints) — no task
  above depends on it; `spirv-dis` is the actual tool used for FR-007's
  SPIR-V check.
- **Separate, unfixed finding** (research.md Decision 8): `test_coopmat_linear_bench`
  showed 5 pre-existing correctness failures, all `linear_dq8ca_q4gsw` at
  Texture3D storage — unrelated to this feature's `4w` fix (no `dq8ca` code
  was touched), not diagnosed or fixed here, flagged for separate follow-up.
