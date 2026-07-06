---

description: "Task list for End-to-End tok/s Report — Texture, Buffer, and WMMA Across 4w/8da4w"
---

# Tasks: End-to-End tok/s Report — Texture, Buffer, and WMMA Across 4w/8da4w

**Input**: Design documents from `/specs/009-e2e-tokrate-report/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md (all present)

**Tests**: One genuinely new test is required — a small rank-3 (batch=1)
correctness check for both coopmat shaders against the CPU/tiled reference
(research.md Decision 2), since no existing test covers this shape class.
Beyond that, no separate automated suite: the smoke-check, the ETDump
dispatch check, and the report self-review are this feature's own inline
verification, matching how `001`/`004`/`006`/`007` validated their own work.

**Organization**: Tasks are grouped by user story. This is **real device
work** (build, export, ETDump capture, e2e capture on `rocky-ryzen`), like
every prior tier-2 feature, and it includes a small, explicitly-authorized
production dispatch-code change (the rank-3 guard relaxation) — the first
production change in this workstream that requires authorization to be
sought *during this feature's own implementation*, rather than already
having been granted in a prior feature (`007`'s fix was authorized on
2026-07-04 during `007`'s own implementation, and is simply reused here).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files/resources, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Paths are relative to the repository root

## Path Conventions

- `backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp` — the rank
  guard relaxation (new) and `007`'s already-authorized `linear_q4gsw`
  registration fix (uncommitted, reused)
- `backends/vulkan/runtime/graph/ops/impl/Q4gswLinear.cpp` — `007`'s fix,
  reused unchanged
- `backends/vulkan/test/custom_ops/` — new rank-3 correctness check
- `specs/009-e2e-tokrate-report/scripts/compare_e2e_tokrate.py` — new
  analysis script
- `specs/009-e2e-tokrate-report/results/pte/`, `results/etdump/`,
  `results/raw/` — new WMMA-eligible exports, ETDump captures, e2e logs
- `specs/009-e2e-tokrate-report/results/e2e-tokrate-report.md` — the
  deliverable

---

## Phase 1: Setup

- [X] T001 Create `specs/009-e2e-tokrate-report/scripts/`, `specs/009-e2e-tokrate-report/results/pte/`, `specs/009-e2e-tokrate-report/results/etdump/`, and `specs/009-e2e-tokrate-report/results/raw/` directories

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Produce and safety-verify the one thing every user story's
WMMA-eligible export depends on — a working coopmat dispatch path for
rank-3, batch=1 activations — before any device time is spent measuring

**⚠️ CRITICAL**: No user story work can begin until this phase is complete AND its safety property (T007) is verified

- [X] T002 Present research.md Decision 1's exact proposed diff to `can_use_q4gsw_coopmat()`'s rank check (`backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp:192-196`) and obtain explicit user authorization to apply it, per FR-009 — **authorized**: user explicitly approved applying the fix, noting that wiring rigor for upstream `pytorch/executorch` submission is a separate, later concern from landing it here on `sarc-acl` (constitution's Repository & Distribution Scope)
- [X] T003 Apply the authorized relaxation: change `graph->dim_of(output) > 2` to reject only a genuine batch (product of all dims before the trailing two `!= 1`), with an inline comment naming what changed and why a size-1 leading dim is safe (cites `quantized_linear_global_wg_size`'s and the coopmat shaders' trailing-two-dims-only addressing, per constitution Principle V/VI) (depends on T002) — applied at `QuantizedLinear.cpp:192-220`, replacing the `dim_of(output) > 2` check with a `leading_dims_numel != 1` check computed from `out_sizes` (reused for the existing M/N extraction below, removing a duplicate `sizes_of` call)
- [X] T004 [P] Confirm `007`'s `linear_q4gsw` registration fix (currently uncommitted diff in `backends/vulkan/runtime/graph/ops/impl/Q4gswLinear.cpp` and `QuantizedLinear.cpp`, routing `et_vk.linear_q4gsw.default` through `add_linear_qw_node`) is present and intact in the working tree used for this feature's build — confirmed present (uncommitted, pre-existing in working tree from `007`)
- [X] T005 [P] Confirm `--vulkan-storage-override` (`006`, already committed in `examples/models/llama/export_llama_lib.py`/`extension/llm/export/partitioner_lib.py`) is present and functional — confirmed present (`export_llama_lib.py:465`, already committed)
- [X] T006 Rebuild per the constitution's Reference Build Recipe with `EXECUTORCH_BUILD_DEVTOOLS=ON` added, incorporating T003's and T004's code (depends on T003, T004) — main install (`libvulkan_backend.a`) and `backends/vulkan/test/custom_ops` sub-build both rebuilt clean; **note**: the separate `EXECUTORCH_ENABLE_EVENT_TRACER=ON` profiling-build step (`002`'s recipe, a distinct build tree to avoid tracer overhead contaminating e2e timing) is deferred to US1's T012, not needed for T007/T008's correctness checks
- [X] T007 **Safety verification (hard gate)**: re-run every existing rank-2 coopmat correctness/benchmark case against the T006 build and confirm zero regressions vs. pre-change behavior (depends on T006) — **`test_fpa_q4gsw_linear` is Buck-only (`targets.bzl`, not wired into this directory's CMakeLists.txt) and Buck isn't installed in this environment, so it could not be run; flagged here rather than silently skipped, per constitution Principle VI.** `test_q4gsw_linear`: 72 PASSED / 40 SKIPPED / 0 FAILED (matches `007`'s own prior baseline count exactly). `test_coopmat_linear_bench`: same 5 pre-existing `linear_dq8ca_q4gsw`-at-`Texture3D` failures as `007` documented (unrelated to this change — different op path, different storage type, no rank/batch involved) — zero new regressions
- [X] T008 Add and run the new rank-3 (batch=1), tile-aligned correctness check for both `linear_q4gsw_coopmat` and `linear_dq8ca_q4gsw_coopmat` against the CPU/tiled reference (research.md Decision 2) in `backends/vulkan/test/custom_ops/test_coopmat_linear_bench.cpp` (depends on T007) — added a `batch` field to `LinearConfig`, made `bench_reference`/`flop_calc`'s M/K/N extraction rank-agnostic (trailing-two-dims, matches production's own addressing), factored the existing deterministic-data recipe into `make_deterministic_correctness_case()` and reused it for one new rank-3 `[1,128,128]` case per op at Buffer storage, plus an explicit post-run dispatch+correctness check. **Result: both ops' output shape confirmed `[1x128x128]`, dispatched kernel confirmed `_coopmat` (not tiled fallback), correctness=PASSED for both** — this is the first real proof the guard relaxation actually enables (and correctly computes) coopmat dispatch for the shape class it was designed to unblock
- [X] T009 [P] Confirm `specs/006-e2e-storage-comparison/results/e2e-storage-comparison-report.md`, `specs/007-wmma-improvement-microbench/results/wmma-improvement-report.md`, and `specs/008-8da4w-parameter-sweep/results/sweep-report.md` are present and readable — read-only inputs this feature never re-captures — confirmed present

**Checkpoint**: Foundation ready — the rank-3 guard relaxation is applied, authorized, and safety-verified against both existing and new correctness coverage; the build has ETDump support; all upstream reports are available

---

## ✅ RESOLVED: T012 found a second, deeper blocker — `--vulkan-force-fp16` silently defeated `006`'s storage override (2026-07-05)

**Not part of the original plan** — found while executing T012 against `006`'s reused "Buffer" `.pte` (see the now-superseded note on T010 below): ETDump showed every real per-layer linear op still dispatching the **tiled** kernel with `_texture3d_` output storage, despite `--vulkan-storage-override buffer`. Root-caused via direct code tracing (constitution Principle VI) to `tag_memory_meta_pass.py`'s `constrain_op_arg_repset()` unconditionally forcing every op argument to `ANY_TEXTURE` whenever `force_fp16` is set — *before* the storage-override preference is ever consulted — an interaction from an unrelated upstream commit (`e4aba1e658`) that predates `006`'s own mechanism entirely. Full root cause, fix, and three-part verification (safety property, fix efficacy, correctness spot-check) in `research.md` Decision 8.

**Resolution**: fixed per explicit user authorization ("dig" → "apply the fix and re-verify"). One-line-scoped change in `tag_memory_meta_pass.py:419-420`, gated so it can only change behavior for the `force_fp16 AND storage_type_override=BUFFER` combination — a combination that never worked correctly before this fix. Default (no-override) behavior confirmed byte-identical via ETDump re-check.

**Consequence for T010/T011 above**: `006`'s existing "Buffer" `.pte` files were exported *before* this fix and do not benefit from it — reusing them was based on a correct-in-isolation but incomplete premise (dispatch is a runtime decision, but runtime dispatch itself was silently gated on a storage property the export never actually achieved). All six configurations are being **freshly re-exported** against the fixed pass — see corrected T010/T014 below.

---

## Phase 3: User Story 1 - Confirm the WMMA-eligible e2e export actually dispatches coopmat (Priority: P1) 🎯 MVP

**Goal**: Prove one (model, scheme) configuration's WMMA-eligible export
actually dispatches the coopmat kernel end-to-end — not a silent tiled
fallback — before trusting any WMMA-arm timing number.

**Independent Test**: Produce one configuration's WMMA-eligible export, run
it, and confirm via ETDump that the coopmat kernel, not the tiled fallback,
actually executed.

- [X] T010 [US1] Export `llama-3.2-1b`/`4w` with `--vulkan-storage-override buffer` against the T006 build → `specs/009-e2e-tokrate-report/results/pte/llama-3.2-1b_4w.pte` (depends on Foundational checkpoint) — **superseded twice**: (1) first symlinked `006`'s existing Buffer `.pte` (reasoning: dispatch is a pure runtime decision, not baked into the `.pte`) — this reasoning is correct but incomplete, since it assumed `006`'s export actually achieved Buffer storage; (2) T012 disproved that, uncovering the `force_fp16`/storage-override conflict (research.md Decision 8). **Final state**: freshly re-exported after the pass fix — `--model llama3_2 -c .../llama3_2_1b/original/consolidated.00.pth -p .../params.json -t .../tokenizer.model -kv --use_sdpa_with_kv_cache -qmode 4w --group_size 32 --max_seq_length 3072 --max_context_length 3072 -V --vulkan-force-fp16 --vulkan-storage-override buffer`. Also relinked `llama_main` (stale, predated T003/T006's rebuild) and built a second `cmake-out-vk-etdump` tree (`EXECUTORCH_ENABLE_EVENT_TRACER=ON`, mirroring `002`'s separate-build precedent to avoid tracer overhead contaminating later e2e timing)
- [X] T011 [US1] Smoke-check it (`006`'s bar: completes without crashing, `generated_tokens` matches request, coherent/non-degenerate output) (depends on T010) — passed on the final (post-fix) export: no crash, `generated_tokens=31` for the standard smoke prompt; separately spot-checked "The capital of France is" → "Paris. The capital of the United Kingdom is London. The capital of the United States is Washington," (coherent, factually correct)
- [X] T012 [US1] Capture ETDump (`--etdump_path`) for that run and, per `002`'s `kernel_name` extraction, confirm every measured linear op's kernel name contains `_coopmat` (FR-003) (depends on T011) — **first attempt (against `006`'s reused `.pte`) found all 112 per-layer linear ops dispatching `linear_q4gsw_tiled_texture3d_texture2d_half` (NOT coopmat) — this is what uncovered Decision 8's blocker, not a pass/fail of this task alone.** After the pass fix and re-export: all 112 confirmed `linear_q4gsw_coopmat_buffer_texture2d_half`; the 1 remaining non-coopmat linear dispatch is the expected GEMV (`M=1`) case (`003`'s classification, no WMMA-capable GEMV kernel exists, unaffected by storage)
- [X] T013 [US1] Record the outcome explicitly: `dispatch_status: confirmed` if T012 passed, or `fallback`/`blocked` with the actual kernel name or error if it didn't — no WMMA number is trusted from this configuration otherwise (FR-007) (depends on T012) — `dispatch_status: confirmed` for `llama-3.2-1b`/`4w` on the final export

**Checkpoint**: US1 complete — the dispatch mechanism is proven (or its
failure mode is understood) on one real configuration

---

## Phase 4: User Story 2 - Measure e2e prefill/decode tok/s for all three dispatch arms, both schemes, all three models (Priority: P2)

**Goal**: Extend the proven mechanism to the remaining five configurations
and capture the WMMA arm's e2e numbers alongside `006`'s already-captured
`Texture3D`/`Buffer` numbers.

**Independent Test**: Run the standard e2e capture procedure against one
additional WMMA-eligible configuration and confirm a directly comparable
prefill/decode tok/s pair, independent of the other four.

- [X] T014 [US2] Export the remaining 5 configurations with `--vulkan-storage-override buffer` → `specs/009-e2e-tokrate-report/results/pte/<model>_<scheme>.pte` (depends on T013) — **superseded**: first symlinked from `006` (same reasoning/correction as T010); all 5 freshly re-exported after research.md Decision 8's pass fix (`--model llama3_2` for 1B/3B, `--model llama3_1` for 8B; same flags as T010 otherwise, checkpoints from `/home/doremy/checkpoints/llama3_2_{1b,3b}/original/` and `/home/doremy/archive/llama3_1_8b/original/`)
- [X] T015 [US2] Smoke-check each of the 5 (same criteria as T011) (depends on T014) — all 5 passed on the final export (no crash, `generated_tokens=31`): `llama-3.2-1b_8da4w`, `llama-3.2-3b_4w`, `llama-3.2-3b_8da4w`, `llama-3.1-8b_4w` (247.1 tok/s prefill), `llama-3.1-8b_8da4w`
- [X] T016 [US2] Capture ETDump and confirm dispatch for each of the 5 (same check as T012) (depends on T015) — `llama-3.2-1b_8da4w`: 112 `linear_dq8ca_q4gsw_coopmat_buffer_texture2d_half`; `llama-3.2-3b_4w`/`llama-3.2-3b_8da4w`: 196 (28 layers × 7 ops) each; `llama-3.1-8b_4w`/`llama-3.1-8b_8da4w`: 224 (32 layers × 7 ops) each — all confirmed `_coopmat`, all `dispatch_status: confirmed`. (First `llama-3.1-8b_8da4w` export attempt failed to save — output directory wasn't pre-created; retried successfully, no data lost since the failure was caught before copying)
- [X] T017 [US2] Record any `fallback`/`blocked` configuration explicitly with its actual kernel name or error — never silently excluded from the six-configuration count (FR-007) (depends on T016) — **none blocked/fallback**: all 6 configurations' `dispatch_status: confirmed`
- [X] T018 [US2] Capture e2e prefill/decode tok/s for every configuration whose dispatch was `confirmed` (T013 or T016), using `001`/`006`'s exact methodology (fixed 2048/1024 workload, 5 repeated runs, discard cold-start drift where already documented, no concurrent load) (depends on T013, T017) — all 6 configurations captured, 5 reps each, `ps aux`/`free -h` confirmed clean before each config (no concurrent export/benchmark, per `001`'s documented contention warning). Results (WMMA prefill tok/s, mean±stdev): `llama-3.1-8b`/`4w` 316.53±1.09, `llama-3.1-8b`/`8da4w` 205.75±0.42, `llama-3.2-3b`/`4w` 649.88±2.99, `llama-3.2-3b`/`8da4w` 432.68±1.66, `llama-3.2-1b`/`4w` 1867.40±33.93, `llama-3.2-1b`/`8da4w` 1265.03±9.14 → `specs/009-e2e-tokrate-report/results/raw/<model>_<scheme>_rep{1..5}.log`

**Checkpoint**: US2 complete — every dispatch-confirmed configuration has a
`Texture3D` (from `006`), `Buffer` (from `006`), and WMMA (this feature) e2e
prefill/decode tok/s triple; every other configuration has an explicit
blocked/fallback reason

---

## Phase 5: User Story 3 - Report whether WMMA actually helps at the e2e level, per configuration (Priority: P3)

**Goal**: Turn the measurement triples into a consolidated answer: does
WMMA actually help this device's real token generation rate, per scheme?

**Independent Test**: Generate the report and confirm each configuration's
verdict traces directly to its own measurement triple and to `007`'s (and,
for `8da4w`, `008`'s) prior findings.

- [X] T019 [US3] Implement `specs/009-e2e-tokrate-report/scripts/compare_e2e_tokrate.py`: load `006`'s `Texture3D`/`Buffer` numbers, `007`'s and `008`'s findings, and this feature's new WMMA capture (T018); compute `wmma_vs_buffer_pct`/`wmma_vs_texture3d_pct`, tag every prefill row with the inherited cross-session caveat (research.md Decision 5), and determine each row's `consistent`/`diverges`/`not_applicable` verdict against `007`'s (and, for `8da4w`, `008`'s) findings (research.md Decision 7) (depends on T018, T009) — implemented; parses `006`'s report table directly via regex rather than re-deriving numbers (one bug found and fixed: the model-name capture group excluded `-`, breaking on `llama-3.1-8b`'s internal hyphens)
- [X] T020 [US3] Implement the report renderer: two per-scheme top-line verdicts (never one blended number, research.md Decision 7), the full comparison table, and a Blocked/Failed section (always present, even if empty) — per `contracts/e2e-tokrate-report-schema.md` (depends on T019) — also added a prominent top-of-report correction note carrying forward research.md Decision 8 (006's own Buffer numbers never actually achieved Buffer storage either — reused as-is per FR-001, but flagged as a second Texture3D-equivalent baseline, not silently presented as a true Buffer measurement)
- [X] T021 [US3] Run end to end → `specs/009-e2e-tokrate-report/results/e2e-tokrate-report.md` (depends on T017, T020) — done. Headline: `4w` +77.8% faster e2e (3/3 consistent with `007`'s +60.6%), `8da4w` -3.2% slower e2e (3/3 consistent with `007`'s -15.2%, smaller magnitude since prefill linear GEMM is only ~51-67.5% of prefill time per `003`)
- [X] T022 [US3] Self-review against SC-001 through SC-004: confirm every configuration appears (measured or explicitly blocked), no WMMA number appears without a `confirmed` dispatch status, every prefill divergence carries its cross-session caveat, and each scheme has its own explicit "does WMMA help" statement (depends on T021) — all four pass: 6/6 configurations in the table, 0 blocked (explicitly stated), every prefill row's caveat present in Notes, both `4w`/`8da4w` have their own verdict headers

**Checkpoint**: US3 complete — the report answers whether WMMA helps this
device's real token generation rate, per scheme, with full traceability

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T023 Reproducibility spot-check: re-run one WMMA-arm configuration's e2e capture and confirm it matches the original within noise, matching `001`'s established reproducibility discipline — re-ran `llama-3.2-1b`/`4w`: prefill 1887.56 tok/s (within the original 5-rep range 1818.83-1906.89), decode 59.975 tok/s (within 59.916-60.014), `generated_tokens=1023` as expected — reproducible within noise
- [X] T024 [P] Update `quickstart.md` with any corrections found during T002-T023 (if any were needed) — updated: documented the pass fix (research.md Decision 8) as a prerequisite step, corrected the export step to require a fresh re-export rather than reuse of `006`'s `.pte`s, and added the ETDump-build/dispatch-check step with the observed per-model kernel-invocation counts

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup. BLOCKS all user stories —
  T002's authorization gate blocks T003 onward; T007's safety-verification
  gate blocks every user story
- **User Story 1 (Phase 3)**: Depends on Foundational (T007, T008, T009 all
  passed)
- **User Story 2 (Phase 4)**: Depends on US1 (T013) — extends the proven
  single-configuration mechanism to the rest
- **User Story 3 (Phase 5)**: Depends on US2 (T018) and Foundational (T009)
- **Polish (Phase 6)**: Depends on US3

### Within Each User Story

- US1: export (T010) → smoke-check (T011) → ETDump capture/dispatch check
  (T012) → record outcome (T013), strictly sequential
- US2: T014 (export 5) → T015 (smoke-check 5) → T016 (ETDump/dispatch check
  5) → T017 (record outcomes), sequential per configuration but the 5
  configurations' export+smoke-check+dispatch-check can proceed in parallel
  with each other (different `.pte`/`.etdump` files, no shared state); T018
  (e2e capture) depends on both T013 and T017 since it only runs against
  configs with `dispatch_status: confirmed`

### Parallel Opportunities

- T004, T005, T009 (Foundational readiness checks) can all run in parallel
  — independent resources, no shared state
- Within T014-T016, the 5 remaining configurations' export/smoke-check/
  dispatch-check can proceed in parallel with each other
- T024 (Polish) has no dependency on T023 and could run alongside it
- **Never** parallelize T010/T012/T014/T016/T018/T023 with each other or any
  other GPU-bound task — they share the MiniPC's one GPU, matching this
  workstream's established discipline

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational — **including the authorization gate
   (T002) and the safety-verification gate (T007)**
3. Complete Phase 3: User Story 1 — one configuration, dispatch proven (or
   its failure mode understood)
4. **STOP and VALIDATE**: the dispatch mechanism works end-to-end before
   spending device time on the other five configurations

### Incremental Delivery

1. Setup + Foundational (+ authorization + safety verification) → rank-3
   dispatch path ready and proven not to regress existing behavior
2. US1 → proven on one real configuration
3. US2 → all six configurations measured (or explicitly blocked/fallback)
4. US3 → the actual answer: does WMMA help this device's real token
   generation rate, per scheme?

---

## Notes

- No commits until the user explicitly asks, per repo convention.
- T002 is non-negotiable: FR-009 requires explicit user authorization before
  any change to `can_use_q4gsw_coopmat()` lands, even though research.md
  already grounds the mechanism as safe — proposing is not the same as
  authorizing.
- T007's safety verification is equally non-negotiable: if the relaxation
  changes behavior for any already-passing rank-2 or genuine-batch(>1) case,
  that is a regression this feature must not ship, not just a risk to note.
- T008's new rank-3 correctness check is required, not optional — unlike
  `007`'s registration fix, which could cite existing 2D coverage, this
  relaxation reaches a shape class with zero prior test coverage.
- FR-008 (shipped tile config only) requires no dedicated task — it is
  satisfied by not adding any tile/subgroup override, since production's own
  `can_use_q4gsw_coopmat()` never exposes one.
- Any configuration where T003's relaxation still doesn't make the rank-3
  fix applicable, or where T012/T016's dispatch check fails, is reported
  per FR-007 in T017/T022 — never silently dropped from the six-
  configuration count.
