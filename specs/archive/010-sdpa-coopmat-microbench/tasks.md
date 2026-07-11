---

description: "Task list for SDPA Coopmat Correctness + Microbenchmark"
---

# Tasks: SDPA Coopmat Correctness + Microbenchmark

**Input**: Design documents from `/specs/010-sdpa-coopmat-microbench/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md (all present)

**Tests**: The correctness check itself (a new `sdpa_test.cpp` case, run both
with and without the coopmat toggle) IS the required new test coverage for
this feature (FR-001) — not optional, and not satisfiable by citing existing
cases (research.md Decision 2: no existing case is tile-aligned to the
coopmat gate).

**Organization**: Tasks are grouped by user story. This is **real device
work** (build, correctness-check, microbenchmark on `rocky-ryzen`), like
every prior microbenchmark-tier feature, but unlike `007`/`009` it adds
**no new production dispatch code** by default — the SDPA coopmat path
already exists from the `yanwen/quant-dev-active` import; this feature
verifies and measures it. A production fix only enters scope if User Story
1 finds a real bug (see Notes).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files/resources, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Paths are relative to the repository root

## Path Conventions

- `backends/vulkan/test/op_tests/sdpa_test.cpp` — existing ATen-referenced
  correctness suite, gets one new tile-aligned case
- `backends/vulkan/test/custom_ops/test_sdpa_coopmat_bench.cpp` — new timed
  microbenchmark harness
- `backends/vulkan/test/custom_ops/test_coopmat_attention_bench.cpp` —
  **out of scope**, tests an unrelated shader family (research.md Decision 1)
- `specs/010-sdpa-coopmat-microbench/scripts/compare_sdpa_coopmat.py` — new
  analysis script
- `specs/010-sdpa-coopmat-microbench/results/` — new SPIR-V dumps, raw
  capture log, and the report

---

## Phase 1: Setup

- [X] T001 Create `specs/010-sdpa-coopmat-microbench/scripts/`, `specs/010-sdpa-coopmat-microbench/results/raw/`, and `specs/010-sdpa-coopmat-microbench/results/spirv/` directories

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Confirm every mechanism this feature depends on is actually
present, and establish a working, regression-tested `op_tests` build before
adding new coverage.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete AND T006's regression baseline passes

- [X] T002 [P] Confirm `009`'s `tag_memory_meta_pass.py` fix (the `force_fp16`/`ANY_STORAGE` change) is present in the working tree — hard prerequisite per research.md Decision 6: without it, `Buffer` storage never reaches Q/K/V/attn_weights, and SDPA's coopmat gate (`sdpa_buf_half`) can never fire regardless of anything else this feature does — confirmed present at `tag_memory_meta_pass.py:419-436`
- [X] T003 [P] Confirm the SDPA coopmat shaders/dispatch code (`SDPA.cpp`'s `sdpa_coopmat_device_ok`/`sdpa_buf_half`/`sdpa_cm_aligned` gates, `sdpa_compute_attn_weights_coopmat.{glsl,yaml}`, `sdpa_compute_out_coopmat.{glsl,yaml}`) are present and intact from the `yanwen/quant-dev-active` import — confirmed present, all 4 shader files + gate functions intact
- [X] T004 Configure `backends/vulkan/test/op_tests` as its own CMake sub-build on top of the installed `cmake-out-vk` tree (research.md Decision 7 — requires `find_package(GTest)`, links Torch/ATen; not currently configured in this environment) — **deviation found**: the correctness binary (actually named `vulkan_sdpa_test`, not `sdpa_test`) requires `custom_ops_aot_lib`, gated behind `EXECUTORCH_BUILD_KERNELS_LLM_AOT` (was `OFF`). Reconfigured and rebuilt the main `cmake-out-vk` install with `-DEXECUTORCH_BUILD_KERNELS_LLM_AOT=ON` first (produced `libcustom_ops_aot_lib.so`), then this sub-build configured cleanly with no skip message
- [X] T005 Build the `op_tests` sub-build; confirm `sdpa_test` compiles and links clean (depends on T004) — `vulkan_sdpa_test`, `vulkan_rope_test`, `quantized_linear_test` all built clean
- [X] T006 **Regression baseline (hard gate)**: run the existing `sdpa_test` suite unmodified and confirm every pre-existing case still passes (depends on T005) — this is the safety baseline T007's new case is added against; a pre-existing failure here is unrelated to this feature and must be understood before proceeding, not silently attributed to the new work later — **all 12 tests PASSED** (9 `VulkanGeneralSDPATest` + 3 `VulkanSDPATest`)

**Checkpoint**: Foundation ready — `op_tests` builds and its existing suite
passes; both hard prerequisites (`009`'s pass fix, the imported SDPA
coopmat code) confirmed present

---

## Phase 3: User Story 1 - Prove the SDPA coopmat path is actually correct (Priority: P1) 🎯 MVP

**Goal**: Prove `sdpa_compute_attn_weights_coopmat`/`sdpa_compute_out_coopmat`
actually compute correct attention output and actually dispatch (not a
silent tiled fallback) before any performance number is trusted.

**Independent Test**: Build the correctness harness, run it at one small,
tile-aligned shape for both shaders, confirm the output matches the ATen
reference and confirm (via GPU query-pool kernel-name data, the same
mechanism `test_coopmat_linear_bench.cpp` already relies on) that the
coopmat kernels actually ran.

- [X] T007 [US1] Add one new `TEST(VulkanSDPATest, ...)` case to `sdpa_test.cpp`: `Buffer` storage, a shape satisfying both shaders' tile alignment (`S`/`context_len` multiples of 128/64/32 per data-model.md's alignment rules — e.g. `S=128, context_len=128`), enabling the graph's query-pool (`GraphConfig::enable_querypool`, the same mechanism `graph.context()->querypool().get_shader_timestamp_data()` exposes to the `custom_ops` benchmark framework) so dispatched kernel names are inspectable after `execute()`; rebuild `sdpa_test` (depends on T006) — implemented as two cases (`test_sdpa_op_coopmat_aligned_tiled_baseline`/`_coopmat`, `head_dim=64, num_heads=8, num_kv_heads=8, S=128`, DECOMPOSED/LLM mode) sharing an extended `test_vulkan_sdpa()` with a new optional `out_dispatched_kernels` param (nullptr-default, zero effect on existing call sites). **Found and fixed a real test-helper gap along the way**: `test_vulkan_sdpa`'s correctness check used `at::allclose(reference_out, vk_out)` with no explicit tolerance (fp64-oriented defaults `rtol=1e-5, atol=1e-8`) — this helper had never been exercised at `at::kHalf` before (existing cases all use default `at::kFloat`); added the same dtype-keyed tolerance (`atol=1e-2, rtol=1e-2` for half) the sibling `test_vulkan_general_sdpa` helper already uses. Strictly loosens the fp32 case (`atol` 1e-8→1e-4), provably safe for already-passing tests
- [X] T008 [US1] Run the new case with `ET_VK_SDPA_COOPMAT` **unset**: confirm it passes against the ATen reference (isolates the new shape itself from any coopmat-specific issue — this must pass on the tiled path first) and confirm via query-pool data that the dispatched kernels are the `_tiled` variants (depends on T007) — **PASSED** (both the correctness assertion and the "no `_coopmat` kernel dispatched" check)
- [X] T009 [US1] Run the same case with `ET_VK_SDPA_COOPMAT=1`: confirm it **still passes** against the identical ATen reference, and confirm via query-pool data that both `sdpa_compute_attn_weights_coopmat` and `sdpa_compute_out_coopmat` actually dispatched (FR-003) — a passing numeric result alone does NOT satisfy this task; the kernel-name check is mandatory (constitution Principle VI: eligibility-gate logic passing is not sufficient evidence, per this workstream's `007`-Decision-8 precedent) (depends on T007) — **PASSED**: correctness held AND both `sdpa_compute_attn_weights_coopmat`/`sdpa_compute_out_coopmat` confirmed dispatched via query-pool kernel names
- [X] T010 [P] [US1] SPIR-V-inspect both coopmat shaders' compiled `.spv` (FR-002) → `specs/010-sdpa-coopmat-microbench/results/spirv/sdpa_compute_attn_weights_coopmat_buffer_buffer_half.dis.txt` and `.../sdpa_compute_out_coopmat_buffer_buffer_half.dis.txt`; confirm `OpCooperativeMatrixLoadKHR`/`OpCooperativeMatrixMulAddKHR` are present (independent of T008/T009 — inspects an already-compiled artifact) (depends on T005) — confirmed: 36 cooperative-matrix instructions (Load/MulAdd/Store) in `sdpa_compute_attn_weights_coopmat`, 20 in `sdpa_compute_out_coopmat`
- [X] T011 [US1] **If T009 fails** (numeric mismatch or dispatch not confirmed): root-cause via direct code tracing (constitution Principle VI), propose the exact fix, and get explicit user authorization before applying it — matching this workstream's established discipline for production dispatch/shader-code changes (`007` Decision 8, `009` Decisions 1 and 8). Do not proceed to User Story 2 until resolved or the shader is explicitly excluded with a stated reason (depends on T009) — **not triggered**: T009 passed cleanly; the only issue found (T007's tolerance gap) was in test infrastructure, not the SDPA coopmat production code, so no production fix or authorization was needed

**Checkpoint**: US1 complete — the SDPA coopmat path is proven correct and
its dispatch is confirmed (or its failure is fully root-caused) on one real
configuration

---

## Phase 4: User Story 2 - Measure SDPA coopmat's real prefill speedup (Priority: P2)

**Goal**: Extend the proven mechanism to all three target models' real
prefill SDPA shapes and capture genuine tiled-vs-coopmat timing.

**Independent Test**: Run the microbenchmark harness at one representative
model's real shape, confirm coopmat dispatch via the harness's own
kernel-name field, and compute a directly comparable speedup for that one
case, independent of the other two models.

- [X] T012 [US2] Author `backends/vulkan/test/custom_ops/test_sdpa_coopmat_bench.cpp` on this workstream's existing `TestCase`/`execute_test_cases` timing framework (research.md Decision 3): build the real `sdpa_with_kv_cache`/`custom_sdpa` op dispatch at a given `(head_dim, num_heads, num_kv_heads)` shape and the fixed 2048-token prefill workload, with a case pair per model (`ET_VK_SDPA_COOPMAT` unset / set), capturing both shaders' dispatched kernel names per case via `get_shader_timings()` (depends on T009, T010) — **superseded (research.md Decision 8)**: the `TestCase`/`ValueSpec` framework has no `SymInt` support (confirmed by reading `utils.h`/`utils.cpp` directly) and `llama.custom_sdpa.default` requires one (`input_pos_symint`); built directly on `ComputeGraph` instead, mirroring `sdpa_test.cpp`'s proven DECOMPOSED-mode construction plus a manual warmup/timed-run loop with mean/stdev from query-pool timestamps, isolating `sdpa_compute_attn_weights_*`/`sdpa_compute_out_*` durations from the cache-update/softmax dispatches in between
- [X] T013 [US2] Add an `add_operator_prototype(test_sdpa_coopmat_bench)` entry to `backends/vulkan/test/custom_ops/CMakeLists.txt`; build (depends on T012) — built clean
- [X] T014 [US2] Run the harness for `llama-3.2-1b` (`head_dim=64, num_heads=32, num_kv_heads=8`) first — prove the mechanism on one model before scaling: confirm both tiled and coopmat runs' dispatched kernel names are as expected, record mean/stdev/iteration count for both (depends on T013) — all 3 models run in a single binary invocation; `llama-3.2-1b`: tiled 23622.8±590.3us, coopmat 9636.6±68.3us, **+59.2% faster**, dispatch confirmed
- [X] T015 [US2] Run the harness for the remaining two models, `llama-3.2-3b` (`head_dim=128, num_heads=24, num_kv_heads=8`) and `llama-3.1-8b` (`head_dim=128, num_heads=32, num_kv_heads=8`) (depends on T014) — `llama-3.2-3b`: tiled 44078.4±1958.8us, coopmat 13136.9±92.5us, **+70.2% faster**; `llama-3.1-8b`: tiled 59666.2±1755.3us, coopmat 17313.3±122.6us, **+71.0% faster** — dispatch confirmed for both → `specs/010-sdpa-coopmat-microbench/results/raw/sdpa_bench_raw.log`
- [X] T016 [US2] Record any model whose real shape fails tile alignment, or whose dispatch/correctness check fails, as explicitly excluded with the specific reason (FR-006) — research.md Decision 5 computed all three as expected-aligned, but this must be confirmed from the actual run, not assumed from that arithmetic alone (depends on T015) — **confirmed: zero exclusions**, all 3 models dispatch-confirmed as predicted

**Checkpoint**: US2 complete — every eligible target model has a directly
comparable tiled/coopmat prefill SDPA timing pair (or an explicit excluded
reason)

---

## Phase 5: User Story 3 - Report whether SDPA coopmat helps, at a glance (Priority: P3)

**Goal**: Turn the measurements into one answer per model: does SDPA
coopmat speed up real prefill attention, and by how much.

**Independent Test**: Generate the report and confirm each model's reported
speedup traces directly to its own two measured times, with dispatch and
correctness status visible alongside it.

- [X] T017 [US3] Implement `specs/010-sdpa-coopmat-microbench/scripts/compare_sdpa_coopmat.py`: parse the microbenchmark raw log, compute per-model speedup % and significance (non-overlapping `mean ± 2·stdev` bands, matching this workstream's established rule) (depends on T016)
- [X] T018 [US3] Implement the report renderer: correctness/dispatch verification summary first (per `contracts/sdpa-coopmat-microbench-schema.md`), then the per-model comparison table, then an Excluded section (always present, even if empty), then one overall statement of whether SDPA coopmat helps at this tier (depends on T017)
- [X] T019 [US3] Run end to end → `specs/010-sdpa-coopmat-microbench/results/sdpa-coopmat-microbench-report.md` (depends on T010, T016, T018) — done. Headline: SDPA coopmat is **66.8% faster** on average (3/3 models, all `real_effect`): `llama-3.1-8b` +71.0%, `llama-3.2-3b` +70.2%, `llama-3.2-1b` +59.2%
- [X] T020 [US3] Self-review against SC-001 through SC-004: confirm the correctness check passed before any perf number appears, every timing carries iteration count/stdev, each of the three models has a clear verdict or exclusion reason, and no reader could mistake an unverified number for a validated one (depends on T019) — all four pass; added an explicit "5 timed runs, 3 discarded warmup" note after finding SC-002's iteration count wasn't stated in the rendered report text itself (only implied by stdev)

**Checkpoint**: US3 complete — the report answers whether SDPA coopmat
helps real prefill attention, with full traceability

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T021 Reproducibility spot-check: re-run one model's microbenchmark case and confirm it matches the original within noise, matching `001`'s established reproducibility discipline — re-ran `llama-3.2-1b`: tiled 23934.0±1307.2us (original 23622.8±590.3, overlapping `mean±2·stdev` bands), coopmat 9681.6±68.6us (original 9636.6±68.3, overlapping), speedup 59.5% vs original 59.2% — reproducible within noise
- [X] T022 [P] Update `quickstart.md` with any corrections found during T002-T021 (if any were needed) — updated: documented the `EXECUTORCH_BUILD_KERNELS_LLM_AOT` prerequisite (T004's finding), corrected the harness description from the planned `TestCase`/`ValueSpec` approach to the actual `ComputeGraph`-direct approach (research.md Decision 8), and added the fp16 tolerance note (T007's finding)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup. BLOCKS all user stories —
  T006's regression baseline is a hard gate before any new case is added
- **User Story 1 (Phase 3)**: Depends on Foundational (T006 passed)
- **User Story 2 (Phase 4)**: Depends on US1 (T009 correctness+dispatch
  confirmed, T010 SPIR-V confirmed) — per constitution Principle I, no
  timing is built or trusted before correctness is proven
- **User Story 3 (Phase 5)**: Depends on US2 (T016)
- **Polish (Phase 6)**: Depends on US3

### Within Each User Story

- US1: T007 (add case) → T008 (tiled run) / T009 (coopmat run), sequential
  since both reuse the same rebuilt binary; T010 (SPIR-V) is independent of
  T008/T009 and can run in parallel once T005's build exists
- US2: T012 (author harness) → T013 (build) → T014 (prove on 1B) → T015
  (remaining 2 models), strictly sequential — each step depends on the
  previous compiling/running correctly, and all are GPU-bound

### Parallel Opportunities

- T002, T003 (Foundational readiness checks) can run in parallel —
  independent resources, no shared state
- T010 (SPIR-V inspection) can run in parallel with T008/T009 — it inspects
  an already-compiled artifact, not live dispatch behavior
- T022 (Polish) has no dependency on T021 and could run alongside it
- **Never** parallelize T008/T009/T014/T015/T021 with each other or any
  other GPU-bound task — they share the MiniPC's one GPU, matching this
  workstream's established discipline

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational — **including the regression-baseline
   hard gate (T006)**
3. Complete Phase 3: User Story 1 — correctness and dispatch proven (or the
   failure fully root-caused) for both coopmat shaders at one shape
4. **STOP and VALIDATE**: the SDPA coopmat path is trustworthy before
   spending device time measuring it across all three models

### Incremental Delivery

1. Setup + Foundational → `op_tests` builds, existing suite passes
2. US1 → SDPA coopmat proven correct and dispatch-confirmed
3. US2 → all three models measured (or explicitly excluded)
4. US3 → the actual answer: does SDPA coopmat help real prefill attention?

---

## Notes

- No commits until the user explicitly asks, per repo convention.
- `test_coopmat_attention_bench.cpp` is deliberately untouched throughout —
  confirmed to test the unrelated matmul tile-sweep shader family, not
  `sdpa_compute_attn_weights_coopmat`/`sdpa_compute_out_coopmat` (research.md
  Decision 1). Do not add it back into scope without revisiting that finding.
- T011 is a contingency, not an expected outcome — research.md Decision 5's
  arithmetic and the code's own existing design suggest T009 should pass,
  but it must actually be run and checked, not assumed.
- If T011 is triggered, any resulting production fix follows this
  workstream's established discipline: proposed during (retroactive)
  planning, applied only with explicit user authorization, documented at
  the point it's made.
- FR-006's exclusion path (T016) is expected to find zero excluded models
  per research.md Decision 5's shape-alignment computation — but this is a
  confirmation to make, not a result to assume going in.
