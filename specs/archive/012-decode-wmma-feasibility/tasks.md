---

description: "Task list for Decode Shader WMMA Acceleration"
---

# Tasks: Decode Shader WMMA Acceleration

**Input**: Design documents from `/specs/012-decode-wmma-feasibility/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md (all present)

**Tests**: Not requested as a separate automated suite for the expected
(roofline-only) path -- there is no new code to test in that path, only a
calculation to verify against its cited sources. If the contingent User
Story 2/3 path executes, it follows `010`'s established correctness
-harness discipline (CPU/ATen reference, SPIR-V inspection) as its own
verification step.

**Organization**: Tasks are grouped by user story. Unlike every other
feature in this workstream, this one's primary deliverable (User Story 1)
is a citable calculation, not a device capture or new code -- planning
already performed this calculation (research.md Decisions 1-3) with a
decisive result (12-50x margin, bandwidth-bound). User Story 2 and User
Story 3 are retained as the contingent path FR-004 requires, gated on User
Story 1's verdict, and are **not expected to execute**.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files/resources, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Paths are relative to the repository root

## Path Conventions

- `backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coop.glsl` (+ included `.glslh`) — existing kernel, read-only reference for US1
- `specs/012-decode-wmma-feasibility/results/decode-wmma-feasibility-report.md` — the feature's deliverable
- Contingent path only (not expected): new shader `.glsl`/`.yaml` under `backends/vulkan/runtime/graph/ops/glsl/`, new correctness case under `backends/vulkan/test/op_tests/`, new benchmark harness under `backends/vulkan/test/custom_ops/`

---

## Phase 1: Setup

- [X] T001 Create `specs/012-decode-wmma-feasibility/results/` directory

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Confirm the inputs the roofline calculation depends on are
real and current -- there is no code to write, only grounding to verify
(mirroring `007`'s and `011`'s Foundational phases, the closest precedents
for a feature with little/no new code).

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T002 [P] Confirm this device's identity via `vulkaninfo --summary` (`AMD Radeon 780M Graphics (RADV PHOENIX)`) and `/proc/cpuinfo` (`AMD Ryzen 9 7940HS`) -- the basis for research.md Decision 1's published-spec lookup -- reconfirmed identical to planning
- [X] T003 [P] Confirm `dmidecode`/`lshw` remain inaccessible without sudo on this box (re-run the same check from planning) -- if access has since become available, note it as a potential refinement to Decision 1's RAM-speed assumption, but do not block on it (spec.md Assumptions: published specs are a safe, conservative default) -- reconfirmed still inaccessible (`Permission denied` on `/sys/firmware/dmi/tables/smbios_entry_point`); published-spec assumption stands
- [X] T004 [P] Re-read `backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coop.glsl` and its included `linear_int4_weight_tile_load.glslh`/`linear_fp_weight_scales_load.glslh` to reconfirm the weight-packing format (4-bit packed + per-group fp16 scale) underlying research.md Decision 2's arithmetic-intensity calculation -- reconfirmed: `t_packed_int4_weight` + `t_weight_scales` (per-`group_size` fp16 scale), unchanged from planning

**Checkpoint**: Foundation ready — device identity, RAM-speed constraint, and kernel weight-format all reconfirmed present and current

---

## Phase 3: User Story 1 - Determine whether decode is actually a WMMA opportunity before building anything (Priority: P1) 🎯 MVP

**Goal**: Produce a citable, source-backed roofline finding: is decode's
linear GEMV kernel compute-bound, memory-bandwidth-bound, or ambiguous on
this device.

**Independent Test**: Confirm the roofline finding's peak-spec numbers,
kernel-intensity numbers, and resulting verdict are each traceable to a
cited source (published device spec or the kernel's own source code), not
asserted without support.

- [X] T005 [US1] Document the device peak-spec figures (17.8 TFLOPS peak FP16 compute, 89.6 GB/s peak memory bandwidth) with their sources (research.md Decision 1) in `specs/012-decode-wmma-feasibility/results/decode-wmma-feasibility-report.md` (depends on Foundational checkpoint)
- [X] T006 [US1] Document the machine balance point calculation (peak compute ÷ peak bandwidth ≈ 198.7 FLOPs/byte) in the report (depends on T005)
- [X] T007 [US1] Document the kernel's arithmetic-intensity calculation (4.0 FLOPs/byte base, 16.0 FLOPs/byte generous-dequant-overhead estimate, research.md Decision 2) in the report, citing the exact source lines confirmed in T004 (depends on T004)
- [X] T008 [US1] State the roofline comparison and verdict in the report: 12-50x margin below the balance point → `bandwidth_bound` (FR-002) (depends on T006, T007) — done: verdict is `bandwidth_bound`, 50x margin (base) / 12x margin (generous)
- [X] T009 [US1] Per FR-003: since T008's verdict is `bandwidth_bound`, document the recommendation (more aggressive weight quantization; batching/speculative decoding for a real `M>1` opportunity — research.md Decision 4) in the report, and explicitly state that User Story 2/3 are not attempted for this reason (depends on T008)

**Checkpoint**: US1 complete — the Roofline Finding is fully documented and
cited; the feature's central question is answered

---

## Phase 4: User Story 2 - Build and correctness-prove a WMMA-capable decode shader for linear GEMV, if warranted (Priority: P2)

**Goal**: Only if User Story 1's verdict is `compute_bound` or `ambiguous`
(FR-004): design a new cooperative-matrix decode shader and prove its
correctness before any performance claim.

**Independent Test**: The new shader's output matches a CPU/ATen reference
at a small tile-aligned shape, and its SPIR-V contains genuine
cooperative-matrix instructions.

- [X] T010 [US2] Check the gate: if T008's verdict is `bandwidth_bound` (expected), mark this entire phase "not attempted -- see Roofline Finding" in the report and skip to Phase 6; otherwise proceed (depends on T009) — gate triggered: verdict is `bandwidth_bound`, report already states Phase 4 as not attempted; skipping T011-T013
- [X] T011 [US2] *(contingent, not expected)* Design a small, tile-aligned decode-shaped test case and implement a new cooperative-matrix-capable shader for the linear GEMV path, following `010`'s established shader-design discipline — **SKIPPED**: T010's gate resolved to not-attempted, per FR-003
- [X] T012 [US2] *(contingent, not expected)* Add a correctness case comparing the new shader's output against a CPU/ATen reference, `010`-style tolerance, in `backends/vulkan/test/op_tests/` — **SKIPPED**: same gate
- [X] T013 [US2] *(contingent, not expected)* Disassemble the new shader's SPIR-V and confirm genuine `OpCooperativeMatrix*KHR` instructions — **SKIPPED**: same gate

**Checkpoint**: US2 complete (explicitly skipped per T010's gate)

---

## Phase 5: User Story 3 - Measure whether the new decode shader actually speeds up real decode, per target model (Priority: P3)

**Goal**: Only if User Story 2 executed and passed correctness: benchmark
the new shader against the existing decode kernel at each target model's
real per-token shape.

**Independent Test**: One target model's existing-kernel-vs-new-shader
timing pair, with iteration count and variance.

- [X] T014 [US3] Check the gate: if Phase 4 was skipped (expected), mark this entire phase "not attempted -- see Roofline Finding" in the report and skip to Phase 6; otherwise proceed (depends on T010 or T013) — gate triggered: Phase 4 was skipped, report already states Phase 5 as not attempted; skipping T015-T016
- [X] T015 [US3] *(contingent, not expected)* Benchmark the new shader against `linear_q4gsw_coop` at each of the three target models' real decode shape, `007`/`010`-style (iteration count, variance, dispatch confirmed) — **SKIPPED**: T014's gate resolved to not-attempted
- [X] T016 [US3] *(contingent, not expected)* State a per-model verdict (real speedup / no meaningful difference / regression) and whether it's consistent with US1's roofline prediction (FR-007) — **SKIPPED**: same gate

**Checkpoint**: US3 complete (explicitly skipped per T014's gate)

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T017 Reproducibility spot-check: independently recompute the roofline comparison (T005-T008) from its cited sources and confirm the same verdict — matching `001`'s established reproducibility discipline, applied here to a calculation rather than a device capture — reproduced exactly: 198.7 FLOPs/byte balance point, 50x/12x margins, `bandwidth_bound` verdict
- [X] T018 [P] Update `quickstart.md` with any corrections found during T002-T017 (if any were needed) — no corrections needed; planning's procedure matched implementation exactly (device identity, sudo inaccessibility, and kernel weight-format all reconfirmed identical)
- [X] T019 Write the final overall statement in the report: whether WMMA acceleration is worth pursuing for decode on this device, and the recommended alternative (SC-004) (depends on T009, and T016 if the contingent path executed) — done: "WMMA acceleration is not worth pursuing for decode's linear GEMV kernel on this device," with the two recommended alternatives named

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup. BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational checkpoint
- **User Story 2 (Phase 4)**: Gated on US1's verdict (T010) -- expected to
  be skipped entirely
- **User Story 3 (Phase 5)**: Gated on US2 having executed (T014) --
  expected to be skipped entirely
- **Polish (Phase 6)**: Depends on US1 (always) and US2/US3 (only if they
  executed)

### Within Each User Story

- US1: T005 → T006/T007 (can proceed in parallel with each other, both
  depend only on T005's grounding) → T008 → T009, strictly sequential
  after the parallel pair
- US2/US3: Each phase's gate check (T010/T014) must resolve before any
  contingent task in that phase is attempted

### Parallel Opportunities

- T002, T003, T004 (Foundational checks) can all run in parallel --
  independent verifications, no shared state
- T006 and T007 (balance point vs. kernel intensity) can proceed in
  parallel once T005/T004 are done -- independent calculations
- T018 (Polish) has no dependency on T017 and could run alongside it

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1 — the Roofline Finding, fully documented
4. **STOP and VALIDATE**: Given the decisive 12-50x margin already found
   during planning, this is expected to be the feature's actual stopping
   point -- User Story 2/3 are gated to skip automatically (T010, T014)

### Incremental Delivery

1. Setup + Foundational → grounding reconfirmed
2. US1 → the roofline finding, cited and documented — the feature's answer
3. US2/US3 → gated contingent path, expected to be skipped
4. Polish → reproducibility spot-check of the calculation itself, and the
   final overall statement

---

## Notes

- No commits until the user explicitly asks, per repo convention.
- This feature is expected to conclude with **no new shader code at all**
  -- that is a complete, valid, successful outcome (spec.md Assumptions),
  not a failure requiring US2/US3 to be forced through regardless of T010/
  T014's gate.
- If T010 or T014's gate check ever finds the roofline finding was wrong
  (e.g. a future re-read of the kernel finds a materially different
  weight format), that is itself a significant, reportable finding
  requiring the same root-cause-then-authorize discipline as this
  workstream's other mid-implementation discoveries -- not something to
  silently route around either direction.
