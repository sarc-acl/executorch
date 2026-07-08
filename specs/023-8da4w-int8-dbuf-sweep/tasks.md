---

description: "Task list for 8da4w Int8 WMMA Double-Buffer Variant Sweep"
---

# Tasks: 8da4w Int8 WMMA Double-Buffer Variant Sweep

**Input**: Design documents from `/specs/023-8da4w-int8-dbuf-sweep/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md (all present; no contracts/ — internal experimental feature)

**Tests**: No separate unit-test tasks — this feature's own deliverable *is* verification
(correctness-pass + dispatch-confirm per variant, per constitution Principle I/VI), so those
checks are embedded directly as User Story 1 implementation tasks, not an optional add-on.

**Organization**: Tasks are grouped by user story (spec.md P1/P2/P3) for independent
implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies). On-device run/measurement
  tasks are deliberately **not** marked [P] even when they target different files: M5 EVT1
  is shared, single hardware (constitution Principle II / project memory on device sharing)
  — only one process should be driving it at a time regardless of file independence.
- **[Story]**: US1, US2, or US3, mapping to spec.md's three user stories.

## Path Conventions

All paths are relative to the new worktree's `executorch/` checkout (see T003), identical
in layout to this checkout (`backends/vulkan/...`), except `specs/023-8da4w-int8-dbuf-sweep/`
paths, which are authored here and committed before the worktree is created (T002).

---

## Phase 1: Setup

**Purpose**: Confirm device availability and stand up the dedicated worktree per the user's
explicit "on a new worktree" instruction (research.md Decision 7).

- [ ] T001 Confirm M5 EVT1 is free (project memory: shared device, don't assume) and read
      `.shared-context/instruction-for-ai/README.md` §Conventions for current
      serial/host/NFS/clock defaults (constitution Principle X)
- [ ] T002 Commit `specs/023-8da4w-int8-dbuf-sweep/{spec.md,plan.md,research.md,data-model.md,quickstart.md,tasks.md}`
      to the `quant-perf-optimization` branch in this checkout, so the new worktree's
      checkout will include them
- [ ] T003 Create the new worktree: `git worktree add <new-dir> -b 023-8da4w-int8-dbuf-sweep-impl quant-perf-optimization`
      from `/local/yanwen.xu/workspace` (research.md Decision 7)
- [ ] T004 Bootstrap the new worktree in `<new-dir>/executorch`: `uv venv .venv --seed`,
      `source .venv/bin/activate`, `./install_executorch.sh --minimal` (constitution
      "Environment & Build Bootstrap")

**Checkpoint**: new worktree exists, is bootstrapped, and contains this feature's spec docs.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared dispatch/harness/tooling infrastructure that every user story depends
on. No variant-specific work starts until this phase is complete.

**⚠️ CRITICAL**: Phase 3+ cannot begin until this phase is complete.

- [ ] T005 Re-verify the on-device Vulkan driver identity (constitution Principle VIII;
      `.shared-context/instruction-for-ai/devices-and-access.md` / `ACTIVE-STATUS.md`) in
      the new worktree's environment — do not assume a prior session's driver still holds
- [ ] T006 Pin GPU/MIF/INT clocks per `.shared-context/instruction-for-ai/README.md`
      §Conventions and confirm the pin bound via a GFLOP/s cross-check (constitution
      Principle VII) — required before any timed measurement in Phase 4/5
- [ ] T007 Add an `ET_VK_DQ8CA_COOPMAT_VARIANT` env-var-gated branch to the existing
      `linear_dq8ca_q4gsw_coopmat` kernel-name selection in
      `backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp`, mirroring the existing
      `ET_VK_Q4GSW_COOPMAT_VARIANT` pattern (research.md Decisions 1 & 3) — unset env var
      MUST preserve today's shipped `dbuf4` default dispatch exactly (spec FR-001)
- [ ] T008 [P] Extend `backends/vulkan/test/custom_ops/test_coopmat_linear_bench.cpp` (or a
      sibling bench file, if extending in place is impractical) to cover the 6-shape
      `dq8ca_q4gsw_coopmat` catalog (`wq` + `w1_gate` x {1B, 3B, 8B}, per spec
      Clarifications) with kernel-name logging on each dispatch, mirroring the existing
      fp16 dbuf1-4 bench's own kernel-name capture (research.md Decision 6)
- [ ] T009 [P] Write `specs/023-8da4w-int8-dbuf-sweep/scripts/run_dbuf_sweep.sh`: invokes
      the Phase 2/T008 bench binary once per `ET_VK_DQ8CA_COOPMAT_VARIANT` value (one
      process per variant, research.md Decision 2), records each invocation's exit code,
      and marks a `pipeline_crash` result (with detail) for any non-zero exit instead of
      aborting the remaining variants
- [ ] T010 Confirm the existing `dq8ca_q4gsw` correctness suite (`test_*_linear` /
      `backends/vulkan/test/op_tests`) can be pointed at one specific variant via
      `ET_VK_DQ8CA_COOPMAT_VARIANT` and yields an unambiguous per-variant pass/fail signal
      (depends on T007)

**Checkpoint**: dispatch hook, bench harness, driver script, and correctness-check
invocation are all in place — User Story 1 can now start.

---

## Phase 3: User Story 1 - Prove each variant builds and runs correctly (Priority: P1) 🎯 MVP

**Goal**: All four dbuf loop structures exist as opt-in, env-var-selected production-graph
variants of the int8 `8da4w` coopmat shader (default dispatch unchanged when unset), each
confirmed to compile, dispatch the coopmat kernel (not a fallback), and pass the existing
correctness check.

**Independent Test**: for each of the 4 variants, a build exists and is confirmed (via
tooling) to dispatch coopmat and pass correctness — per spec.md User Story 1's own
Independent Test.

### Implementation for User Story 1

- [ ] T011 [P] [US1] Port the dbuf4 ("store-first", already-shipped) loop structure into
      `backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_q4gsw_coopmat_dbuf4.{glsl,yaml}`
      as a faithful copy of the production `linear_dq8ca_qw_coopmat.glsl`'s loop structure,
      for an apples-to-apples in-sweep baseline (plan.md Project Structure)
- [ ] T012 [P] [US1] Port the dbuf1 ("prefetch-first") loop structure from
      `.shared-context/reference-codes/shmem_double_buf.comp` into
      `backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_q4gsw_coopmat_dbuf1.{glsl,yaml}`,
      re-deriving it against the int8 shader's nested groups x chunks loop and second
      wsum/wsc ping-pong pair (research.md Decision 4) — document any new Xclipse PAL
      workaround inline per constitution Principle V
- [ ] T013 [P] [US1] Port the dbuf2 ("store-first", non-peeled) loop structure from
      `.shared-context/reference-codes/shmem_double_buf2.comp` into
      `backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_q4gsw_coopmat_dbuf2.{glsl,yaml}`,
      same adaptation approach and documentation requirement as T012
- [ ] T014 [P] [US1] Port the dbuf3 (peeled, no-conditional-epilog) loop structure from
      `.shared-context/reference-codes/shmem_double_buf3.comp` into
      `backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_q4gsw_coopmat_dbuf3.{glsl,yaml}`,
      same adaptation approach and documentation requirement as T012 — this is the
      shader variant the dbuf3-faster-for-int8 hypothesis is actually about
- [ ] T015 [US1] Build all four variants for Android per
      `.shared-context/instruction-for-ai/build.md`'s two-step recipe (core runtime +
      `--target install`, then the T008 bench target); confirm each pipeline compiles
      without a driver crash (depends on T007, T008, T011, T012, T013, T014)
- [ ] T016 [US1] Run `run_dbuf_sweep.sh --correctness-only dbuf1` on M5 EVT1; confirm the
      kernel-name log shows `linear_dq8ca_q4gsw_coopmat_dbuf1` (not a fallback) and the
      T010 correctness check passes; record `compiles`/`dispatches_coopmat`/
      `correctness_passed`/`failure_reason` per data-model.md's Double-Buffer Variant
      (depends on T009, T010, T015)
- [ ] T017 [US1] Same as T016 for `dbuf2` (depends on T009, T010, T015)
- [ ] T018 [US1] Same as T016 for `dbuf3` (depends on T009, T010, T015)
- [ ] T019 [US1] Same as T016 for `dbuf4` (depends on T009, T010, T015)
- [ ] T020 [US1] Disassemble each variant's compiled SPIR-V (`spirv-dis`/`spirv-cross` or
      equivalent) and confirm genuine int8 cooperative-matrix instructions
      (`OpCooperativeMatrixMulAddKHR` on 8-bit component types) are present, setting
      `spirv_verified` per data-model.md for each variant that passed T016-T019
      (research.md Decision 6)

**Checkpoint**: User Story 1 (MVP) complete — every variant's validity is proven or its
`failure_reason` is recorded; no untrusted timing has been taken yet.

---

## Phase 4: User Story 2 - Measure all four variants across representative shapes (Priority: P2)

**Goal**: every correctness-verified variant has a 3-run mean + CoV execution time for each
of the 6 representative shapes, with pinned/verified clocks.

**Independent Test**: per spec.md User Story 2's own Independent Test — a 3-run mean + CoV
exists per (variant, shape) pair for every variant that passed Phase 3.

### Implementation for User Story 2

- [ ] T021 [US2] Re-verify the on-device driver identity and re-confirm the clock pin is
      bound (constitution Principles VII/VIII apply "before every coopmat measurement," not
      just once) if any session gap occurred since T005/T006 — otherwise explicitly record
      that Phase 3 and Phase 4 ran within one continuous, already-verified session and this
      is a no-op check
- [ ] T022 [US2] Run the timed sweep for `dbuf1` across all 6 shapes (3 runs each) via
      `run_dbuf_sweep.sh dbuf1`; record each `Timing Result` (mean_us, cov,
      clock_pin_verified, driver_verified) per data-model.md (depends on T016, T020, T021)
      — skip entirely if T016 recorded a `failure_reason`
- [ ] T023 [US2] Same as T022 for `dbuf2` (depends on T017, T020, T021)
- [ ] T024 [US2] Same as T022 for `dbuf3` (depends on T018, T020, T021)
- [ ] T025 [US2] Same as T022 for `dbuf4` (depends on T019, T020, T021) — this run doubles
      as the in-sweep `dbuf4` production-baseline measurement (spec SC-004)
- [ ] T026 [US2] Save all raw sweep output (per-run timings, kernel-name logs, exit codes)
      under `specs/023-8da4w-int8-dbuf-sweep/results/raw/`, matching this workstream's
      existing raw-log convention (depends on T022, T023, T024, T025)

**Checkpoint**: User Stories 1 AND 2 complete — every `Timing Result` in data-model.md
exists (or a variant's `failure_reason` explains why not).

---

## Phase 5: User Story 3 - Report the fastest variant for int8 (Priority: P3)

**Goal**: a report names the fastest variant (per shape and overall), confirms/refutes the
dbuf3 hypothesis, and states the margin vs. the shipped dbuf4 baseline.

**Independent Test**: per spec.md User Story 3's own Independent Test — the report alone
answers all three questions, backed only by numbers already in it.

### Implementation for User Story 3

- [ ] T027 [US3] Compute the fastest variant per shape and overall (or "varies by shape" if
      the data doesn't support a single winner) from the T026 `Timing Result` set, per
      data-model.md's Sweep Report (depends on T026)
- [ ] T028 [US3] State the dbuf3-is-faster-for-int8 hypothesis verdict (confirmed/refuted)
      with the specific `Timing Result` numbers that support it (depends on T027)
- [ ] T029 [US3] Compute the fastest variant's margin (percentage or factor) over the T025
      in-sweep `dbuf4` baseline measurement (depends on T025, T027)
- [ ] T030 [US3] Write `specs/023-8da4w-int8-dbuf-sweep/results/m5-dq8ca-dbuf-sweep-report.md`
      synthesizing T027-T029, plus every variant with a `failure_reason` from Phase 3
      (spec FR-004/SC-001), so the report alone satisfies spec SC-001 through SC-005
      (depends on T028, T029)

**Checkpoint**: all three user stories complete and independently verifiable from the
report alone.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T031 [P] Walk through `quickstart.md` end-to-end in the new worktree and correct any
      step that didn't reproduce as documented
- [ ] T032 [P] If a new Xclipse PAL compiler workaround was discovered while porting
      dbuf1/2/3 (T012-T014) that isn't already covered by an existing entry, append it to
      `.specify/memory/gotchas.md` per that file's own append convention (constitution
      Principle V / Development Workflow "Gotchas Reference")
- [ ] T033 Commit all changes in the new worktree (shader variants, `QuantizedLinear.cpp`
      hook, bench harness extension, driver script, raw results, report) — do not push
      without explicit confirmation, per this workstream's standing discipline on
      shared-state actions

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies — start immediately
- **Foundational (Phase 2)**: depends on Setup (needs the bootstrapped worktree) — BLOCKS
  all user stories
- **User Story 1 (Phase 3)**: depends on Foundational completion
- **User Story 2 (Phase 4)**: depends on User Story 1 (needs each variant's
  correctness/dispatch/SPIR-V verification before its timing is trustworthy) — not
  independent of US1 the way a typical CRUD feature's stories would be, because spec.md
  itself makes US2 depend on US1's proof (see spec.md User Story 2 "Why this priority")
- **User Story 3 (Phase 5)**: depends on User Story 2 (needs the full `Timing Result` set)
- **Polish (Phase 6)**: depends on all three user stories

### Within Each User Story

- T011-T014 (shader ports) are independent file edits — parallelizable
- T015 (build) depends on all four ports
- T016-T019 (per-variant dispatch/correctness runs) depend on T015 and are **not**
  parallel with each other (shared M5 EVT1 hardware)
- T020 (SPIR-V) depends on T016-T019
- T021 (re-verify driver/clock) gates all of Phase 4's timing runs
- T022-T025 (timed sweeps) depend on their respective variant's T020 completion plus T021,
  and are **not** parallel with each other (shared hardware)

### Parallel Opportunities

- T008 and T009 (Foundational: bench harness extension, driver script) touch different
  files and can be done in parallel
- T011, T012, T013, T014 (the four shader ports) touch different files and can be done in
  parallel
- T031 and T032 (Polish) are independent and can be done in parallel

---

## Parallel Example: User Story 1 shader ports

```bash
Task: "Port dbuf4 into linear_dq8ca_q4gsw_coopmat_dbuf4.{glsl,yaml}"
Task: "Port dbuf1 into linear_dq8ca_q4gsw_coopmat_dbuf1.{glsl,yaml}"
Task: "Port dbuf2 into linear_dq8ca_q4gsw_coopmat_dbuf2.{glsl,yaml}"
Task: "Port dbuf3 into linear_dq8ca_q4gsw_coopmat_dbuf3.{glsl,yaml}"
```

All four are independent file edits (T011-T014); the on-device runs that depend on them
(T016-T019) are sequential, not parallel, per the shared-hardware note above.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (new worktree)
2. Complete Phase 2: Foundational (dispatch hook, bench harness, driver script,
   correctness-check integration)
3. Complete Phase 3: User Story 1 — all four variants proven to compile, dispatch, and
   pass correctness
4. **STOP and VALIDATE**: confirm every variant in data-model.md's Double-Buffer Variant
   set has either full validity or a stated `failure_reason` — this alone already answers
   "is a dbuf-variant sweep on this shader even feasible," independent of any timing result

### Incremental Delivery

1. Setup + Foundational → environment ready
2. User Story 1 → MVP: know which variants are even valid to time
3. User Story 2 → all valid variants timed across all 6 shapes
4. User Story 3 → the report that actually answers the dbuf3 hypothesis
5. Polish → quickstart re-validated, any new gotcha recorded, work committed

---

## Notes

- [P] tasks = different files, no dependencies — except on-device run tasks, which are
  never marked [P] regardless of file independence (shared M5 EVT1 hardware).
- Every task that runs on-device MUST have re-verified the driver and confirmed the clock
  pin before it is trusted: T005/T006 cover Phase 3, T021 explicitly re-covers Phase 4/5
  per constitution Principles VII/VIII's "before every measurement" wording.
- A variant that fails T016-T019 or T020 is not deleted from scope — it flows into T030's
  report as an explicit failure, per spec FR-004.
