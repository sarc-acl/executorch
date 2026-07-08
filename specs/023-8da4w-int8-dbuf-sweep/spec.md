# Feature Specification: 8da4w Int8 WMMA Double-Buffer Variant Sweep

**Feature Branch**: `023-8da4w-int8-dbuf-sweep`

**Created**: 2026-07-07

**Status**: Draft

**Input**: User description: "for all the 4 versions of the double buffer implementations
(`.shared-context/reference-codes/shmem_double_buf{,2,3,4}.comp`), the assumption is that
version 1 is fastest for FP16 WMMA instruction. I have a hypothesis that version 3 (dbuf3)
is faster for int8 instructions. On a new worktree, Modify the 8da4w shaders (int8 WMMA)
to try each double buffer implementation, and determine the fastest shader for int8."

## Context

The prior dbuf1-dbuf4 loop-structure sweep (`report-for-human/dbuf-sweep-q4gsw-m2048.md`)
was run against the **fp16** WMMA linear shader (`linear_qw_coopmat.glsl`) and found dbuf1
("prefetch-first") 1.87x faster than the shipped baseline on M5 EVT1 -- that shader now
ships with dbuf1. The **int8** WMMA linear shader used by the `8da4w` quantization scheme
(`linear_dq8ca_qw_coopmat.glsl`, dispatched for `WEIGHT_NBITS=4` as
`linear_dq8ca_q4gsw_coopmat`) currently ships with **dbuf4** ("store-first", per its own
header comment referencing `shmem_double_buf4.comp`) -- a choice its header attributes to
LDS-layout and per-group ping-pong constraints, not to a measured dbuf1-vs-dbuf4 comparison
for int8. No dbuf sweep has ever been run against the int8 shader itself. This feature runs
that sweep, testing the hypothesis that dbuf3 outperforms the other three variants for int8
WMMA, and identifies which of the four loop structures is actually fastest for this shader
on the M5 EVT1 target.

## Clarifications

### Session 2026-07-07

- Q: What shape/model coverage breadth should the sweep use? → A: Curated subset — wq + w1_gate per model = 6 shapes per variant (matches `specs/008`'s default sweep-phase set).
- Q: Is the linear-op microbenchmark (User Story 2) sufficient to declare a variant "fastest", or must the winner also be validated end-to-end (model-level tok/s)? → A: Microbenchmark-only — no e2e validation is required by this feature.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Prove each variant builds and runs correctly (Priority: P1) 🎯 MVP

As the contributor running this workstream, I want all four double-buffer loop structures
ported onto the `8da4w` int8 coopmat linear shader as separate, opt-in-selectable variants,
each confirmed to compile, dispatch the int8 coopmat kernel (not a silent fallback), and
pass the existing correctness check, so that a timing comparison between them is
trustworthy before any device time is spent measuring performance.

**Why this priority**: the int8 shader's own header already documents that this exact loop
structure is fragile on the Xclipse PAL compiler (flattening the group/chunk loop crashes
`vkCreateComputePipelines` at large trip counts) -- porting a different double-buffer
variant onto it is exactly the kind of change that could silently fail to compile, silently
fall back to a non-coopmat path, or silently produce wrong numbers. Proving all four
variants are genuinely valid comes before trusting any timing from them.

**Independent Test**: for each of the 4 variants, a test-only shader build exists, is
confirmed (via tooling, not assumption) to dispatch the int8 coopmat kernel, and passes the
correctness check already used for the shipped `8da4w` coopmat shader.

**Acceptance Scenarios**:

1. **Given** the four reference loop structures in
   `.shared-context/reference-codes/shmem_double_buf{,2,3,4}.comp`, **When** each is ported
   onto the int8 `8da4w` coopmat shader as an opt-in, env-var-selected variant, **Then**
   all four variants compile, and the *default* dispatch behavior (env var unset) is
   unchanged from what ships today.
2. **Given** a compiled variant, **When** it is run once, **Then** tooling confirms it
   dispatched the int8 coopmat kernel and its output passes the existing correctness check
   for the `8da4w` linear op.
3. **Given** a variant that fails to compile, fails to dispatch coopmat, or fails
   correctness, **When** the sweep proceeds, **Then** that failure does not abort or corrupt
   the results for the other three variants.

---

### User Story 2 - Measure all four variants across representative shapes (Priority: P2)

As the contributor running this workstream, I want each correctness-verified variant timed
across a representative set of `8da4w` linear shapes on M5 EVT1 with pinned, verified
clocks, so the comparison between variants is statistically sound and not noise.

**Why this priority**: this is the actual measurement the feature exists to produce; it
depends on User Story 1 having already proven every variant it measures is valid.

**Independent Test**: for each of the (correctness-verified) variants, a 3-run mean + CoV
execution time exists for each shape in the representative shape set, captured with
pinned clocks whose pin is verified bound and the on-device driver identity re-verified
beforehand.

**Acceptance Scenarios**:

1. **Given** the correctness-verified variants from User Story 1, **When** each is measured
   across the representative shape set on M5 EVT1, **Then** every measurement is a 3-run
   mean with CoV, not a single untimed sample.
2. **Given** the measurement run, **When** it is captured, **Then** clock pinning is
   verified bound and the on-device driver identity is re-verified before measuring, per
   this workstream's standing discipline.

---

### User Story 3 - Report the fastest variant for int8 (Priority: P3)

As a reader deciding whether the shipped `8da4w` coopmat shader should change its
double-buffer loop structure, I want a report stating which of the four variants is fastest
for int8 WMMA (overall and per shape), how it compares to the currently-shipped dbuf4
baseline, and whether the dbuf3-is-faster-for-int8 hypothesis is confirmed or refuted, so
that decision can be made from evidence instead of assumption.

**Why this priority**: the raw measurements from User Story 2 don't answer the question
this feature was created to answer until they're synthesized into a stated conclusion.

**Independent Test**: open the report and confirm it names a fastest variant (or states
that no single variant wins across all shapes), states the hypothesis verdict, and states
the fastest variant's margin over the shipped dbuf4 baseline -- all supported only by
numbers already in the report.

**Acceptance Scenarios**:

1. **Given** all measurements from User Story 2, **When** the report is produced, **Then**
   it states the fastest variant per shape and overall (or explicitly that the winner
   varies by shape, if that is what the data shows).
2. **Given** the report's stated results, **When** it addresses the dbuf3 hypothesis,
   **Then** it explicitly confirms or refutes it with the supporting numbers, not just a
   restatement of the raw table.
3. **Given** the report's fastest variant, **When** it is compared to the shipped dbuf4
   baseline, **Then** the report states the measured difference (percentage or factor).

---

### Edge Cases

- What if a variant crashes the Xclipse PAL compiler (the exact failure mode the shipped
  shader's own header warns about for a flattened loop)? -- must be caught by process-level
  isolation (this workstream's established `specs/008` precedent: an in-process crash
  cannot be caught by `try`/`catch`), and reported as an explicit build failure, not
  silently dropped or allowed to erase other variants' results.
- What if a variant compiles and dispatches but fails the correctness check? -- reported as
  a correctness failure with no timing number attached, not presented as a valid result.
- What if the fastest variant differs by shape (e.g. dbuf3 wins for one op but dbuf1 wins
  for another)? -- the report must state this explicitly rather than forcing a single
  overall winner the data doesn't support.
- What if M5 EVT1 isn't free when this work is attempted, or the driver has changed since
  the last session? -- confirm device availability and re-verify driver identity before
  measuring, per this workstream's standing discipline; don't assume continuity from a
  prior session.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: This feature MUST produce four variants of the `8da4w` int8 coopmat linear
  shader (`linear_dq8ca_qw_coopmat.glsl` / dispatched as `linear_dq8ca_q4gsw_coopmat`), one
  per double-buffer loop structure defined in
  `.shared-context/reference-codes/shmem_double_buf.comp`, `shmem_double_buf2.comp`,
  `shmem_double_buf3.comp`, and `shmem_double_buf4.comp`. Per this workstream's own
  constitution (reuse the existing dbuf1-4 harness rather than re-deriving one), these are
  built as opt-in, env-var-selected production-graph variants -- the same pattern already
  proven for the fp16 `4w` shader's own dbuf1-4 sweep -- not as fully separate test-only
  ops. This MUST NOT change the shader's or dispatch code's *default* behavior: with the
  selector env var unset, `linear_dq8ca_q4gsw_coopmat` dispatches exactly as it does today
  (shipped `dbuf4`).
- **FR-002**: Each variant MUST be confirmed, via tooling, to dispatch the int8 coopmat
  kernel (not a silent tiled or scalar fallback) before any timing from it is trusted.
- **FR-003**: Each variant MUST pass the existing correctness check for the `8da4w`
  (`dq8ca`/`q4gsw`) linear op before its timing is included in the measurement set.
- **FR-004**: A variant that fails to compile, fails to dispatch coopmat, or fails
  correctness MUST still appear in the final report with an explicit failure reason -- none
  are silently omitted.
- **FR-005**: This feature MUST measure each correctness-verified variant's execution time
  across a representative set of `8da4w` linear shapes on M5 EVT1 (this workstream's active
  performance target), with clocks pinned and the pin verified bound, reporting a 3-run
  mean with CoV for each variant/shape pair.
- **FR-006**: This feature MUST re-verify the on-device driver identity before measuring,
  per this workstream's standing discipline -- not assume a prior session's driver state
  still holds.
- **FR-007**: This feature MUST produce a report identifying the fastest variant for int8
  WMMA, per shape and overall (or explicitly stating that the winner varies by shape),
  comparing it against the currently-shipped dbuf4 production baseline, and explicitly
  confirming or refuting the hypothesis that dbuf3 is fastest for int8.

### Key Entities

- **Double-Buffer Variant**: one of four loop structures (dbuf1-dbuf4), each identified by
  its reference source file. Fields: variant id, source reference, compiles (bool),
  dispatches_coopmat (bool), correctness_passed (bool), failure reason (if any), timing
  results per shape (3-run mean + CoV).
- **8da4w Linear Shape**: a representative (K, N, group_size) combination drawn from this
  workstream's existing model/op catalog. Fixed to the `wq` and `w1_gate` ops for each of
  the three target models (1B, 3B, 8B) -- 6 shapes total per variant, matching
  `specs/008-8da4w-parameter-sweep`'s curated sweep-phase set -- used to measure each
  variant.
- **Sweep Report**: the synthesized conclusion -- fastest variant per shape, fastest
  variant overall (or "varies by shape"), hypothesis verdict (confirmed/refuted, with
  numbers), and comparison against the shipped dbuf4 baseline.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All four double-buffer variants are attempted, and each appears in the final
  report as either measured or explicitly failed with a stated reason -- none are silently
  missing.
- **SC-002**: The report states which variant is fastest for int8 WMMA, both overall and
  per representative shape, with every timing claim backed by a 3-run mean and CoV.
- **SC-003**: The report explicitly states whether the dbuf3-is-faster-for-int8 hypothesis
  is confirmed or refuted, with the numeric evidence for that verdict.
- **SC-004**: The report states how the fastest measured variant compares to the currently
  shipped dbuf4 production baseline, as a percentage or factor.
- **SC-005**: A reader can distinguish, from the report alone, a correctness-verified,
  coopmat-dispatch-confirmed result from a failed or unverified one -- no failed variant is
  mistaken for a valid measurement.

## Assumptions

- M5 EVT1 is the target device for all measurements in this feature (this workstream's
  active performance target), following the same clock-pinning and driver-verification
  discipline as prior specs in this workstream; device availability is confirmed before
  use rather than assumed free.
- This work is done in a new git worktree/branch dedicated to this experiment, per the
  user's explicit instruction, isolated from the currently uncommitted work sitting in this
  branch's working tree today.
- The representative shape set is the `wq` + `w1_gate` ops for each of the three target
  models (1B, 3B, 8B) -- 6 shapes total, matching `specs/008-8da4w-parameter-sweep`'s
  curated sweep-phase set -- rather than the full 3-model x 7-op catalog, keeping total
  device time bounded (4 variants x 6 shapes x 3 runs = 72 timed runs).
- New shader variants are built as opt-in, env-var-selected production-graph variants,
  reusing this workstream's existing dbuf1-4 harness pattern already proven for the fp16
  `4w` shader (a new env var read inside `QuantizedLinear.cpp`'s existing dispatch-selection
  logic; see `plan.md`/`research.md`) -- not as `specs/008`-style fully separate test-only
  ops. Default production dispatch behavior (env var unset) is unaffected while this
  experiment is in progress.
- This is a measurement-and-reporting feature: it determines which double-buffer variant is
  fastest for int8 WMMA, but does not itself require switching the shipped production
  shader's loop structure -- that is a follow-up decision informed by this feature's report,
  not an in-scope requirement here.
- The linear-op microbenchmark from User Story 2 alone is sufficient to declare a variant
  "fastest" -- no additional end-to-end (model-level tok/s) validation of the winning
  variant is required by this feature, matching `specs/008`'s precedent for this exact kind
  of sweep.
- "The 8da4w shaders (int8 WMMA)" refers specifically to the `dq8ca`/`q4gsw` int8
  cooperative-matrix linear shader (`linear_dq8ca_qw_coopmat.glsl`); the fp16 linear shader
  (`linear_qw_coopmat.glsl`, already dbuf1) and the SDPA coopmat shaders are out of scope.
- Per constitution Principle VI, each variant's compiled SPIR-V is disassembled once to
  confirm genuine int8 cooperative-matrix instructions are present -- required by the
  constitution for any coopmat shader change, even though no functional requirement above
  names it separately.
