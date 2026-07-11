# Feature Specification: SDPA Coopmat Correctness + Microbenchmark

**Feature Branch**: `010-sdpa-coopmat-microbench`

**Created**: 2026-07-05

**Status**: Draft

**Input**: User description: "SDPA prefill correctness + microbench"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Prove the SDPA coopmat path is actually correct (Priority: P1) 🎯 MVP

As the contributor driving this workstream, I need to know that the SDPA
cooperative-matrix (WMMA) prefill path -- `sdpa_compute_attn_weights_coopmat`
and `sdpa_compute_out_coopmat`, ported from `yanwen/quant-dev-active`'s WIP
backup and currently sitting in the working tree unbuilt and untested --
actually computes correct attention output, before any performance number
from it is trusted.

**Why this priority**: This code has never passed a correctness check in
this repository. `test_coopmat_attention_bench.cpp` exists as source but
isn't wired into the build; the origin commit's own message admits its
sibling matmul-tile-sweep code was "re-derived... NOT byte-identical to the
lost original." Per this workstream's constitution (Principle I,
non-negotiable), no coopmat performance number may substitute for a passing
correctness check, and none exists yet for this path.

**Independent Test**: Can be fully tested by building the correctness
harness, running it at one small, tile-aligned representative shape for
both the QK^T and attn·V coopmat shaders, and confirming the output matches
the CPU/tiled reference within tolerance -- independent of any timing
measurement.

**Acceptance Scenarios**:

1. **Given** the imported SDPA coopmat shaders and dispatch code, **When**
   `test_coopmat_attention_bench` (wired into the build) is run at a small,
   tile-aligned shape, **Then** its output matches the CPU/tiled reference
   within the same tolerance discipline this workstream already uses for
   the quantized-linear coopmat correctness checks.
2. **Given** a passing correctness run, **When** the compiled SPIR-V for
   both shaders is disassembled, **Then** genuine cooperative-matrix
   instructions (`OpCooperativeMatrixLoadKHR`/`OpCooperativeMatrixMulAddKHR`
   or equivalent) are confirmed present -- not just GLSL source that "looks
   right."
3. **Given** the correctness check fails, **When** this is discovered,
   **Then** it is reported and root-caused as a real bug in the imported
   code, not silently patched around or excluded from the count.

---

### User Story 2 - Measure SDPA coopmat's real prefill speedup (Priority: P2)

As the contributor driving this workstream, I need to know how much the
SDPA coopmat path actually speeds up prefill attention at the real shapes
each target model produces, at the shader-microbenchmark tier, once User
Story 1 has established it is trustworthy to measure at all.

**Why this priority**: This is the actual question this feature exists to
answer -- "does the newly-ported SDPA coopmat path help" -- but it can only
be answered once User Story 1 proves the numbers mean something.

**Independent Test**: Can be fully tested by running the microbenchmark
harness at one representative model's real prefill SDPA shape (confirming
coopmat dispatch via the harness's own kernel-name field) and computing a
directly comparable coopmat-vs-tiled speedup for that one case, independent
of the other two models.

**Acceptance Scenarios**:

1. **Given** a real target model's prefill (2048-token) SDPA shape, **When**
   the microbenchmark harness runs it with the coopmat path enabled (via its
   existing opt-in toggle) and again with it disabled, **Then** both timings
   are recorded with iteration count and standard deviation, and the
   dispatched kernel name is confirmed to actually differ between the two
   runs (coopmat vs. tiled), not just assumed from the toggle.
2. **Given** all three target models' real prefill SDPA shapes, **When**
   measurement is complete, **Then** each has a directly comparable
   tiled/coopmat pair, or an explicit blocked/excluded status with a stated
   reason.

---

### User Story 3 - Report whether SDPA coopmat helps, at a glance (Priority: P3)

As the contributor driving this workstream, I need the measurements turned
into one answer: does the SDPA coopmat path speed up real prefill attention,
and by how much, per model.

**Why this priority**: Turns raw measurements into the actual deliverable --
consistent with how `007` closed out the equivalent question for the
quantized-linear coopmat path.

**Independent Test**: Can be fully tested by generating the report and
confirming each model's reported speedup traces directly to its own two
measured times, with dispatch and correctness status visible alongside it.

**Acceptance Scenarios**:

1. **Given** the completed measurements from User Story 2, **When** the
   report is generated, **Then** it states, per model, the tiled and coopmat
   timings, the speedup percentage, and whether that case's dispatch and
   correctness were confirmed -- with an overall figure that does not
   obscure a per-model outlier.
2. **Given** any model whose shape doesn't meet the coopmat shaders' tile
   alignment requirements, **When** the report is generated, **Then** that
   model is listed as excluded with the specific unmet alignment condition,
   never silently dropped from the three-model count.

### Edge Cases

- What happens if the correctness check in User Story 1 fails? It is
  reported as a real, root-caused bug in the imported code (per this
  workstream's precedent for `007`'s own mid-implementation dispatch-wiring
  discovery), and User Story 2's measurement is blocked until it is
  resolved or the affected shader is explicitly excluded with a stated
  reason -- never worked around silently.
- What happens if a target model's real prefill SDPA shape doesn't satisfy
  the coopmat shaders' tile-alignment requirements (distinct M/N/K tiles for
  the QK^T vs. attn·V shader)? That model's SDPA coopmat measurement is
  reported as blocked/excluded with the specific unmet condition, matching
  `007`'s precedent for shape-ineligible cases.
- What happens with decode-phase SDPA? Out of scope -- the imported code
  itself keeps decode on the existing `_coop` path (no WMMA-capable GEMV
  kernel exists for attention any more than it does for linear, per `003`'s
  classification), and this feature does not change that.
- What happens with the matmul tile-sweep code (`GemmCoopmat.h`/`.cpp`
  changes) that arrived in the same import as the SDPA shaders? Out of
  scope -- confirmed via direct code reading that SDPA's coopmat dispatch
  uses its own dedicated shaders and does not call `add_matmul_coopmat_node`
  at all; the tile-sweep code is an independent thread not touched here.
- What happens with the internal board-codename reference (`M5 EVT1`)
  already sitting in the imported `SDPA.cpp` comment? Out of scope for this
  feature -- already flagged separately; safe to remain on this fork per
  the constitution's Repository & Distribution Scope, to be scrubbed only
  if/when this code is proposed for an upstream contribution.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST build a genuinely new correctness check
  (existing test suites contain no coverage of the SDPA coopmat shaders)
  comparing the coopmat path's output against the CPU/tiled reference at a
  small, tile-aligned shape, for both the QK^T and attn·V coopmat shaders,
  before any performance number from this path is reported.
- **FR-002**: Any change made to a coopmat shader or its dispatch code to
  resolve a correctness failure MUST have its compiled SPIR-V inspected to
  confirm genuine cooperative-matrix instructions are present, per this
  workstream's constitution (Principle VI).
- **FR-003**: The system MUST confirm, via the microbenchmark harness's own
  kernel-name reporting, that the coopmat shader actually dispatched (not a
  silent tiled fallback) before trusting any coopmat timing.
- **FR-004**: The system MUST measure microbenchmark-level prefill
  (2048-token) SDPA timing, both coopmat and tiled, for all three target
  models, with iteration count and standard deviation reported alongside
  every timing.
- **FR-005**: The system MUST report, per model, the tiled/coopmat speedup
  percentage, computed directly from that model's own two measured times.
- **FR-006**: Any model whose real prefill SDPA shape does not satisfy the
  coopmat shaders' tile-alignment requirements MUST be reported as
  explicitly excluded with the specific unmet condition -- never silently
  dropped from the three-model count.
- **FR-007**: The report MUST distinguish correctness-verified, dispatch-
  confirmed results from any case that could not be fully verified -- no
  reader may mistake an unverified number for a validated one.

### Key Entities

- **SDPA Coopmat Correctness Case**: one (shader, shape) pair's output
  compared against the CPU/tiled reference, its pass/fail outcome, and
  whether its SPIR-V was confirmed to contain genuine cooperative-matrix
  instructions.
- **SDPA Prefill Comparison Case**: one target model's tiled vs. coopmat
  prefill SDPA timing pair, its dispatch-confirmation status, and its
  computed speedup percentage (or its blocked/excluded status and reason).
- **SDPA Coopmat Microbenchmark Report**: the consolidated document with a
  per-model speedup table, an explicit excluded/blocked section (present
  even if empty), and a correctness/dispatch verification summary.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The SDPA coopmat path passes a genuine, newly-authored
  correctness check against the CPU/tiled reference before any performance
  number from it is reported anywhere in this feature's output.
- **SC-002**: Every reported timing carries its iteration count and standard
  deviation -- no single untimed run is presented as evidence.
- **SC-003**: A reader can determine, for each of the three target models,
  whether SDPA coopmat helps and by how much, without needing to re-derive
  anything from raw logs.
- **SC-004**: Any model that could not be measured (shape misalignment,
  failed dispatch or correctness check) is stated explicitly with its
  reason, never silently absent from the report.

## Assumptions

- The SDPA coopmat shaders, dispatch code (`SDPA.cpp`), and bench harness
  source (`test_coopmat_attention_bench.cpp`) already exist in the working
  tree, imported from `yanwen/quant-dev-active` in a prior session -- this
  feature's job is to build, correctness-verify, and benchmark them, not to
  author new shaders from scratch.
- SDPA's shape and dispatch behavior are independent of the surrounding
  linear layers' quantization scheme (`4w` vs. `8da4w`): the coopmat gate
  (`sdpa_coopmat_device_ok`/`sdpa_buf_half`) requires `Buffer`+`half` storage
  for Q/K/V/output uniformly regardless of which quantized-linear scheme
  produced them, and prefill SDPA's shape depends only on model
  architecture (head count, head dimension, context length) and the fixed
  2048-token workload, not on weight quantization. This feature therefore
  measures **one** representative case per target model (three
  configurations total), not the constitution's default six -- a deliberate,
  justified deviation, matching how `007` documented its own `lm_head`
  exclusion as a deliberate scope decision rather than an oversight.
- The SDPA coopmat path's opt-in toggle (`ET_VK_SDPA_COOPMAT` env var, plus
  the shared `ET_VK_DISABLE_COOPMAT` kill switch) already exists in the
  imported code and is reused as-is as this feature's tiled-vs-coopmat
  toggle mechanism, mirroring how `007` reused the existing
  `ET_VK_FORCE_TILED_LINEAR` toggle rather than inventing a new one.
- Scope is `rocky-ryzen` MiniPC only (tier-1, shader microbenchmark),
  matching every prior microbenchmark-tier feature in this workstream;
  on-device Samsung/Xclipse validation and any tier-2 (model-level) e2e
  measurement of this same path are explicitly out of scope, left to a
  future feature (mirroring how `006`/`009` followed `004`/`007` at the
  linear-coopmat tier).
- If the correctness check in User Story 1 reveals a real bug requiring a
  production code change (mirroring `007`'s own mid-implementation
  dispatch-wiring discovery), that change is proposed during planning and
  applied only with explicit user authorization, documented at the point
  it's made -- matching this workstream's established discipline for
  production dispatch-code changes.
