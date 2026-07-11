# Feature Specification: Decode Shader WMMA Acceleration

**Feature Branch**: `012-decode-wmma-feasibility`

**Created**: 2026-07-05

**Status**: Draft

**Input**: User description: "improve decode shaders with WMMA. give them WMMA goodness"

## Clarifications

### Session 2026-07-05

- Q: How should FR-002/SC-001's "compute-bound vs. memory-bandwidth-bound vs. ambiguous" conclusion be determined? → A: Roofline model -- compare the kernel's theoretical arithmetic intensity (FLOPs per byte of weight data read, computed from its known data-access pattern) against this device's machine balance point (peak compute throughput ÷ peak memory bandwidth); intensity well below the balance point → bandwidth-bound, well above → compute-bound, close to it → ambiguous. This is computable analytically from the kernel's known access pattern and the device's published peak specs, without requiring GPU hardware performance-counter tooling.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Determine whether decode is actually a WMMA opportunity before building anything (Priority: P1) 🎯 MVP

As the contributor driving this workstream, I need to know whether decode's
per-token linear layer computation is limited by this device's compute
throughput or by its memory bandwidth, before investing in a new
cooperative-matrix shader for it -- because if decode is memory-bandwidth-
bound (reading the entire weight matrix once per generated token, at a
roughly 1:1 ratio of bytes read to multiply-adds performed), a
cooperative-matrix shader would only accelerate the multiply-add
throughput, which was never the bottleneck, and would show little to no
real speedup no matter how well it is implemented.

**Why this priority**: Every prior WMMA effort in this workstream (prefill
linear, prefill SDPA) targeted computation that was demonstrably compute-
bound, and each paid off substantially (60-70%+ gains). Decode has never
been measured this way -- it has only been assumed to be a "no
implementation exists" gap. Per this workstream's constitution (Principle
I, Correctness Before Performance; Principle VI, Verify With Tools, Never
Assume), building a new shader on an unverified assumption risks
substantial engineering effort for an outcome that could have been ruled
out cheaply first.

**Independent Test**: Can be fully tested by computing the existing
`linear_q4gsw_coop` GEMV kernel's (`M=1`) theoretical arithmetic intensity
and comparing it against the `rocky-ryzen` MiniPC's machine balance point
(peak compute throughput ÷ peak memory bandwidth) -- a roofline analysis,
independent of any new shader code and not requiring GPU hardware
performance-counter tooling.

**Acceptance Scenarios**:

1. **Given** the existing linear GEMV kernel's known data-access pattern,
   **When** its theoretical arithmetic intensity is computed and compared
   against this device's machine balance point, **Then** a stated
   conclusion is produced: compute-bound (intensity well above the balance
   point), memory-bandwidth-bound (well below it), or ambiguous (close to
   it).
2. **Given** the roofline analysis concludes decode is memory-bandwidth
   -bound, **When** this is found, **Then** it is reported as a real,
   actionable finding (recommending against building a new WMMA decode
   shader, and naming what would actually help instead) -- not overridden
   by proceeding to build the shader anyway.
3. **Given** the roofline analysis concludes decode is compute-bound or
   ambiguous, **When** this is found, **Then** the feature proceeds to
   User Story 2 (ambiguous cases proceeding as an explicit, documented
   judgment call per Edge Cases, not a silent default).

---

### User Story 2 - Build and correctness-prove a WMMA-capable decode shader for linear GEMV, if warranted (Priority: P2)

As the contributor driving this workstream, I need a new cooperative-
matrix-based shader for decode's linear layer (the dominant cost in
decode, ahead of decode SDPA), proven correct against a reference
implementation at a small, tile-aligned shape, before any performance
number is quoted for it.

**Why this priority**: This is genuinely new shader design, not a wiring
fix -- there is no existing coopmat decode shader to unblock (unlike every
prior WMMA feature in this workstream). Linear GEMV is scoped first
because it is the larger share of decode time (52-69% per the original
candidate assessment); decode SDPA's GEMV kernels are an explicit
follow-on, not part of this feature.

**Independent Test**: Can be fully tested by dispatching the new shader on
a small, tile-aligned decode-shaped input (`M=1`, `K`/`N` chosen to align
with this device's cooperative-matrix tile dimensions) and comparing its
output against a CPU/ATen reference implementation, independent of any
timing measurement.

**Acceptance Scenarios**:

1. **Given** User Story 1's roofline analysis concluded decode is
   compute-bound or ambiguous, **When** a new WMMA-capable decode shader is
   designed and dispatched at a small tile-aligned shape, **Then** its
   output matches a CPU/ATen reference implementation within the tolerance
   already established for this workstream's half-precision comparisons.
2. **Given** the new shader's compiled SPIR-V, **When** it is disassembled,
   **Then** it is confirmed to contain genuine cooperative-matrix
   instructions -- not a tiled kernel renamed to look like one.
3. **Given** User Story 1's roofline analysis concluded decode is
   memory-bandwidth-bound, **When** this feature is scoped,
   **Then** User Story 2 is not attempted, and the feature's deliverable is
   User Story 1's finding and recommendation instead.

---

### User Story 3 - Measure whether the new decode shader actually speeds up real decode, per target model (Priority: P3)

As the contributor driving this workstream, I need to know whether the new,
correctness-proven WMMA decode shader measurably speeds up real decode
token generation for each of this workstream's three target models,
compared to the existing tiled/coop decode kernel -- at the shader
-microbenchmark tier first (matching this workstream's two-tier
discipline), before any claim is made about real end-to-end decode speed.

**Why this priority**: Turns a correctness-proven shader into an actual
performance verdict -- the same two-tier discipline (tier-1 microbenchmark
before tier-2 e2e) already applied to prefill linear (`007`→`009`) and
prefill SDPA (`010`→`011`).

**Independent Test**: Can be fully tested by benchmarking the new shader
against the existing decode GEMV kernel for one target model's real
decode shape (`M=1`, that model's real `K`/`N`), independent of the other
two models.

**Acceptance Scenarios**:

1. **Given** a correctness-proven new decode shader (User Story 2), **When**
   it is benchmarked against the existing tiled/coop decode kernel at each
   target model's real per-token shape, **Then** each model has a directly
   comparable pair of timings with iteration count and variance reported.
2. **Given** the three models' results, **When** the report is produced,
   **Then** it states a clear verdict per model -- real speedup, no
   meaningful difference, or regression -- consistent with (or explicitly
   diverging from) User Story 1's profiling-based prediction.

### Edge Cases

- What happens if User Story 1's profiling is ambiguous (decode shows
  significant utilization of both compute and bandwidth, with no clear
  dominant bottleneck)? This is reported as its own finding -- the decision
  to proceed to User Story 2 is then a judgment call made explicitly and
  documented with its reasoning, not silently defaulted either way.
- What happens if the new shader in User Story 2 cannot be made correct at
  any shape within reasonable effort (e.g. a fundamental tiling mismatch
  for `M=1` that has no viable workaround)? This is reported as a real,
  named blocker -- the feature's deliverable becomes that finding, and User
  Story 3 is not attempted.
- What happens if User Story 3 shows a real correctness-proven shader that
  is nonetheless slower than the existing kernel at real model shapes
  (e.g. tile-padding overhead at `M=1` outweighs any compute-throughput
  gain)? This is reported as a real, named result -- not hidden, and not
  used to retroactively second-guess User Story 1's methodology without
  cause.
- What happens to decode SDPA (the other decode WMMA gap, 26-34% of decode
  time)? Explicitly out of scope for this feature -- a natural follow-on
  once linear decode's outcome (positive or negative) is known.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST compute the existing linear GEMV kernel's
  (`linear_q4gsw_coop`, `M=1`) theoretical arithmetic intensity (FLOPs per
  byte of weight data read, from its known data-access pattern) and compare
  it against the `rocky-ryzen` MiniPC's machine balance point (peak compute
  throughput ÷ peak memory bandwidth, from published device specs) --
  a roofline analysis, not a requirement for GPU hardware performance
  -counter tooling.
- **FR-002**: The system MUST state an explicit conclusion from FR-001's
  roofline comparison: whether decode's linear layer is compute-bound
  (intensity well above the balance point), memory-bandwidth-bound
  (intensity well below it), or ambiguous (close to it) -- before any new
  shader is designed.
- **FR-003**: If FR-002 concludes decode is predominantly memory-bandwidth
  -bound, the system MUST NOT proceed to build a new WMMA decode shader in
  this feature; it MUST instead report this conclusion, with a
  recommendation of what would actually help decode speed (e.g. more
  aggressive weight quantization, or batching/speculative decoding to
  create a real `M>1` opportunity).
- **FR-004**: If FR-002 concludes decode is compute-bound or ambiguous, the
  system MUST design a new cooperative-matrix-capable shader for the
  linear GEMV decode path and prove its correctness against a CPU/ATen
  reference at a small, tile-aligned shape before any performance number is
  produced for it.
- **FR-005**: The system MUST confirm, via SPIR-V disassembly, that the new
  shader's compiled output contains genuine cooperative-matrix
  instructions.
- **FR-006**: The system MUST benchmark the new shader against the
  existing decode GEMV kernel at each of the three target models' real
  per-token decode shape, reporting iteration count and variance for every
  timing.
- **FR-007**: The system MUST report a clear verdict per target model
  (real speedup / no meaningful difference / regression), and state
  whether each agrees with User Story 1's profiling-based prediction.
- **FR-008**: Decode SDPA's GEMV kernels are explicitly out of scope for
  this feature.

### Key Entities

- **Decode Roofline Finding**: the result from User Story 1's roofline
  analysis -- the kernel's theoretical arithmetic intensity, this device's
  machine balance point, and the resulting compute-bound/bandwidth-bound/
  ambiguous conclusion that gates the rest of the feature.
- **Decode WMMA Correctness Case**: the new shader's output at a small
  tile-aligned shape vs. its CPU/ATen reference, plus its SPIR-V
  cooperative-matrix confirmation.
- **Decode WMMA Microbenchmark Case**: one target model's existing-kernel
  vs. new-shader timing pair at its real decode shape, with a per-model
  verdict.
- **Decode WMMA Feasibility Report**: the consolidated document --
  the roofline finding first, then (if applicable) the correctness proof,
  then the per-model microbenchmark verdicts, then one overall statement
  of whether WMMA acceleration is worth pursuing for decode on this
  device.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A roofline-based conclusion about decode's compute-vs
  -bandwidth bottleneck (arithmetic intensity vs. this device's machine
  balance point) exists before any new shader is designed, backed by the
  kernel's actual data-access pattern and the device's published peak
  specs rather than assumption.
- **SC-002**: If a new shader is built, its correctness against a
  reference implementation is established before any performance number is
  reported for it.
- **SC-003**: If a new shader is built, each of the three target models
  has a directly comparable existing-kernel-vs-new-shader timing pair, with
  iteration count and variance on every number.
- **SC-004**: A reader can determine, without re-deriving anything from raw
  data, whether WMMA acceleration is worth pursuing for decode on this
  device -- and if the answer is no, exactly why, and what would actually
  help instead.

## Assumptions

- Scope is `rocky-ryzen` MiniPC only, matching every prior tier-1 feature
  in this workstream -- Samsung/Xclipse validation is a separate, future
  feature.
- Scope is decode's **linear** GEMV kernel only (`linear_q4gsw_coop`,
  covering both `4w` and `8da4w` via its existing `DYNAMIC_QUANT_VARIANT`
  parameter) -- decode SDPA's two GEMV kernels
  (`sdpa_compute_attn_weights_coop`, `sdpa_compute_out_coop`) are an
  explicit follow-on feature, not part of this one, mirroring how prefill
  linear (`003`→`009`) and prefill SDPA (`010`→`011`) were scoped as
  separate features rather than one combined effort.
- This feature's outcome is genuinely open -- unlike every prior WMMA
  feature in this workstream (which unblocked already-correct, already
  -written shader code), this is new shader design against a workload
  (`M=1` GEMV) that may be fundamentally memory-bandwidth-bound. A
  conclusion of "not worth building" is a valid, complete outcome for this
  feature, not a failure to be avoided by building the shader regardless
  of what the profiling shows.
- "This device's peak compute throughput" and "peak memory bandwidth" for
  the roofline comparison in FR-001 are read from already-available
  hardware specifications/vendor documentation for the `rocky-ryzen`
  MiniPC's GPU, not re-derived via a new benchmarking methodology.
- If User Story 2 is undertaken, its correctness tolerance and
  verification approach mirror `010`'s established methodology (small
  tile-aligned shape, CPU/ATen reference, dtype-appropriate tolerance,
  SPIR-V inspection) rather than inventing a new one.
