# Feature Specification: M5 EVT1 Full Microbenchmark Suite — Stable Results Report

**Feature Branch**: `020-run-existing-linear`

**Created**: 2026-07-06

**Status**: Draft

**Input**: User description: "Given we have linear microbench (test_coopmat_linear_bench.cpp), SDPA microbenchmark (test_sdpa_coopmat_bench.cpp), and baseline microbenchmark (test_llama_baseline_bench.cpp), all running real per-model shader shapes (K/N/head_dim/num_heads/num_kv_heads derived from each checkpoint's real params.json, matching the real e2e workload) for all 3 models (LLaMA 3.2-1B, 3.2-3B, 3.1-8B) -- DO the microbenchmarks, get stable results, and produce a report."

## Context (why this feature exists now)

Three shader microbenchmark harnesses already exist in-tree, each already
using real per-model shapes for all 3 target models (1B/3B/8B), derived
from each checkpoint's actual architecture config, not synthetic/arbitrary
shapes:

- `test_coopmat_linear_bench.cpp` — quantized-linear (`4w`, `8da4w`)
  tiled-vs-coopmat, per real `(K,N)` weight shape per model.
- `test_sdpa_coopmat_bench.cpp` — SDPA prefill tiled-vs-coopmat, per real
  `(head_dim, num_heads, num_kv_heads)` per model.
- `test_llama_baseline_bench.cpp` — tiled-only baseline (both `4w`/`8da4w`,
  both prefill M=2048 and decode M=1 regimes, both texture/buffer storage),
  per real per-model shape including `lm_head`.

The first two were already run once on M5 EVT1 (`specs/016-m5-linear-sdpa-microbench`),
producing single-invocation reports. `test_llama_baseline_bench.cpp` has
**never been run on M5 EVT1** — every prior reference to it
(`specs/001`, `004`, `007`, `010`, `014`) is MiniPC-era or planning-only.
Separately, `specs/016`'s numbers were captured from one binary
invocation per harness; this workstream's own statistical-rigor
convention (Principle IV: every tier-2/e2e number is a 3-rep
mean+CoV, never a single sample) has not yet been applied at the
microbenchmark tier — a single invocation's internal 5-run mean±stddev
proves the *op* is stable within that invocation, but not that the
*whole binary run* is repeatable run-to-run (thermal drift, GPU state
carried over from a prior process, a stale build). This feature closes
both gaps: run `test_llama_baseline_bench` on M5 EVT1 for the first time,
re-run the other two with an explicit cross-invocation repeatability
check, and produce one consolidated report spanning all three.

## Clarifications

### Session 2026-07-06

- Q: What threshold defines "unstable" across the 3 invocations
  (FR-006/SC-001) -- a fixed numeric CoV cutoff, or this workstream's
  existing practice? → A: No fixed cutoff. Always report the observed
  spread/CoV% for every case; flag a case as unstable only when it is a
  clear outlier relative to its peers (the other cases' spreads in the
  same run), matching this workstream's existing precedent
  (`specs/015-m5-e2e-wmma-validation`'s "769.35 tok/s, high CoV flagged"
  was called out by comparison against its peers, not against a
  predefined number) rather than inventing a magic-number threshold.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run all three microbenchmarks on M5 EVT1 with verified preconditions (Priority: P1)

An engineer (or an agent acting on their behalf) needs trustworthy
microbenchmark numbers for the linear, SDPA, and baseline op families on
the real target hardware, not numbers that might reflect a stale driver,
an unpinned clock state, or a stale build that doesn't match the current
shader source.

**Why this priority**: Every other user story consumes this one's output.
An untrustworthy raw measurement invalidates any report built on top of it
— this is the MVP: without it, there is nothing to report.

**Independent Test**: Can be fully tested by confirming, for each of the
3 binaries, that the on-device driver identity, clock-pin state, and
binary build freshness were all verified immediately before that binary's
run, and that its raw console output was captured to a results file.

**Acceptance Scenarios**:

1. **Given** the M5 EVT1 target is free and reachable, **When** this
   story executes, **Then** the on-device Vulkan driver identity is
   confirmed against the workspace's known-good table before any
   measurement, per Principle VIII.
2. **Given** the workspace's default is pinned clocks, **When** this
   story executes, **Then** GPU/MIF/INT clocks are verified pinned
   (509/2730/663 MHz) via sysfs readback before any measurement, per
   Principle VII — correcting the floating state left over from the
   unrelated, stopped `specs/019` session first.
3. **Given** any of the three harness source files may have changed since
   the on-device binary was last built (e.g. the uncommitted 1B/3B shape
   extension already in `test_coopmat_linear_bench.cpp`), **When** this
   story executes, **Then** each binary is rebuilt and re-pushed before
   its run, not assumed current.
4. **Given** all three binaries have run, **When** their raw console
   output is captured, **Then** each is saved to this feature's own
   `results/raw/` directory for the next story to consume.

---

### User Story 2 - Confirm results are stable, not a single lucky/unlucky sample (Priority: P1)

The same engineer needs to know the numbers reflect steady, repeatable
device behavior — not a one-off thermal state, a GPU reset artifact, or
noise — before trusting any of them enough to put in a report.

**Why this priority**: Tied for top priority with US1: a single
invocation's numbers are not yet evidence of anything beyond "this is
what happened once." This is the gap `specs/016`'s prior single-run
capture left open, and the literal meaning of "get stable results" in
this feature's own input.

**Independent Test**: Can be fully tested by confirming each of the 3
binaries was invoked at least 3 separate times (matching this
workstream's established e2e repeat convention) and that the per-shape/
per-op values across those invocations are within the documented
tolerance of each other, with any run that falls outside tolerance
flagged rather than silently averaged in.

**Acceptance Scenarios**:

1. **Given** a binary has been invoked 3 separate times end-to-end,
   **When** the same shape/op's reported value is compared across the 3
   invocations, **Then** the spread is reported explicitly (not just a
   single blended mean), consistent with Principle VII's per-rep
   reporting requirement for exactly this kind of run-to-run comparison.
2. **Given** any shape/op's 3-invocation spread exceeds a documented
   tolerance, **When** the report is produced, **Then** that shape/op is
   flagged as unstable in the report, with its actual spread stated, not
   dropped or smoothed over.
3. **Given** a harness's own dispatch-confirmation output (e.g. the
   linear bench's `!` fired-flag, the SDPA bench's `confirmed`/`NOT
   CONFIRMED` column), **When** any invocation reports a config as not
   having dispatched the intended coopmat kernel, **Then** that config's
   throughput number is never reported as if it were a coopmat result.

---

### User Story 3 - Produce one consolidated, plain-language report (Priority: P2)

A reader who is not a GPU/shader specialist (e.g. reviewing this
workstream's weekly status) needs one document that says, for each op
family and model, whether coopmat/WMMA wins, by how much, and where it
doesn't — without needing to read three separate raw logs or reconcile
three different report formats.

**Why this priority**: This is the feature's actual deliverable per its
own input ("produce a report"), but it depends on US1/US2's verified,
stable data existing first — it adds no new measurement, only synthesis.

**Independent Test**: Can be fully tested by confirming a single report
file exists that covers all three microbenchmarks, states per-model
per-scheme results, and is understandable without reading the raw harness
source.

**Acceptance Scenarios**:

1. **Given** US1/US2's stable results exist for all three harnesses,
   **When** the report is produced, **Then** it contains one section per
   harness (linear, SDPA, baseline), each with a per-model/per-scheme
   results table and a plain-language summary sentence.
2. **Given** this workstream already has prior single-invocation reports
   for linear and SDPA (`specs/016`), **When** this feature's report is
   produced, **Then** it explicitly states whether the new stable numbers
   are consistent with those prior numbers (and by how much they differ,
   if at all) rather than presenting fresh numbers with no acknowledgment
   of the prior measurement.
3. **Given** any anomaly surfaced in US2 (unstable shape, unconfirmed
   dispatch, crash), **When** the report is produced, **Then** that
   anomaly is named explicitly in the report body, not buried in a raw
   log the reader is expected to find themselves.

### Edge Cases

- What happens when a shape doesn't satisfy the coopmat tile-alignment
  gate (e.g. a model's `K`/`N` isn't a multiple of the required tile
  size)? The harness already marks this (fired=false / `!` flag); the
  report must surface it as "tiled fallback," never count it as a
  coopmat data point.
- What happens when the 3 invocations of a binary disagree beyond
  tolerance? Flagged as unstable per US2, with the actual spread stated —
  not averaged away, not silently dropped.
- What happens when the device is unreachable, busy with another
  session's process, or running an unrecognized driver? Blocked before
  any measurement starts (Principle VIII) — this feature does not
  override the shared-device safety checks already established
  elsewhere in this workstream.
- What happens when a harness binary's on-device copy predates a source
  change (e.g. the uncommitted linear-bench shape-table extension)?
  Rebuilt and re-pushed before use (US1, Acceptance Scenario 3) — never
  measured against a stale binary.
- What happens when `test_sdpa_coopmat_bench`'s per-model loop reports
  `NOT CONFIRMED` for a model? That model's speedup number is reported as
  unconfirmed/fallback in the output, per the harness's own existing
  `any_failure` check — this feature surfaces that flag, it does not
  invent a new one.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Before any measurement, the system MUST confirm the
  on-device Vulkan driver identity against the workspace's known-good
  driver table (Principle VIII) and MUST confirm GPU/MIF/INT clocks are
  pinned to the workspace default (509/2730/663 MHz), correcting any
  leftover floating state from an unrelated prior session first
  (Principle VII).
- **FR-002**: The system MUST run all three existing microbenchmark
  binaries — `test_coopmat_linear_bench`, `test_sdpa_coopmat_bench`,
  `test_llama_baseline_bench` — on the M5 EVT1 target, rebuilding and
  re-pushing each from current source first if its on-device copy may be
  stale.
- **FR-003**: For `test_coopmat_linear_bench`, the system MUST capture
  results for both quantization schemes (`4w`, `8da4w`) at all 3 models'
  real per-model `(K,N)` shapes, with each reported "coopmat" value backed
  by the harness's own dispatch-fired confirmation — a shape that falls
  back to tiled MUST NOT be reported as a coopmat data point.
- **FR-004**: For `test_sdpa_coopmat_bench`, the system MUST capture
  results for all 3 models' real per-model prefill shape, with each
  reported speedup backed by the harness's own `dispatch_confirmed`
  check.
- **FR-005**: For `test_llama_baseline_bench`, the system MUST capture
  results across both regimes (prefill M=2048, decode M=1), both storage
  types (texture3d, buffer), and both schemes (`4w`, `8da4w`), for all 3
  models including `lm_head` — this is this harness's first-ever run on
  M5 EVT1, so no prior M5 baseline exists to compare against.
- **FR-006**: Each of the three binaries MUST be invoked at least 3
  separate times end-to-end (not just relying on the harness's own
  internal 5-run-per-case average from a single invocation), and the
  per-shape/op value's spread (CoV%) across those 3 invocations MUST be
  reported for every case — no fixed numeric cutoff is used; a shape/op
  MUST be flagged as unstable when its spread is a clear outlier relative
  to its peer cases' spreads in the same run (per Clarifications session
  2026-07-06), never silently averaged over regardless of magnitude.
- **FR-007**: Every reported throughput/GFLOP-s number MUST retain the
  harness's own internal statistic (mean ± stddev over its timed runs)
  alongside the cross-invocation spread from FR-006 — never collapse to
  a single bare number.
- **FR-008**: The system MUST produce one consolidated report covering
  all three microbenchmarks, each in its own clearly separated section
  (they measure different, non-comparable op families), with a
  plain-language summary per section stating which configurations
  coopmat/WMMA wins for, by how much, and where it does not.
- **FR-009**: The report MUST explicitly compare its new linear/SDPA
  numbers against `specs/016-m5-linear-sdpa-microbench`'s existing
  single-invocation M5 EVT1 numbers, stating whether they are consistent.
- **FR-010**: Any correctness failure, crash, or unconfirmed dispatch
  encountered during measurement MUST be named explicitly in the report,
  not omitted or silently dropped from the results tables.

### Key Entities

- **Microbenchmark harness**: one of the three existing on-device C++
  binaries (linear, SDPA, baseline). Each defines its own set of
  shape/op/scheme/storage cases and reports GFLOP/s or latency per case.
- **Measurement invocation**: one full end-to-end run of a harness
  binary, producing raw console output for every case it defines.
- **Case result**: one (model, scheme, shape/op, storage/regime)
  combination's measured value from one invocation — mean ± stddev over
  that harness's internal timed runs, plus a dispatch-confirmation flag.
- **Stability verdict**: the cross-invocation comparison for one case
  result across its 3 (or more) invocations — stable or flagged, with the
  observed spread.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every reportable case across all three microbenchmarks has
  a stability verdict (stable, with its cross-invocation spread/CoV%
  stated, or explicitly flagged as an outlier relative to its peers) — no
  case is reported with only a single invocation's number, and no fixed
  numeric cutoff is invented to decide the flag.
- **SC-002**: 100% of case results reported as "coopmat" in the final
  report are backed by that harness's own dispatch-fired/confirmed flag
  from every one of its invocations — zero unconfirmed-but-reported
  coopmat claims.
- **SC-003**: A reader unfamiliar with the GLSL shader internals can read
  the report and correctly state, for each of the 3 models and each op
  family, whether coopmat/WMMA is faster and by roughly how much, without
  consulting any file other than the report itself.
- **SC-004**: Every anomaly (unstable case, unconfirmed dispatch,
  crash, correctness failure) encountered during measurement appears by
  name in the report — zero anomalies silently dropped from the final
  document.

## Assumptions

- Clocks are pinned to this workspace's documented default
  (509/2730/663 MHz) for every measurement in this feature — floating-clock
  measurement is out of scope here (that was `specs/019`'s concern, now
  stopped) and is not revisited by this feature.
- "Stable" is operationalized as: 3 separate end-to-end binary invocations
  per harness, with each case's spread (CoV%) across those invocations
  reported explicitly; per Clarifications session 2026-07-06, there is no
  fixed numeric cutoff for "unstable" — a case is flagged only when its
  spread is a clear outlier relative to its peers, matching this
  workstream's existing e2e precedent.
- `test_coopmat_attention_bench.cpp` is out of scope — it was already
  deleted from the tree in a separate, unrelated cleanup and was never
  one of the three harnesses this feature concerns.
- Rebuilding a harness binary when its source may have changed follows
  this workstream's already-documented build procedure; this feature does
  not need a new build mechanism, only to actually invoke the existing one
  before assuming a binary is current.
- The M5 EVT1 target is available for this feature's use for the duration
  of its measurement (shared-device availability is checked per Principle
  VIII/existing gotchas, not re-litigated here).
