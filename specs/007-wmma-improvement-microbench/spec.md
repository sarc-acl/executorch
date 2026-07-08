# Feature Specification: WMMA Coopmat Improvement Microbenchmark

**Feature Branch**: `007-wmma-improvement-microbench`

**Created**: 2026-07-04

**Status**: Draft

**Input**: User description: "Conduct WMMA study on this MiniPC, for all the baseline we studied, we identified candidates for WMMA. I alway have some WMMA shaders, and the shader suppose to be double buffer implementation with tile sizes 128x64. Conduct the study that report to me the WMMA improvement over baselines on at microbenchmark level"

## Clarifications

### Session 2026-07-04

- Q: What evidence should the study require before trusting a WMMA measurement as both real and correct, given `test_coopmat_linear_bench.cpp`'s existing correctness tests only cover small synthetic shapes (64/128/256), not the exact production K/N dimensions? → A: Kernel-name dispatch check (FR-004) + SPIR-V inspection confirming genuine cooperative-matrix instructions in the compiled shader + citing the existing generic-shape correctness tests already covering that op/kernel family. New correctness tests at the exact production shapes are explicitly out of scope for this study.
- Q: Which existing dataset is the correct tiled-baseline reference to diff WMMA numbers against, given coopmat dispatch requires Buffer storage while `001`'s microbench data is Texture3D-only (no storage dimension at all)? → A: `004-linear-storage-comparison`'s Buffer-storage tiled numbers -- holds storage type constant at Buffer on both sides of the comparison, avoiding the storage-vs-dispatch-path confound. `001`'s Texture3D-only numbers are not used as the baseline reference for this study.
- Q: How should the single "overall improvement" figure (FR-008) be computed across ops of very different sizes? → A: Time-weighted average -- each op's speedup weighted by its share of total baseline runtime, reusing `003`'s existing per-op `pct_of_phase` weighting data, so the figure answers "how much real wall-clock time did WMMA save" rather than treating every op equally regardless of size.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Prove the comparison on one configuration (Priority: P1) 🎯 MVP

As the contributor running this workstream, I want a directly comparable,
statistically sound measurement of the existing double-buffered, 128x64-tile
WMMA/coopmat quantized-linear shader against the already-established tiled
baseline, for one representative model/scheme configuration, so I can prove
the measurement methodology is sound before spending device time on the full
matrix.

**Why this priority**: Every prior feature in this workstream (001, 004, 006)
proved its mechanism on one configuration before scaling to all six --
repeating that discipline here catches a broken comparison early, not after
a full run.

**Independent Test**: Measure both the tiled and WMMA dispatch time for one
configuration's prefill linear op, confirm the WMMA dispatch actually
occurred (not a silent tiled fallback), and confirm the reported speedup/
regression number is directly traceable to the two measured times.

**Acceptance Scenarios**:

1. **Given** the existing tiled-baseline measurement for one configuration
   (from `001`), **When** the same op/shape is measured with the WMMA/coopmat
   path enabled, **Then** both timings are reported side by side with an
   explicit percentage difference.
2. **Given** a measured WMMA result, **When** the dispatched kernel name is
   inspected, **Then** it confirms the coopmat kernel actually ran (not a
   tiled fallback silently mislabeled as WMMA).

---

### User Story 2 - Measure every WMMA-candidate configuration (Priority: P2)

As the contributor, I want the proven comparison extended to every
model/scheme configuration and operation already identified as a WMMA
candidate in `003-wmma-shader-candidates`, so the study covers this
workstream's full established scope rather than one anecdotal data point.

**Why this priority**: A result for one configuration is not evidence for
the other five; the constitution's default benchmark scope requires all
three target models at both int4 schemes before a finding counts as
complete.

**Independent Test**: For each of the six configurations, run the standard
capture procedure and confirm a directly comparable tiled-vs-WMMA pair is
produced, independent of the other five.

**Acceptance Scenarios**:

1. **Given** all six model/scheme configurations, **When** each is measured,
   **Then** every one produces either a measured tiled-vs-WMMA pair or an
   explicit, stated reason it could not be measured.
2. **Given** an operation that only ever runs as a GEMV (`M=1`) -- decode's
   linear ops, and vocab/`lm_head` projection even though it is timed within
   the "prefill" phase bucket -- **When** the study runs, **Then** it is
   reported as explicitly out of scope with the reason, not silently
   omitted.

---

### User Story 3 - Report the improvement, in full and at a glance (Priority: P3)

As the contributor, I want a report that states the overall WMMA improvement
alongside a full per-configuration breakdown, so I can both quickly answer
"did this workstream's coopmat investment pay off" and drill into any
configuration that behaved differently.

**Why this priority**: The measurement only has value once it answers the
question that motivated it; without a synthesized report, the raw
measurements from US2 are not yet an "improvement study."

**Independent Test**: Generate the report from the US2 measurements and
confirm every configuration's consistency/verdict traces directly back to
its own two measured numbers, with an overall summary computed from the
per-configuration results.

**Acceptance Scenarios**:

1. **Given** all measured (or explicitly excluded) configurations, **When**
   the report is generated, **Then** it states an overall improvement
   figure and a full per-configuration comparison table.
2. **Given** a configuration whose WMMA dispatch path cannot complete the
   kernel-dispatch + SPIR-V + existing-test correctness check (FR-007),
   **When** it appears in the report, **Then** it is flagged as not
   correctness-verified rather than presented as a validated result.

---

### Edge Cases

- What happens when a candidate shape does not satisfy the WMMA shader's
  tile-alignment requirements and silently falls back to the tiled kernel?
  The study must detect this (by checking the dispatched kernel name) and
  report it as "no WMMA dispatch occurred," not as a WMMA measurement.
- What happens when a WMMA-eligible op's kernel has no existing correctness
  test at any shape (not just the exact production shape), or when SPIR-V
  inspection cannot confirm genuine cooperative-matrix instructions in the
  dispatched kernel? The study must flag it as unverified rather than
  presenting an unvalidated perf number as a finding.
- How does the study handle GEMV (`M=1`) operations -- decode's linear ops,
  and vocab/`lm_head` projection specifically, which `003`'s classification
  data shows is *always* `M=1` even when timed inside the "prefill" phase
  bucket (only the final token's logits are needed) -- which `003` already
  found have no WMMA-capable kernel? They are reported as explicitly out of
  scope with that stated reason, never silently dropped from the
  configuration count.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The study MUST measure both the tiled-baseline dispatch time
  and the double-buffered WMMA/coopmat dispatch time for every WMMA-candidate
  prefill linear operation identified in `003-wmma-shader-candidates`, at
  real per-model shapes.
- **FR-002**: The study MUST cover all six model/scheme configurations
  already established as this workstream's default benchmark scope (LLaMA
  3.1 8B / 3.2 3B / 3.2 1B, at both the `4w` and `8da4w` int4 schemes).
- **FR-003**: Every reported timing MUST include its iteration count and
  standard deviation; a single untimed run is never reported as a result.
- **FR-004**: For every measured operation, the study MUST verify the WMMA/
  coopmat kernel actually dispatched (via the reported kernel name) before
  reporting it as a WMMA result, distinguishing a true WMMA measurement from
  a silent tiled fallback.
- **FR-005**: The study MUST report an explicit percentage speedup or
  regression for each measured configuration/operation, comparing the WMMA
  time against the corresponding **Buffer-storage** tiled-baseline time
  already captured in `004-linear-storage-comparison` (not `001`'s
  Texture3D-only numbers), holding storage type constant across the
  comparison.
- **FR-006**: The study MUST report every GEMV (`M=1`) operation -- decode's
  linear ops, and `lm_head`/vocab projection specifically, which is `M=1`
  even within the "prefill" phase bucket -- as explicitly out of scope,
  citing that no WMMA-capable GEMV kernel exists (per `003`'s finding),
  rather than omitting them without explanation.
- **FR-007**: The study MUST establish correctness confidence for each
  measured WMMA kernel via all three of: (a) the kernel-dispatch
  verification from FR-004, (b) inspecting that kernel's compiled SPIR-V to
  confirm genuine cooperative-matrix instructions are present (per the
  project constitution's Principle VI), and (c) citing the existing
  generic-shape correctness test(s) already covering that op/kernel family
  -- authoring new correctness tests at the exact production K/N shapes is
  explicitly out of scope for this study. Any operation for which this
  three-part check cannot be completed MUST be flagged as not
  correctness-verified rather than reported as a validated comparison.
- **FR-008**: The study MUST produce both a per-configuration comparison
  table and an overall summary figure: a time-weighted average speedup,
  where each measured operation's speedup is weighted by its share of total
  baseline runtime (reusing `003`'s existing per-op `pct_of_phase`
  weighting), so the result is understandable both at a glance -- as real
  wall-clock impact, not an unweighted per-op average -- and in full detail.
- **FR-009**: Every one of the six configurations' candidate operations MUST
  appear in the final report -- either measured, or excluded with an
  explicit, stated reason -- never silently missing.

### Key Entities

- **Benchmark Configuration**: One (model, quantization scheme, operation)
  combination drawn from the six-configuration default scope and the WMMA
  candidates already identified in `003`.
- **Measurement Pair**: The tiled-baseline time and the WMMA/coopmat time for
  one Benchmark Configuration, each with its iteration count and standard
  deviation, plus the dispatched kernel name actually observed.
- **Comparison Result**: The percentage speedup/regression derived from one
  Measurement Pair, together with its correctness-verification status and,
  where applicable, an explicit exclusion reason.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every WMMA-candidate operation across all six model/scheme
  configurations appears in the final report, either measured or excluded
  with a stated reason -- none are silently missing.
- **SC-002**: For every measured operation, a reader can determine the exact
  percentage speedup or regression of WMMA over the tiled baseline using
  only the report, without consulting any other file.
- **SC-003**: A reader can distinguish, from the report alone, a
  statistically confident measurement (iteration count and variance shown)
  from a single untimed sample -- no result is presented without both.
- **SC-004**: The report distinguishes correctness-verified results from
  unverified ones, so a reader can never mistake an unvalidated number for a
  validated finding.
- **SC-005**: The report states one overall, time-weighted improvement
  figure that summarizes the full per-configuration table by each
  operation's share of total baseline runtime, answering "did WMMA help,
  in real wall-clock terms" without requiring the reader to average the
  table by hand.

## Assumptions

- The "WMMA shaders" referenced are the existing, already-implemented
  double-buffered, 128x64-tile (`WG_TILE_M=128`, `WG_TILE_N=64`) coopmat
  quantized-linear shaders (`linear_q4gsw_coopmat`,
  `linear_dq8ca_q4gsw_coopmat`); this study measures them as they exist
  today and does not involve building or modifying a shader.
- The tiled baseline is `004-linear-storage-comparison`'s **Buffer-storage**
  tiled measurement (`ET_VK_FORCE_TILED_LINEAR=1`, at Buffer storage since
  that's what coopmat dispatch requires) -- not `001`'s Texture3D-only
  numbers. This study does not re-capture that baseline from scratch.
- Scope is the shader microbenchmark tier only (this workstream's tier-1
  measurement, per the project constitution) -- end-to-end model-level WMMA
  impact is a separate, later concern, mirroring how the storage-comparison
  work went from a microbenchmark (`004`) to an end-to-end study (`006`).
- Only the two in-scope int4 schemes (`4w`, `8da4w`) are covered, matching
  every prior feature in this workstream; `fp16`/`8w`/`8da8w` are out of
  scope here.
- Decode/GEMV operations are out of scope, per `003`'s finding that no
  WMMA-capable GEMV kernel exists today.
- Measurement runs on the `rocky-ryzen` MiniPC (RDNA3 integrated GPU), this
  workstream's primary local validation platform.
- "All the baselines we studied" refers to the six model/scheme
  configurations and the WMMA-candidate operations already classified in
  `003-wmma-shader-candidates`, not a new candidate-identification pass.
