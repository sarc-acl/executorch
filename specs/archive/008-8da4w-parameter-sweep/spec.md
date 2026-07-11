# Feature Specification: 8da4w Coopmat Tile/Subgroup Parameter Sweep

**Feature Branch**: `008-8da4w-parameter-sweep`

**Created**: 2026-07-04

**Status**: Draft

**Input**: User description: "On this RDNA3 iGPU mini PC, do a parameter sweep on many tile shapes, warp sizes etc, and find the optimal parameter for this device. Produce a report on 8da4w"

## Clarifications

### Session 2026-07-04

- Q: How broad should the swept parameter set be, given each combination requires a fresh shader compile, correctness pass, and timed run? → A: Curated, hypothesis-driven set (~8-12 combinations) -- both subgroup sizes crossed with a few tile shapes bracketing the currently-shipped 128x64/K=32 configuration, plus the pre-restructure 64x64/K=32 layout this exact shader used before the double-buffered restructure (`git show 49a51b1776^:.../linear_dq8ca_q4gsw_coopmat.yaml`, verified directly, not assumed) -- not an exhaustive combinatorial grid. Matches how this exact codebase's prior tile-sweeps ("M5 EVT1" tuning commits) were actually done: comparing specific named layouts, not exhaustive grids.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Prove the sweep mechanism on one configuration (Priority: P1) 🎯 MVP

As the contributor running this workstream, I want to build one alternate
tile-shape/subgroup-size variant of the `8da4w` coopmat shader, confirm it
compiles, dispatches, passes correctness, and produces a real timing number
on `rocky-ryzen`, so I know the sweep mechanism works before spending device
time compiling and measuring many variants.

**Why this priority**: `007-wmma-improvement-microbench` already found that
the shipped `8da4w` coopmat configuration (tuned on Samsung Xclipse hardware)
is consistently 10-22% *slower* than tiled on `rocky-ryzen` — a real,
reproducible regression, not noise. Before committing device time to a wide
sweep, prove that at least one alternate parameter combination can be built
and measured correctly.

**Independent Test**: Build a single alternate configuration (e.g. a
different `WG_TILE_K` or subgroup size than shipped), confirm it dispatches
the coopmat kernel (not a silent tiled fallback), passes the existing
correctness check, and produces a statistically sound timing number.

**Acceptance Scenarios**:

1. **Given** one alternate tile/subgroup parameter combination, **When** it
   is built and run against a real `8da4w` production shape, **Then** it
   either produces a correctness-verified timing number or an explicit,
   specific reason it could not (compile failure, pipeline-creation crash,
   or correctness failure).
2. **Given** a successfully measured alternate configuration, **When** its
   kernel name is inspected, **Then** it confirms the coopmat kernel actually
   dispatched, not a tiled fallback silently mislabeled as coopmat.

---

### User Story 2 - Sweep the parameter space (Priority: P2)

As the contributor, I want every reasonable combination of workgroup tile
shape and subgroup size swept and measured on `rocky-ryzen`, so the search
covers the space thoroughly rather than a handful of guesses.

**Why this priority**: A single alternate data point cannot support a
recommendation; the point of a sweep is confidence that better-or-worse
alternatives were actually compared, not assumed.

**Independent Test**: Run the full swept parameter set and confirm every
combination produces either a measured result or an explicit failure
reason, independent of which combination "wins."

**Acceptance Scenarios**:

1. **Given** the full set of swept tile-shape/subgroup-size combinations,
   **When** the sweep runs, **Then** every combination appears in the
   results as measured, or as explicitly failed (with which failure mode:
   compile failure, pipeline-creation crash, or correctness failure).
2. **Given** a combination that is mathematically incompatible with this
   workload (e.g. `WG_TILE_K` not dividing the quantization group size),
   **When** it is attempted, **Then** it is recorded as a correctness
   failure -- the shader compiles and dispatches, but the group-tiling
   arithmetic silently produces wrong output, which the correctness check
   (not a separate pre-flight validity check) is what actually catches it.
   One such combination is deliberately included in the curated set
   specifically to prove this detection path works (research.md Decision 4).

---

### User Story 3 - Report the optimal configuration for this device (Priority: P3)

As the contributor, I want a report naming the best-performing,
correctness-verified configuration found for `8da4w` on `rocky-ryzen`,
compared against both the currently-shipped (Xclipse-tuned) configuration
and the tiled baseline, so I can decide whether a device-specific override
is worth pursuing.

**Why this priority**: The sweep only has value once it answers the actual
question motivating it — is there a configuration that closes or reverses
the regression `007` found, and if so, by how much.

**Independent Test**: Generate the report from the sweep results and
confirm its recommendation (or explicit "no improvement found" conclusion)
traces directly back to specific measured numbers.

**Acceptance Scenarios**:

1. **Given** all measured (or explicitly failed) configurations, **When**
   the report is generated, **Then** it states whether any configuration
   beats the shipped configuration, whether any beats the tiled baseline,
   and which configuration (if any) is recommended.
2. **Given** no swept configuration beats the tiled baseline, **When** the
   report is generated, **Then** it states this explicitly as a finding
   rather than presenting the least-bad option as if it were a win.

---

### Edge Cases

- What happens when a swept combination fails to compile? Recorded as a
  compile failure with the compiler's actual error, not silently skipped.
- What happens when a combination compiles but crashes at pipeline
  creation (the same failure mode `007`'s investigation found documented
  for Xclipse, from a different driver)? Recorded as a pipeline-creation
  crash with the actual error, not retried or silently worked around.
- What happens when a combination runs but produces numerically incorrect
  output? Excluded from the performance ranking and reported as a
  correctness failure, never presented as a candidate "optimal" result.
- What happens when no tested configuration outperforms the tiled
  baseline at all? Reported as an explicit finding ("tuning did not close
  the gap on this device"), not obscured by only showing the best-of-a-bad-lot
  number.
- What happens when the current shipped configuration itself does not
  appear in the sweep by construction? It is included as the baseline
  comparison point regardless (already measured in `007`), not omitted.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The study MUST sweep 11 new, curated, hypothesis-driven
  workgroup tile-shape (`WG_TILE_M`, `WG_TILE_N`, `WG_TILE_K`) and
  subgroup-size combinations for the `8da4w` (`linear_dq8ca_q4gsw`)
  coopmat shader specifically, on `rocky-ryzen` -- both subgroup sizes
  crossed with a few tile shapes bracketing the currently-shipped
  128x64/K=32 configuration and this shader's own pre-restructure
  64x64/K=32 layout (Clarification Session), not an exhaustive
  combinatorial grid. The already-measured shipped configuration is reused
  as a 12th, non-rebuilt reference point (from `007`), not a 12th new
  variant this feature builds.
- **FR-002**: The study MUST measure every valid, correctness-verified
  configuration's performance at real `8da4w` production shapes, with
  iteration count and standard deviation reported for every timing
  (constitution Principle IV) — no single untimed sample is reported as a
  result.
- **FR-003**: The study MUST verify correctness for each configuration
  before including its performance in any ranking or recommendation
  (constitution Principle I); a configuration whose correctness cannot be
  established MUST NOT be presented as a candidate optimum.
- **FR-004**: The study MUST record every configuration that fails to
  compile, crashes at pipeline creation, or fails correctness, with the
  specific failure reason — never silently dropped from the sweep's
  accounting.
- **FR-005**: The study MUST verify, for each measured configuration, that
  the coopmat kernel actually dispatched (via the reported kernel name)
  before reporting it as a valid measurement, distinguishing a true
  measurement from a silent tiled fallback (constitution Principle VI).
- **FR-006**: The study MUST report the best-performing correctness-verified
  configuration found, if any, compared explicitly against both (a) the
  currently-shipped configuration's measured performance (from `007`) and
  (b) the tiled baseline's measured performance (from `004`) on this same
  device.
- **FR-007**: If no swept configuration outperforms the tiled baseline,
  the study MUST state this explicitly as a finding rather than
  recommending the least-bad coopmat configuration as if it were a win.
- **FR-008**: This study MUST NOT modify the currently-shipped dispatch
  code, shader registration, or production tile parameters — it produces
  findings and a recommendation only; applying any recommendation is a
  separate, later decision.

### Key Entities

- **Swept Configuration**: One (`WG_TILE_M`, `WG_TILE_N`, `WG_TILE_K`,
  subgroup size) combination drawn from the sweep space, together with the
  derived subgroup-grid arrangement needed to realize it.
- **Sweep Result**: The outcome of attempting one Swept Configuration --
  either a correctness-verified timing (mean, standard deviation, iteration
  count, confirmed coopmat kernel name) or an explicit failure (compile
  failure, pipeline-creation crash, or correctness failure) with its
  specific reason. A combination that is mathematically incompatible with
  this workload (e.g. `WG_TILE_K` not dividing the quantization group
  size) is not a separate category -- it surfaces as a correctness
  failure, since the shader still compiles and dispatches but computes
  wrong output (verified directly: the group-tiling arithmetic truncates
  to zero K-iterations per group under integer division).
- **Optimal Configuration Recommendation**: The best-performing Sweep
  Result (if any exists), stated alongside its measured margin over both
  the shipped configuration and the tiled baseline.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every attempted configuration in the sweep appears in the
  final report -- either measured or explicitly failed with a stated
  reason -- none are silently missing.
- **SC-002**: The report states whether any swept configuration
  outperforms the currently-shipped configuration, and by how much, using
  only the report itself.
- **SC-003**: The report states whether any swept configuration
  outperforms the tiled baseline, and by how much, using only the report
  itself -- including the case where none do.
- **SC-004**: A reader can distinguish, from the report alone, a
  statistically confident measurement (iteration count and variance shown)
  from a single untimed sample -- no result is presented without both.
- **SC-005**: A reader can distinguish correctness-verified results from
  failed or unverified ones, so no configuration is mistaken for a
  validated recommendation when it is not.

## Assumptions

- Scope is limited to the `8da4w` (`linear_dq8ca_q4gsw`) coopmat shader
  only, per the user's explicit request; the `4w` scheme already shows a
  large, consistent win on this device (`007`) and is out of scope here.
- The swept parameters are the shader's existing template parameters
  (`WG_TILE_M`, `WG_TILE_N`, `WG_TILE_K`, subgroup size, and the derived
  subgroup-grid arrangement), covered by a curated ~8-12 combination set
  rather than an exhaustive grid (Clarification Session); the underlying
  hardware cooperative-matrix operation size (`MMA_M`/`MMA_N`/`MMA_K` =
  16x16x16) is fixed by the device (confirmed via `test_coopmat_probe`,
  all 14 reported configurations are 16x16x16) and is not swept.
- Whether subgroup size 32 is safe to build and run for this shader on
  `rocky-ryzen` is an open, empirically-answerable question, not assumed
  either way: the shipped configuration forces subgroup size 64
  specifically to work around a Samsung Xclipse compiler crash at 32; this
  device reports `min_subgroup_size: 32` / `max_subgroup_size: 64` via
  `test_coopmat_probe` and already runs the `4w` coopmat shader at a
  forced subgroup size of 32 successfully (per `007`), on a different
  driver (RADV/Mesa) than Xclipse's. The sweep tests this directly rather
  than assuming the same crash does or doesn't apply to `8da4w`.
- This is a measurement and reporting study only (per FR-008): it does not
  change shipped dispatch code or tile parameters. Applying any winning
  configuration to production, or making tile-parameter selection
  device-adaptive, is explicitly out of scope and left to a future,
  separate decision.
- Measurement reuses this workstream's established discipline: real
  `8da4w` production shapes (from `001`'s `shapes.json`), GPU
  timestamp-based timing with iteration count and standard deviation
  (constitution Principle IV), and the existing correctness-check
  machinery (`test_coopmat_linear_bench.cpp`'s reference comparison),
  extended to cover the newly-built variants.
- Measurement happens in two phases to bound device time across 11
  variants: the sweep phase (US2) measures each variant at a reduced,
  representative shape set (one square + one rectangular shape per model,
  6 total -- research.md Decision 3), not the full 3-model x 7-op `8da4w`
  catalog; only the best-performing configuration(s) identified from that
  phase are then re-measured against the full 21-case catalog (matching
  `007`'s exact shapes) before appearing in the final recommendation
  (US3).
- Measurement runs on the `rocky-ryzen` MiniPC (AMD Radeon 780M, RADV/Mesa
  driver) only; this is explicitly a device-specific tuning study, not a
  claim about Xclipse or any other device.
