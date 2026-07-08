# Feature Specification: M5 EVT1 Linear + SDPA Coopmat Microbenchmark Validation

**Feature Branch**: `016-m5-linear-sdpa-microbench`

**Created**: 2026-07-06

**Status**: Draft

**Input**: User description: "Run the microbenchmark on linear and SPDA, run the same microbenchark that was ran on the minipc, get report like specs/010-sdpa-coopmat-microbench/results/sdpa-coopmat-microbench-report.md and specs/007-wmma-improvement-microbench/results/wmma-improvement-report.md"

## Clarifications

### Session 2026-07-06

- Q: Which SDPA benchmark harness should this feature use --
  `test_coopmat_attention_bench.cpp` (already wired into the build, but
  confirmed by reading its source to exercise the generic
  `matmul_coopmat`/`coopmat_mm_ref` path, unrelated to `SDPA.cpp`, and to
  crash on an unrelated-shape assertion during this session's exploration)
  or `test_sdpa_coopmat_bench.cpp` (not currently wired into the build,
  but its header comment explicitly names `specs/010-sdpa-coopmat-microbench`
  and it isolates the exact `sdpa_compute_attn_weights_*`/`sdpa_compute_out_*`
  dispatches that `specs/010`'s report actually measured)? → A:
  `test_sdpa_coopmat_bench.cpp` -- it is the harness that actually produced
  `specs/010`'s report; using the other one would silently benchmark the
  wrong shader family. Wiring it into the build is now part of this
  feature's scope (FR-001a).

## Context (why this feature exists now)

`specs/015-m5-e2e-wmma-validation` set out to validate this workstream's
`4w`/`8da4w` linear coopmat and SDPA-coopmat e2e wins (originally measured
on the `rocky-ryzen` MiniPC, `specs/007` and `specs/010`) on the real M5
EVT1 target. That feature's own dispatch-confirmation method (ETDump
per-event kernel-name capture, run against the full LLaMA model graph) is
now known to be **unreliable**: a direct wall-clock A/B test against
`ET_VK_FORCE_TILED_LINEAR` (a genuine, source-verified kill switch) showed
the *default* dispatch path is ~1.8x faster than *genuinely forced* tiled
on the same model/PTE/prompt -- yet ETDump's own per-event kernel-name
field claimed 100% tiled dispatch for that same default run. The
isolated, correctness-validated shader microbenchmark
(`test_coopmat_linear_bench`) that specs `007`/`010` already used on
MiniPC does not share this defect: it confirms coopmat dispatch via its
own kernel-name capture *and* reports GFLOP/s directly comparable to a
correctness-checked reference, at the exact production K/N shapes,
independent of the full-model-graph ETDump path that has proven unreliable.

This feature re-runs that same, already-trusted microbenchmark
methodology on M5 EVT1, for both linear (`007`'s harness,
`test_coopmat_linear_bench`) and SDPA (`010`'s actual harness,
`test_sdpa_coopmat_bench` -- see Clarifications), and reports the results
in the same format as the two existing MiniPC reports -- giving this workstream a
clean, tool-verified answer to "does coopmat/WMMA actually run faster than
tiled on M5 EVT1" that does not depend on the now-suspect e2e
dispatch-confirmation method.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Linear coopmat microbenchmark on M5 EVT1 (Priority: P1)

As this workstream's engineer, I need per-op (`wq`/`wk`/`wv`/`wo`/`w1_gate`/
`w2_down`/`w3_up`), per-model (1B/3B/8B), per-scheme (`4w`/`8da4w`) tiled-vs-
coopmat timing on M5 EVT1, at the exact production K/N shapes, with
dispatch and correctness independently confirmed per row -- mirroring
`specs/007`'s MiniPC report exactly, so the two are directly comparable.

**Why this priority**: This is the direct, tool-reliable replacement for
what `specs/015` could not trust from ETDump; it is the load-bearing
evidence for whether this workstream's coopmat work delivers on M5 EVT1
at all.

**Independent Test**: Run `test_coopmat_linear_bench` (or the
already-built `_spec014`/current-HEAD variant) on M5 EVT1 for all 42
(model x scheme x op) cases and produce a report with the same columns as
`specs/007`'s `wmma-improvement-report.md`.

**Acceptance Scenarios**:

1. **Given** the current HEAD's coopmat shader and dispatch code, **When**
   the linear microbenchmark runs on M5 EVT1 at production shapes for a
   given (model, scheme, op), **Then** the report records tiled and
   coopmat mean±stdev timing, speedup %, a significance classification,
   and confirms both dispatch (kernel-name) and correctness (existing
   `test_coopmat_linear_bench` correctness-shape coverage) for that row.
2. **Given** all 42 cases have run, **When** the report is assembled,
   **Then** it states an overall `4w` and `8da4w` speedup figure (parallel
   to `specs/007`'s "+60.6% / -15.2%" summary line) computed the same way
   (time-weighted across each scheme's measured ops).

---

### User Story 2 - SDPA coopmat microbenchmark on M5 EVT1 (Priority: P2)

As this workstream's engineer, I need per-model SDPA `sdpa_compute_attn_weights`/
`sdpa_compute_out` tiled-vs-coopmat timing on M5 EVT1, mirroring
`specs/010`'s MiniPC report format, independent of the SDPA e2e
crash/dispatch-ambiguity findings from `specs/015` (Q12).

**Why this priority**: Lower priority than linear because SDPA coopmat is
still opt-in and less central to this workstream's current scope, but the
same tool-reliability gap applies and this workstream needs a trustworthy
number before making any SDPA-coopmat claim on M5 EVT1.

**Independent Test**: Wire `test_sdpa_coopmat_bench.cpp` into the build
(it is not currently a build target) and run it on M5 EVT1 for all 3
target models, producing a report with the same columns as `specs/010`'s
`sdpa-coopmat-microbench-report.md`.

**Acceptance Scenarios**:

1. **Given** the current HEAD's SDPA coopmat shaders, **When** the
   attention microbenchmark runs on M5 EVT1 for a given model's real
   prefill SDPA shape (`head_dim`/`num_heads`/`num_kv_heads` from that
   model's `params.json`), **Then** the report records tiled and coopmat
   mean±stdev timing, speedup %, and a significance classification for
   that model, with dispatch confirmed via the harness's own kernel-name
   capture (not assumed).
2. **Given** all 3 models have run, **When** the report is assembled,
   **Then** it states an overall average speedup figure across the
   models that produced a valid (non-crashed, non-excluded) measurement,
   parallel to `specs/010`'s "66.8% faster... 3/3 real-effect" summary
   line.

---

### Edge Cases

- What happens if a shape in the sweep does not fit the harness's tile-alignment
  preconditions (e.g. `test_coopmat_linear_bench`'s existing `!` "buffer case did
  NOT dispatch a coopmat shader" marker from this session's own smoke test)? →
  Recorded as `not_applicable`/excluded with the reason stated, same as `specs/007`'s
  Excluded/Out-of-Scope section (e.g. `lm_head` M=1 gemv is out of scope there);
  never silently dropped from the case count.
- What happens if `test_sdpa_coopmat_bench` (once wired into the build) fails
  to build or crashes for a given model's shape? → That model's case is
  recorded as `blocked`, with the exact build/assertion/error text, and the
  harness is re-run for the remaining models; a single model's failure does
  not invalidate the rest of the sweep. (`test_coopmat_attention_bench`,
  the harness this spec originally cited before the Clarifications session,
  is not used -- confirmed via source reading to exercise the generic
  `matmul_coopmat`/`coopmat_mm_ref` path, not `SDPA.cpp`'s own shaders, and
  it crashed on an unrelated-shape assertion during this session's
  exploration of it.)
- What happens if a case's coopmat timing is *slower* than tiled (as `specs/007`
  found for every `8da4w` op on MiniPC)? → Reported as-is with a `real_effect`
  (not `noise`) significance label if the effect is consistent across runs,
  exactly like `specs/007`'s existing `8da4w` regression rows -- a regression is
  not suppressed or treated differently from a win.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST run `test_coopmat_linear_bench` (linear) and
  `test_sdpa_coopmat_bench` (SDPA) on the M5 EVT1 target, using
  the same per-model production shapes (`wq`/`wk`/`wv`/`wo`/`w1_gate`/
  `w2_down`/`w3_up` K/N dims and SDPA `head_dim`/`num_heads`/`num_kv_heads`)
  that `specs/007`/`specs/010` used on the `rocky-ryzen` MiniPC, derived
  from each target model's own `params.json` (Principle II's three named
  models: LLaMA 3.1 8B, 3.2 3B, 3.2 1B).
- **FR-001a**: `test_sdpa_coopmat_bench.cpp` MUST be wired into the Android
  build (it exists as source but is not currently a build target, per
  Clarifications) before it can be run on M5 EVT1.
- **FR-002**: Every reported timing MUST be a mean ± standard deviation
  over multiple timed runs (matching `specs/007`/`010`'s 5-timed-run, 3-
  discarded-warmup-run discipline) -- no single untimed run is presented
  as evidence, per this workstream's constitution Principle IV.
- **FR-003**: Every row MUST have its coopmat dispatch confirmed via the
  microbenchmark harness's own kernel-name capture (not the ETDump
  full-model-graph method this session found unreliable, and not assumed
  from the eligibility-gate code alone), per constitution Principle VI.
- **FR-004**: Every row MUST cite existing correctness coverage for that
  op/kernel family (the harness's own correctness-shape checks, or SPIR-V
  inspection confirming genuine `OpCooperativeMatrix*KHR` instructions),
  per constitution Principle I -- a performance number is never reported
  as the only signal that a dispatch path is correct.
- **FR-005**: The linear report MUST classify each row's speedup as
  `real_effect` or `noise` and state an overall time-weighted `4w` and
  `8da4w` speedup figure, in the same format as `specs/007`'s report.
- **FR-006**: The SDPA report MUST state an overall average speedup
  figure across the models that produced a valid measurement, in the
  same format as `specs/010`'s report.
- **FR-007**: Clock pins MUST be verified bound (GFLOP/s cross-check, per
  constitution Principle VII) and the on-device driver identity verified
  (Principle VIII) before any timing in either report is trusted, exactly
  as `specs/015` already established for this M5 EVT1 session.
- **FR-008**: Both reports MUST be written to
  `specs/016-m5-linear-sdpa-microbench/results/`, one file per
  microbenchmark (`linear-coopmat-microbench-report.md`,
  `sdpa-coopmat-microbench-report.md`), each explicitly labeled M5 EVT1
  (not MiniPC) and cross-referencing the MiniPC reports it mirrors.

### Key Entities

- **Linear Benchmark Case**: one (model, scheme, op) triple -- 42 total
  (3 models x 2 schemes x 7 ops), matching `specs/007`'s case set exactly.
  Fields: tiled timing (mean, stdev), coopmat timing (mean, stdev),
  speedup %, significance, dispatch status, correctness citation.
- **SDPA Benchmark Case**: one model -- 3 total, matching `specs/010`'s
  case set exactly. Fields: `head_dim`/`num_heads`/`num_kv_heads`, tiled
  timing, coopmat timing, speedup %, significance.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A reader can open `linear-coopmat-microbench-report.md` and
  see, for all 42 (model, scheme, op) cases (or an explicit excluded/blocked
  reason per case), a tiled-vs-coopmat comparison with dispatch and
  correctness independently confirmed -- without needing to trust the e2e
  ETDump method `specs/015` found unreliable.
- **SC-002**: A reader can open `sdpa-coopmat-microbench-report.md` and see
  the same, for all 3 models (or an explicit excluded/blocked reason).
- **SC-003**: Every number in both reports carries a stated iteration
  count and standard deviation; no number is a single untimed sample.
- **SC-004**: Both reports state an overall summary speedup figure in the
  same format/wording style as their MiniPC counterparts, making a
  side-by-side MiniPC-vs-M5-EVT1 comparison possible at a glance.

## Assumptions

- The linear microbenchmark harness (`test_coopmat_linear_bench.cpp`) and
  the shaders/dispatch code it exercises are already built for M5 EVT1 in
  this repo's current HEAD (this session already confirmed
  `test_coopmat_linear_bench_spec014` runs and reports coopmat-vs-tiled
  GFLOP/s correctly at production shapes); no new C++ code is needed for
  User Story 1, only running the existing binary (rebuilt fresh from
  current HEAD, not reusing the pre-existing `_spec014` binary as-is, to
  rule out staleness per this session's own recurring stale-binary
  findings) and formatting its output into the report file.
- `test_sdpa_coopmat_bench.cpp` (User Story 2's harness, per Clarifications)
  exists as source and is written directly against `ComputeGraph` (not the
  `TestCase`/`ValueSpec` framework the linear harness uses), so wiring it
  into the CMake build is expected to be a small, mechanical addition
  (new executable target, same pattern as the other `test_coopmat_*_bench`
  targets already in `backends/vulkan/test/custom_ops/CMakeLists.txt`),
  not new benchmark logic.
- Clock pin and driver verification from `specs/015`'s already-established
  M5 EVT1 session are reused (not re-derived from scratch), since this
  feature runs in the same session/device state.
- This feature does not re-open or attempt to resolve `specs/015`'s Q11
  (why the e2e ETDump dispatch-confirmation method is unreliable) or Q12
  (the SDPA env-var wall-clock anomaly) -- it produces an independent,
  trustworthy data point via a different (already-proven-reliable)
  measurement method, which may inform a future investigation into Q11/Q12
  but does not itself root-cause them.
