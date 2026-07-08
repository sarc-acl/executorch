# Implementation Plan: M5 EVT1 Full Microbenchmark Suite — Stable Results Report

**Branch**: `020-run-existing-linear` | **Date**: 2026-07-06 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/020-run-existing-linear/spec.md`

## Summary

Run all three already-in-tree microbenchmark harnesses
(`test_coopmat_linear_bench`, `test_sdpa_coopmat_bench`,
`test_llama_baseline_bench`) on M5 EVT1, each invoked 3 separate times
end-to-end (not just relying on each harness's own internal 5-run
average from one invocation), and produce one consolidated report. No new
benchmark logic, shape tables, or CMake targets are needed — unlike
`specs/016`, which had to extend `test_coopmat_linear_bench`'s shape table
and wire up a missing CMake target, all three binaries are confirmed
already built and already registered in
`backends/vulkan/test/custom_ops/CMakeLists.txt`'s `add_operator_prototype`
list. The actual gaps closed here are: (1) `test_llama_baseline_bench` has
never been pushed to or run on M5 EVT1 at all; (2) `test_sdpa_coopmat_bench`
is built locally but not currently staged on-device; (3) neither
`test_coopmat_linear_bench` nor `test_sdpa_coopmat_bench` has ever been
invoked more than once per session on this target, so no cross-invocation
stability evidence exists yet for either.

## Technical Context

**Language/Version**: C++ (all three harnesses, unmodified — no source
changes are in this feature's scope) for the measurement; a short Python
aggregation script (matching the pattern already used for `specs/007`/
`010`/`016`'s report generation) to parse each harness's raw stdout into
per-case rows, compute the 3-invocation CoV per case, and render the
consolidated report.

**Primary Dependencies**:
- `backends/vulkan/test/custom_ops/test_coopmat_linear_bench.cpp` — already
  has the 1B/3B/8B shape table (`kShapes`, added in `specs/016`, still
  uncommitted per this workspace's known state); confirmed already built
  fresh (binary mtime newer than source) and already staged on-device as
  `test_coopmat_linear_bench_016` (size-identical to the current local
  build) — reused as-is unless a size check at execution time shows drift.
- `backends/vulkan/test/custom_ops/test_sdpa_coopmat_bench.cpp` — already
  wired into `CMakeLists.txt` (confirmed via direct file read, not a
  piped grep, per `.specify/memory/gotchas.md` G8) and already built
  locally; not yet staged on M5 EVT1 — needs a push, no rebuild.
- `backends/vulkan/test/custom_ops/test_llama_baseline_bench.cpp` — same
  build status as the SDPA bench (built locally, not staged); this is its
  first-ever M5 EVT1 run.
- `.shared-context/scripts/analyze_etdump_shaders.py` is NOT used here —
  these harnesses report dispatch/correctness via their own kernel-name
  capture and `BenchmarkResult` machinery, not ETDump (Principle VI's
  documented ETDump-attribution unreliability, `.specify/memory/gotchas.md`
  G6, does not apply to this feature's evidence path).
- `.shared-context/instruction-for-ai/build.md`'s Android cross-build
  recipe (Principle X) — consulted only if a rebuild does turn out to be
  needed at execution time; the current builds are believed current per
  the mtime check above, so this is a contingency, not an expected step.

**Storage**: Flat files under `specs/020-run-existing-linear/results/` —
`raw/` (one raw stdout capture per binary per invocation, 3 per harness =
9 files), and the consolidated report
(`results/microbenchmark-suite-report.md`).

**Testing**: No separate automated test suite. Each harness's own
dispatch-confirmation (kernel-name capture) and correctness-verdict
(PASSED/FAILED/SKIPPED per case, from each harness's built-in reference
comparison) output is the verification evidence, per constitution
Principle I/VI — this feature adds a cross-invocation aggregation layer
on top, it does not replace the harnesses' own checks.

**Target Platform**: M5 EVT1 (Samsung Exynos 2500 / Xclipse 970), Android
arm64 — constitution Principle II's sole active target.

**Project Type**: Measurement/reporting feature. Zero source changes to
any harness; one new aggregation script; results/report artifacts only.

**Performance Goals**: N/A — this feature *is* the performance
measurement; there is no separate performance target for the measurement
process itself.

**Constraints**: Clocks pinned (509/2730/663 MHz), verified via sysfs
readback, before any measurement (Principle VII) — correcting the
floating state left over from the stopped `specs/019` session first;
on-device driver identity verified against the known-good table
(Principle VIII) before any measurement.

**Scale/Scope**: 3 harness binaries × 3 invocations each = 9 full binary
runs. Case counts per invocation: linear bench ~24 perf cases (2 schemes ×
3 models × 4 shapes) plus its own correctness-only cases; SDPA bench 3
cases (one per model); baseline bench 96 perf cases (2 regimes × 2
storage × 2 schemes × 3 models × 8 ops). Aggregation compares each case's
value across its 3 invocations.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Check | Status |
|---|---|---|
| I. Correctness Before Performance | FR-003/004/005 require each harness's own dispatch/correctness verdict to accompany every reported case; a case that fails correctness or falls back to tiled is never reported as a coopmat win. | PASS |
| II. Samsung M5 EVT1 Sole Target | All measurement is on M5 EVT1; no MiniPC re-run in scope. | PASS |
| III. Explicit Eligibility Gating | No new eligibility-gating code — all three harnesses reuse existing gates (`can_use_q4gsw_coopmat`, `SDPA.cpp`'s coopmat gate) unchanged. | PASS (N/A, no new gate) |
| IV. Two-Tier Benchmarking | This feature is entirely tier-1 (shader microbenchmark); it does not claim or substitute for a tier-2 (model-level e2e) result. | PASS |
| V. Document Driver Workarounds | No new driver workaround expected; none of the three harnesses are being modified. | PASS (N/A) |
| VI. Verify With Tools, Never Assume | FR-003/004 require dispatch confirmed via each harness's own kernel-name capture — this feature does not introduce or rely on ETDump's known-unreliable full-graph attribution. | PASS |
| VII. Clock Discipline | FR-001 requires pinned-clock verification (correcting the leftover floating state from `specs/019`) before any measurement; FR-006's per-invocation spread reporting is itself an application of this principle's per-rep (not blended-mean) discipline. | PASS |
| VIII. Verify Driver Before Every Coopmat Measurement | FR-001 also covers driver-identity verification. | PASS |
| IX. Never Disclose Samsung-Internal Specifics Upstream | This feature's artifacts stay under `specs/`, not upstream-bound. | PASS (N/A) |
| X. Consult `instruction-for-ai` Before Acting | Plan cites `.shared-context/instruction-for-ai/build.md` as the reference to consult if a rebuild turns out to be needed — not skipped, just not expected given the mtime check already performed. | PASS |

No violations; Complexity Tracking not needed.

**Post-Phase-1 re-check**: `data-model.md`/`quickstart.md` introduced no
new gate risk — the aggregation script is read-only with respect to
harness source/binaries (it only parses their stdout), and every reported
field traces back to a dispatch/correctness check already required by
FR-003/004/005 or a pin/driver check already required by FR-001.
Constitution Check still PASSES across all ten principles.

## Project Structure

### Documentation (this feature)

```text
specs/020-run-existing-linear/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── checklists/
│   └── requirements.md  # /speckit-specify output, already validated
└── tasks.md              # Phase 2 output (/speckit-tasks, not this command)
```

### Source Code (repository root)

```text
backends/vulkan/test/custom_ops/
├── test_coopmat_linear_bench.cpp     # UNCHANGED: already has 1B/3B/8B shapes
├── test_sdpa_coopmat_bench.cpp       # UNCHANGED: already a build target
├── test_llama_baseline_bench.cpp     # UNCHANGED: already a build target
└── CMakeLists.txt                    # UNCHANGED: all three already registered

.shared-context/scripts/
└── aggregate_microbench_results.py   # NEW: parses 3x raw stdout per harness,
                                       # computes per-case cross-invocation CoV,
                                       # flags peer-relative outliers, renders
                                       # the consolidated report

specs/020-run-existing-linear/results/
├── raw/                               # 9 raw stdout captures (3 harnesses x 3 invocations)
└── microbenchmark-suite-report.md    # the one consolidated report
```

**Structure Decision**: No source changes anywhere — all three harnesses
and their CMake registration are confirmed already in place. This
feature's only new artifact is one aggregation script (placed in
`.shared-context/scripts/`, this workspace's canonical location for
location-independent tooling, per that directory's own README convention
already followed by `analyze_etdump_shaders.py` and `run_m5_full_sweep.py`)
plus its own `specs/020.../results/` directory, following the exact
precedent of `specs/007`/`010`/`016`.

## Complexity Tracking

*No violations -- table not needed.*
