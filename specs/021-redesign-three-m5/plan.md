# Implementation Plan: Unify M5 EVT1 Microbenchmark Structure, Shapes, and Statistics

**Branch**: `021-redesign-three-m5` | **Date**: 2026-07-07 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/021-redesign-three-m5/spec.md`

## Summary

Modify all three existing microbenchmark harnesses
(`test_coopmat_linear_bench.cpp`, `test_sdpa_coopmat_bench.cpp`,
`test_llama_baseline_bench.cpp`) so they (1) call `execute_test_cases()`
once per individual case instead of once per full case vector, printing
one shared `RESULT,...` line immediately after each call returns; (2)
this per-case calling pattern, applied to `test_llama_baseline_bench`
(organized as an outer per-model loop for output grouping, per research.md
Decision 3), directly eliminates its confirmed deterministic OOM by
bounding peak memory to a single case's own tensors instead of all 192
(or even one model's 64) at once; (3) linear bench measures real `prefill(M=2048)` and
`decode(M=1)` regimes instead of the `M=1024` compromise; (4) SDPA bench
reports its two real sub-shaders (`qk`, `av`) separately plus the
existing combined `total`, and adds a `decode(S=1)` case; (5) the
aggregation script is rewritten around one shared parser instead of
three harness-specific ones. All changes are confined to the three
harness `.cpp` files and the Python aggregator — no shared `utils.cpp`,
shader, dispatch-gate, or `CMakeLists.txt` changes.

## Technical Context

**Language/Version**: C++17 (all three harnesses, matching
`CMakeLists.txt`'s existing `CXX_STANDARD 17`); Python 3 (aggregation
script, no new dependencies beyond the standard library already used by
`aggregate_microbench_results.py`).

**Primary Dependencies**:
- `backends/vulkan/test/custom_ops/utils.h`/`utils.cpp` — `TestCase`,
  `ValueSpec`, `BenchmarkResult`, `execute_test_cases()` (read-only
  dependency; NOT modified — `execute_test_cases()` is called more times
  with smaller case sets per call, not changed internally).
- `backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp`'s
  `is_gemv_case` short-circuit in `pick_linear_qw_shader`/
  `pick_linear_dqa_qw_shader` (read-only; confirmed via direct source
  read during task planning — NOT `can_use_q4gsw_coopmat()`'s
  `M % tile_m != 0` check, which is never reached for decode at all:
  `if (weight_is_4bit && is_gemv_case) { kernel_name += "_coop"; }`
  fires first and dispatches a dedicated `_coop` kernel, which is what
  makes linear bench's new decode cases honestly report
  `not_applicable`, not `fallback_tiled`).
- `backends/vulkan/runtime/graph/ops/impl/SDPA.cpp`'s `is_gemv` /
  `sdpa_coopmat_device_ok()` gate (read-only; structurally identical
  short-circuit to `QuantizedLinear.cpp`'s `is_gemv_case` above — makes
  SDPA bench's new decode case honestly report `not_applicable`).
- `.shared-context/scripts/aggregate_microbench_results.py` (from
  `specs/020`) — rewritten parser section; the stability-aggregation
  (`aggregate()`, peer-relative outlier logic) and report-rendering
  structure are reused, only the per-harness parsing functions
  (`parse_linear`/`parse_sdpa`/`parse_baseline`) collapse into one
  `parse_result_line()`.

**Storage**: Flat files under `specs/021-redesign-three-m5/results/` —
`raw/` (3 invocations × 3 harnesses = 9 captures, now uniform
`RESULT,...` format), and the consolidated report
(`results/microbenchmark-suite-report.md`).

**Testing**: No separate automated test suite. Each harness's own
existing correctness machinery (small-shape reference comparison,
`PASSED`/`FAILED`/`SKIPPED` verdicts) is preserved unchanged and is the
verification for FR-001–FR-008 (constitution Principle I/VI) — this
feature changes what gets *printed* and *when*, and adds new *cases*
(decode regime), but does not touch how correctness itself is computed
for any existing case.

**Target Platform**: M5 EVT1 (Samsung Exynos 2500 / Xclipse 970), Android
arm64 — constitution Principle II's sole active target.

**Project Type**: Benchmark-harness refactor + one new measurement axis
(decode regime) per harness. No new build targets (all three already
registered in `CMakeLists.txt`), no shader changes.

**Performance Goals**: N/A — this feature changes what is measured and
how it's reported, not a performance target for the measurement process
itself.

**Constraints**:
- `test_llama_baseline_bench`'s peak memory during any single
  `execute_test_cases()` call must stay well under M5 EVT1's ~11GB total
  RAM — calling `execute_test_cases()` once per individual case
  (research.md Decision 8, required for FR-001's per-case immediate
  printing) bounds the worst case to one case's own tensors (~525MB, a
  `lm_head` prefill case), down from the current ~6.3GB (12 `lm_head`
  cases across all 3 models materialized simultaneously in one call).
  Per-model grouping (Decision 3) is retained for output organization but
  is not itself the memory-safety mechanism.
- No change to `execute_test_cases()`/`BenchmarkResult` in `utils.cpp`
  (shared by ~15 other prototype benchmark binaries) — confirmed via
  direct read this session that all needed behavior changes (per-case
  granularity, incremental printing) can be done entirely in each
  harness's own `main()`, by calling the existing `execute_test_cases()`
  API once per case instead of once per full case vector.
- Clocks pinned (509/2730/663 MHz) and driver identity verified before
  any measurement (Principles VII/VIII), matching `specs/020`'s already-
  established session-start checklist.

**Scale/Scope**: Linear bench's case count roughly doubles (24 → 48 perf
cases: 2 regimes × 2 schemes × 3 models × 4 shapes); SDPA bench grows
from 3 rows (1 per model) to 12 (3 variants × prefill + 1 decode, per
model — 4 rows/model × 3 models); baseline bench's 192 cases are
unchanged in count, only in execution granularity — 192 individual
`execute_test_cases()` calls (one per case) instead of 1 call with all
192, organized under a 3-iteration outer loop (one per model) purely for
output grouping.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Check | Status |
|---|---|---|
| I. Correctness Before Performance | FR-001–FR-008 preserve each harness's existing correctness verification unchanged; new decode-regime cases reuse the same reference-comparison machinery (small shapes still get a real reference check, oversized perf shapes are still `SKIPPED` by the harness's own existing size-cap logic — unchanged). | PASS |
| II. Samsung M5 EVT1 Sole Target | All measurement remains M5 EVT1; no MiniPC re-run in scope. | PASS |
| III. Explicit Eligibility Gating | No new eligibility-gating code is introduced — this feature only *observes and reports* the existing `can_use_q4gsw_coopmat`/`SDPA.cpp` gates' behavior at new shapes (decode), per FR-011. | PASS (N/A, no new gate) |
| IV. Two-Tier Benchmarking | Entirely tier-1 (shader microbenchmark); does not claim or substitute for a tier-2 e2e result. | PASS |
| V. Document Driver Workarounds | No driver workaround expected; no shader/dispatch code is touched. | PASS (N/A) |
| VI. Verify With Tools, Never Assume | FR-002/FR-006/FR-008 require the three-way `dispatch_status` to come from the harness's own kernel-name capture at run time, never assumed from reading gate source alone — the same discipline `specs/016`/`020` already established. | PASS |
| VII. Clock Discipline | Measurement runs (User Story 2's validation, and any final full sweep) pin clocks and verify via sysfs, per `specs/020`'s already-established procedure — this feature does not change that procedure. | PASS |
| VIII. Verify Driver Before Every Coopmat Measurement | Same pre-flight check as `specs/020`, reused unchanged. | PASS |
| IX. Never Disclose Samsung-Internal Specifics Upstream | Artifacts stay under `specs/`, not upstream-bound. | PASS (N/A) |
| X. Consult `instruction-for-ai` Before Acting | Any rebuild follows `.shared-context/instruction-for-ai/build.md`'s documented Android cross-build sequence, per `specs/020`'s precedent (gotcha G1: don't skip `--target install`'s two-step rebuild). | PASS |

No violations; Complexity Tracking not needed.

**Post-Phase-1 re-check**: `data-model.md`/`quickstart.md` introduce no
new gate risk — the unified `RESULT,...` schema and per-model batching
are both purely additive/organizational changes to existing, already-
verified measurement and correctness paths. Constitution Check still
PASSES across all ten principles.

## Project Structure

### Documentation (this feature)

```text
specs/021-redesign-three-m5/
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
├── test_coopmat_linear_bench.cpp     # MODIFIED: regime axis (prefill/decode) replaces kM=1024;
│                                     #           unified RESULT,... print per case
├── test_sdpa_coopmat_bench.cpp       # MODIFIED: qk/av/total split; decode(S=1) case added;
│                                     #           unified RESULT,... print per case
├── test_llama_baseline_bench.cpp     # MODIFIED: one execute_test_cases() call per individual
│                                     #           case (not per model/batch), organized under a
│                                     #           per-model outer loop; RESULT,... printed per case
└── CMakeLists.txt                    # UNCHANGED: all three targets already registered

.shared-context/scripts/
└── aggregate_microbench_results.py   # MODIFIED: single parse_result_line() replaces the three
                                       # harness-specific parsers; reconciliation section states
                                       # the linear shape-basis change explicitly

specs/021-redesign-three-m5/results/
├── raw/                               # 9 raw captures (3 harnesses x 3 invocations), unified format
└── microbenchmark-suite-report.md
```

**Structure Decision**: No new files, no new build targets, no new
top-level directories — every change is a modification to one of the
three already-existing harness `.cpp` files, the already-existing
aggregation script, and this feature's own `specs/021.../results/`
directory, following the exact precedent of `specs/016`/`020`.

## Complexity Tracking

*No violations -- table not needed.*
