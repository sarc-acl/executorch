# Implementation Plan: M5 EVT1 Linear + SDPA Coopmat Microbenchmark Validation

**Branch**: `016-m5-linear-sdpa-microbench` | **Date**: 2026-07-06 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/016-m5-linear-sdpa-microbench/spec.md`

## Summary

Re-run the same microbenchmark methodology `specs/007` (linear) and
`specs/010` (SDPA) used on the `rocky-ryzen` MiniPC, on the real M5 EVT1
target, and produce two reports in the identical format. Planning found
two concrete gaps versus a literal re-run: (1) this repo's current
`test_coopmat_linear_bench.cpp` only has LLaMA 3.1 8B's shapes hardcoded
(`kShapes`, K/N pairs for `wq/wk/wv/wo/w1_gate/w2_down/w3_up`) -- 1B's and
3B's shapes need to be added as additional, model-tagged entries to reach
`007`'s full 42-case (3 models x 2 schemes x 7 ops) set; (2) `specs/010`'s
actual report-producing harness, `test_sdpa_coopmat_bench.cpp`, exists as
source in this repo but is not wired into the Android CMake build (see
spec Clarifications) -- it needs one new build target, mirroring the
existing `test_coopmat_linear_bench` target's pattern, no new benchmark
logic. Both harnesses' correctness/dispatch-confirmation machinery is
otherwise reused as-is; this feature adds no new eligibility-gating or
shader code.

## Technical Context

**Language/Version**: C++ (existing `test_coopmat_linear_bench.cpp`,
extended with 1B/3B shape entries; `test_sdpa_coopmat_bench.cpp`, wired
into the build, unmodified otherwise) for the measurement; a short Python
or shell aggregation step, matching every prior microbenchmark feature in
this workstream (`007`, `010`), to turn raw harness output into the two
report tables.

**Primary Dependencies**:
- `backends/vulkan/test/custom_ops/test_coopmat_linear_bench.cpp` --
  `kShapes` (currently 8B-only, 4 K/N pairs) extended with 1B
  (dim=2048/ffn=8192) and 3B (dim=3072/ffn=8192) K/N pairs, each tagged
  with a model label so the aggregation step can group rows by model; the
  op-classification (`wq`/`wk`/`wv`/`wo`/`w1_gate`/`w2_down`/`w3_up`) and
  tiled-vs-coopmat/correctness/dispatch logic are reused unchanged.
- `backends/vulkan/test/custom_ops/test_sdpa_coopmat_bench.cpp` -- built
  directly against `ComputeGraph` (research.md Decision 8 in `specs/010`),
  already implements the exact `sdpa_compute_attn_weights_coopmat`/
  `sdpa_compute_out_coopmat` vs tiled timing this feature needs; only
  needs a new executable target in
  `backends/vulkan/test/custom_ops/CMakeLists.txt` (same pattern as the
  existing `test_coopmat_linear_bench`/`test_coopmat_attention_bench`
  targets) plus an Android cross-build + push to M5 EVT1, per
  `.shared-context/instruction-for-ai/build.md` (constitution Principle X
  -- read that doc's Android cross-build recipe before building, don't
  re-derive cmake flags from scratch).
- `SDPA.cpp`'s existing `ET_VK_SDPA_COOPMAT` opt-in toggle and
  `QuantizedLinear.cpp`'s existing `ET_VK_FORCE_TILED_LINEAR` toggle
  (confirmed real and working this session via a direct wall-clock A/B),
  reused as-is as each harness's tiled-vs-coopmat switch.
- Each target model's `params.json` (`/local/yanwen.xu/models/<id>/original/`)
  as the source of truth for `dim`/`ffn_dim_multiplier`/`multiple_of`/
  `n_heads`/`n_kv_heads` -- the same derivation this session already used
  for `specs/015`'s dispatch-gate diagnostics, not re-guessed.

**Storage**: Flat files under `specs/016-m5-linear-sdpa-microbench/results/`
-- `raw/` (per-run timing logs), `spirv/` (SPIR-V-inspection output
confirming genuine `OpCooperativeMatrix*KHR` instructions, reused from
`specs/007`/`010`'s existing citations where the shader is unchanged, or
freshly captured if this session's investigation changed the compiled
output), and the two final reports
(`linear-coopmat-microbench-report.md`, `sdpa-coopmat-microbench-report.md`).

**Testing**: No separate automated test suite -- each harness's own
dispatch (kernel-name capture) and correctness (existing correctness-shape
coverage / SPIR-V inspection) checks are the verification, matching how
`specs/007`/`010` validated their own work, per constitution Principle I/VI.

**Target Platform**: M5 EVT1 (Samsung Exynos 2500 / Xclipse 970), Android
arm64 cross-build -- constitution Principle II's sole active target. This
is the one deliberate difference from `specs/007`/`010`'s MiniPC target;
everything else about the methodology is unchanged.

**Project Type**: Measurement/reporting feature. Two small, additive C++
changes (shape-table extension; one new CMake build target) -- no changes
to shader source, eligibility-gating code, or dispatch logic.

**Performance Goals**: N/A -- this feature *is* the performance
measurement; there is no separate performance target for the measurement
process itself.

**Constraints**: Clock pins verified bound via GFLOP/s cross-check
(constitution Principle VII) and on-device driver identity verified
(Principle VIII) before any timing is trusted -- reusing `specs/015`'s
already-established M5 EVT1 session state per this feature's Assumptions,
re-verified fresh if the device has rebooted since.

**Scale/Scope**: 42 linear cases (3 models x 2 schemes x 7 ops) + 3 SDPA
cases (one per model) = 45 total benchmark cases across both reports.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Check | Status |
|---|---|---|
| I. Correctness Before Performance | FR-004 requires a correctness citation (existing correctness-shape coverage or SPIR-V inspection) alongside every timing row -- no perf number stands alone. | PASS |
| II. Samsung M5 EVT1 Sole Target | This feature explicitly targets M5 EVT1, not a re-run on `rocky-ryzen` MiniPC (which stays archived/historical per Principle II). | PASS |
| III. Explicit Eligibility Gating | No new eligibility-gating code -- reuses `can_use_q4gsw_coopmat`/`SDPA.cpp`'s existing gates unchanged. | PASS (N/A, no new gate) |
| IV. Two-Tier Benchmarking | This feature is entirely tier-1 (shader microbenchmark); it does not claim or substitute for a tier-2 (model-level e2e) result -- `specs/015` remains the only e2e-tier claim, now flagged as needing its own re-verification (Q11/Q12). | PASS |
| V. Document Driver Workarounds | No new driver workaround expected; if the CMake wiring for `test_sdpa_coopmat_bench` or the Android build hits one, it will be documented inline per this principle. | PASS (contingent) |
| VI. Verify With Tools, Never Assume | FR-003 requires dispatch confirmed via each harness's own kernel-name capture, not ETDump's now-unreliable full-graph method and not the eligibility-gate code alone. | PASS |
| VII. Clock Discipline | FR-007 requires the GFLOP/s pin-verification cross-check before any timing is trusted. | PASS |
| VIII. Verify Driver Before Every Coopmat Measurement | FR-007 also covers driver-identity verification. | PASS |
| IX. Never Disclose Samsung-Internal Specifics Upstream | This feature's artifacts stay under `specs/`, not upstream-bound; no PR is prepared by this feature. | PASS (N/A) |
| X. Consult `instruction-for-ai` Before Acting | Plan explicitly cites `.shared-context/instruction-for-ai/build.md` as the Android cross-build reference to consult before wiring/building the new SDPA bench target. | PASS |

No violations; Complexity Tracking not needed.

**Post-Phase-1 re-check**: `data-model.md`/`contracts/`/`quickstart.md`
introduced no new gate risk -- the two C++ changes stayed additive (shape
table extension, new CMake target) with no new eligibility-gating logic,
and every report field traces back to a dispatch/correctness/pin/driver
check already required by FR-003/004/007. Constitution Check still PASSES
across all ten principles.

## Project Structure

### Documentation (this feature)

```text
specs/016-m5-linear-sdpa-microbench/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md         # Phase 1 output
├── quickstart.md         # Phase 1 output
├── checklists/
│   └── requirements.md  # /speckit-specify output, already validated
└── tasks.md              # Phase 2 output (/speckit-tasks, not this command)
```

### Source Code (repository root)

```text
backends/vulkan/test/custom_ops/
├── test_coopmat_linear_bench.cpp     # MODIFIED: kShapes extended with 1B/3B entries + model tag
├── test_sdpa_coopmat_bench.cpp       # UNCHANGED: existing source, not yet a build target
└── CMakeLists.txt                    # MODIFIED: new test_sdpa_coopmat_bench executable target

specs/016-m5-linear-sdpa-microbench/results/
├── raw/                               # per-run timing logs (both harnesses)
├── spirv/                             # SPIR-V inspection output
├── linear-coopmat-microbench-report.md
└── sdpa-coopmat-microbench-report.md
```

**Structure Decision**: No new top-level directories or components -- this
feature extends two existing files under `backends/vulkan/test/custom_ops/`
(one shape-table addition, one new CMake target) and writes its own
results under its own `specs/016.../results/` directory, following the
exact precedent of `specs/007`/`specs/010`.

## Complexity Tracking

*No violations -- table not needed.*
