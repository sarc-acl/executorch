# Implementation Plan: WMMA Coopmat Improvement Microbenchmark

**Branch**: `007-wmma-improvement-microbench` | **Date**: 2026-07-04 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/007-wmma-improvement-microbench/spec.md`

## Summary

Measure how much the already-implemented, double-buffered 128x64-tile WMMA/
coopmat quantized-linear shaders (`linear_q4gsw_coopmat`,
`linear_dq8ca_q4gsw_coopmat`) actually speed up the operations `003` already
identified as candidates, at the shader-microbenchmark tier, across all six
model/scheme configurations. No shader or dispatch-code changes are required
-- the existing `test_llama_baseline_bench.cpp` harness already reads real
per-model shapes at Buffer storage and already respects the
`ET_VK_FORCE_TILED_LINEAR` env var read by `can_use_q4gsw_coopmat()`; running
it a second time with that env var *unset* lets the natural coopmat dispatch
occur (verified: every real per-model shape is coopmat-tile-aligned for both
the 4w and 8da4w variants). The tiled-baseline half of the comparison is
`004`'s already-captured Buffer-storage numbers -- no re-capture needed.

## Technical Context

**Language/Version**: C++ (existing `test_llama_baseline_bench.cpp` harness,
unmodified) for the measurement; Python 3.10+ (`uv`-managed `.venv`) for the
comparison/report script, matching every prior feature in this workstream.

**Primary Dependencies**:
- `backends/vulkan/test/custom_ops/test_llama_baseline_bench.cpp` (already
  built, already committed by this same workstream) -- run twice, toggled
  purely by the presence/absence of the `ET_VK_FORCE_TILED_LINEAR`
  environment variable read in `QuantizedLinear.cpp`'s
  `can_use_q4gsw_coopmat()`. No source change needed.
- `spirv-dis` (Vulkan SDK, confirmed present at
  `~/vulkansdk/1.4.341.1/x86_64/bin/spirv-dis`) against the already-compiled
  `.spv` artifacts under `cmake-out-vk/vulkan_compute_shaders/` -- no extra
  build step, these are produced by the normal build.
- `specs/003-wmma-shader-candidates`'s classification JSON (candidate list +
  `pct_of_phase` weights) and `specs/004-linear-storage-comparison`'s
  Buffer-storage raw log (tiled-baseline reference) as read-only inputs.

**Storage**: Flat files -- new raw capture log under this feature's own
`results/raw/`, a SPIR-V-inspection summary under `results/spirv/`, and the
final report at `results/wmma-improvement-report.md`. No re-capture of `004`'s
already-published tiled-baseline data.

**Testing**: No separate automated test suite -- this feature's correctness
signal (kernel-dispatch check + SPIR-V instruction-presence check + citing
existing generic-shape correctness tests, per the spec's Clarification
Session) *is* the verification, matching how `001`/`004`/`006` validated
their own work inline rather than via a dedicated test phase.

**Target Platform**: `rocky-ryzen` MiniPC (RDNA3 integrated GPU) -- this
workstream's primary local validation platform (constitution Principle II).
Samsung/Android on-device validation is out of scope for this feature, same
as `001`/`004`; a future feature would extend this study on-device.

**Project Type**: Originally planned as measurement/reporting only with zero
production code changes. **Revised during implementation**: US1's own proof
step (T005/T006) found that `4w`'s coopmat path was unreachable from any
registered op -- not a measurement question, a real dispatch-wiring gap
affecting the shipped model. Fixing it became in-scope (user decision,
2026-07-04): a new `linear_q4gsw` function in `QuantizedLinear.cpp` routing
through the already-existing, already-compiled `add_linear_qw_node`/coopmat
path, replacing `et_vk.linear_q4gsw.default`'s registration (previously
`Q4gswLinear.cpp`'s non-coopmat `q4gsw_linear`). See `research.md` Decision 8
for the full root cause and fix.

**Performance Goals**: N/A -- this feature measures performance; it does not
carry its own target (that's `005`'s role).

**Constraints**:
- Correctness-confidence bar (Clarification Q1): kernel-dispatch check +
  SPIR-V cooperative-matrix-instruction-presence check + citing existing
  generic-shape correctness tests. Authoring new production-shape correctness
  tests is explicitly out of scope.
- **`lm_head` is excluded from this study's measured op set** (see
  `research.md` Decision 3) -- a real, non-obvious discovery made during
  planning: `test_llama_baseline_bench.cpp` (the harness `001`/`004` already
  built) runs `lm_head` at a synthetic `M=2048` during the "prefill" regime
  for measurement convenience, but `003`'s classification data (grounded in
  real ETDump captures of the actual exported model) shows `lm_head` is
  *always* `M=1` in production regardless of phase. Measuring a "WMMA
  speedup" for a shape that never occurs in the deployed model -- and whose
  `N=128256` would dominate a time-weighted average given its size -- would
  misrepresent the real-world question this study exists to answer. This is
  a forward-looking decision for `007` only; it does not retroactively
  change `001`/`004`'s already-published reports.
- No concurrent GPU load during capture (established workstream discipline).
- RGA (Radeon GPU Analyzer) is **not installed** on this machine (checked:
  absent from `PATH` and common install locations) -- ISA/occupancy-level
  analysis is out of scope for this feature; `spirv-dis`-based
  instruction-presence checking is what's actually performed. Flagged here
  rather than silently assumed away, per constitution Principle VI.

**Scale/Scope**: 3 models x 2 int4 schemes x 7 ops (`wq`, `wk`, `wv`, `wo`,
`w1_gate`, `w3_up`, `w2_down` -- excludes `lm_head` per above) = 42 measured
tiled-vs-WMMA pairs, all at prefill `M=2048`, Buffer storage.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against `.specify/memory/constitution.md` (Vulkan Cooperative-Matrix
(WMMA) GEMM Constitution, v1.4.0):

- **I. Correctness Before Performance (NON-NEGOTIABLE)**: PASS. Every
  reported number is backed by the three-part correctness-confidence check
  (kernel-dispatch + SPIR-V + existing generic-shape correctness tests) from
  Clarification Q1 / FR-007. No perf number is reported without it.
- **II. Samsung RDNA3 iGPU Is the Target, Not a Fallback**: PASS with scope
  note. This feature validates on `rocky-ryzen` MiniPC only, consistent with
  `001`/`004`'s precedent that MiniPC precedes Android validation; on-device
  Samsung/Xclipse validation of these same numbers is explicitly left to a
  future feature, not silently skipped.
- **III. Explicit Eligibility Gating, Safe Fallback Always**: PASS. This
  study observes and verifies the existing `can_use_q4gsw_coopmat()` gate; it
  does not modify eligibility logic. The kernel-name check (FR-004) is
  exactly how this study confirms the gate did what it claims.
- **IV. Two-Tier, Statistically Sound Benchmarking**: PASS, tier-1 (shader
  microbenchmark) only by explicit scope -- every reported number carries
  iteration count and stdev (FR-003), matching `001`/`004`. Tier-2
  (model-level) WMMA impact is explicitly out of scope for this feature,
  mirroring how `004` (tier-1) preceded `006` (tier-2) for storage.
- **V. Document Every Driver Workaround at the Point of Use**: N/A -- this
  feature introduces no new driver workaround.
- **VI. Verify With Tools, Never Assume**: PASS. This is the first feature to
  explicitly operationalize this principle: kernel-dispatch verification via
  the harness's own `kernel` CSV field, and SPIR-V inspection via `spirv-dis`
  confirming genuine `OpCooperativeMatrix*KHR` instructions in the compiled
  shader (already spot-verified present during planning). The principle's
  ETDump clause applies to tier-2 studies only, which this feature is not;
  RGA's absence is documented above rather than silently ignored.

No violations. Complexity Tracking is not needed.

## Project Structure

### Documentation (this feature)

```text
specs/007-wmma-improvement-microbench/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── wmma-improvement-report-schema.md
└── tasks.md              # Phase 2 output (/speckit-tasks, not this command)
```

### Source Code (repository root)

No new production source files. This feature reuses, unmodified:

```text
backends/vulkan/test/custom_ops/test_llama_baseline_bench.cpp   # existing harness, run twice (env var toggle)
backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp       # existing ET_VK_FORCE_TILED_LINEAR read (unchanged)
cmake-out-vk/vulkan_compute_shaders/*coopmat*_buffer_buffer_half.spv  # already-compiled SPIR-V, inspected read-only

specs/007-wmma-improvement-microbench/
├── scripts/
│   └── compare_wmma.py   # new: loads 004's Buffer tiled baseline + this feature's new WMMA capture, computes per-op and time-weighted overall speedup, renders the report
└── results/
    ├── raw/               # new WMMA-dispatch capture log (RESULT CSV lines, natural dispatch)
    ├── spirv/              # spirv-dis output for the two inspected shader variants
    └── wmma-improvement-report.md
```

**Structure Decision**: Same lightweight structure as `004`/`006`: one new
Python analysis script under this feature's own `scripts/`, reading two
existing datasets (`004`'s tiled baseline, `003`'s candidate/weight data) plus
one newly-captured raw log, and rendering a single report. No new C++ or
shader code -- the entire measurement mechanism already exists in-tree from
prior features in this workstream.
