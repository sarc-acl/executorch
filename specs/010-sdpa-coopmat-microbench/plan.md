# Implementation Plan: SDPA Coopmat Correctness + Microbenchmark

**Branch**: `010-sdpa-coopmat-microbench` | **Date**: 2026-07-05 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/010-sdpa-coopmat-microbench/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Correctness-verify and microbenchmark the SDPA cooperative-matrix (WMMA)
prefill path (`sdpa_compute_attn_weights_coopmat`/`sdpa_compute_out_coopmat`,
`SDPA.cpp`) that was imported from `yanwen/quant-dev-active` in a prior
session and has never passed a correctness check in this repository.
Planning found the ported "bench harness" (`test_coopmat_attention_bench.cpp`)
actually tests an unrelated shader family (the generic `matmul_coopmat` tile
sweep applied to attention-shaped GEMMs, not the real SDPA op) -- so this
feature extends the existing ATen-referenced `sdpa_test.cpp` for
correctness (genuinely new tile-aligned coverage) and authors a new, timed
microbenchmark harness for the actual speedup question, following this
workstream's established two-tier discipline exactly as `007` did for the
quantized-linear coopmat path.

## Technical Context

**Language/Version**: C++ (a new `TEST` case in the existing
`sdpa_test.cpp`, GTest-based; a new test-owned microbenchmark file under
`backends/vulkan/test/custom_ops/`) for the measurement; Python 3.10+
(`uv`-managed `.venv`) for the comparison/report script, matching every
prior feature in this workstream.

**Primary Dependencies**:
- `backends/vulkan/test/op_tests/sdpa_test.cpp`'s existing
  `test_vulkan_sdpa`/`test_reference_sdpa` ATen-ground-truth machinery --
  extended with one new tile-aligned, `Buffer`-storage case, not rebuilt
  from scratch (research.md Decision 2).
- `SDPA.cpp`'s existing `ET_VK_SDPA_COOPMAT`/`ET_VK_DISABLE_COOPMAT` opt-in
  toggle, reused as-is as this feature's tiled-vs-coopmat switch (already
  implemented by the imported code).
- This workstream's existing `BenchmarkResult`/`execute_test_cases` timing
  framework (`backends/vulkan/test/custom_ops/utils.h`), reused for the new
  microbenchmark harness, mirroring `test_coopmat_linear_bench.cpp`'s
  pattern.
- `009`'s `tag_memory_meta_pass.py` fix (already applied, uncommitted in the
  working tree) -- a hard prerequisite, since SDPA's coopmat gate requires
  the same `Buffer` storage that fix made reachable (research.md Decision 6).

**Storage**: Flat files -- new SPIR-V-inspection output under
`results/spirv/`, a new microbenchmark raw log under `results/raw/`, and
the final report at `results/sdpa-coopmat-microbench-report.md`.

**Testing**: The correctness check itself (new `sdpa_test.cpp` case, both
env-toggle states) *is* the test suite for this feature, per this
workstream's established pattern -- no separate, additional automated test
layer beyond it and the microbenchmark harness's own dispatch/timing
verification.

**Target Platform**: `rocky-ryzen` MiniPC (RDNA3 integrated GPU) -- this
workstream's primary local validation platform, matching every prior
microbenchmark-tier feature.

**Project Type**: Small, genuinely new test coverage (one `op_tests` case,
one new `custom_ops` microbenchmark file) plus reuse of already-implemented
production dispatch code (`SDPA.cpp`'s coopmat path) -- no new production
shader or dispatch-code changes anticipated, unless the correctness check
in User Story 1 finds a real bug (Assumptions, mirroring `007`'s own
mid-implementation discovery precedent).

**Performance Goals**: N/A -- this feature measures performance; it does
not carry its own target.

**Constraints**:
- No performance number is reported for the SDPA coopmat path until the new
  correctness check (research.md Decision 2) and SPIR-V inspection both
  pass (constitution Principle I, non-negotiable).
- Scope is one configuration per target model (three total), not the
  constitution's default six -- SDPA's shape/dispatch is scheme-independent
  (spec.md Assumptions, research.md), a deliberate, justified narrowing.
- `test_coopmat_attention_bench.cpp` and the matmul tile-sweep code that
  arrived in the same import are out of scope -- confirmed to test an
  unrelated shader family (research.md Decision 1); this feature does not
  build, fix, or extend them.
- No concurrent GPU load during any capture (established workstream
  discipline).

**Scale/Scope**: 3 target models x 1 representative prefill SDPA
configuration each = 3 microbenchmark comparison cases, plus 2 new
correctness cases (one per coopmat shader) at a small tile-aligned shape.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against `.specify/memory/constitution.md` (v1.4.0):

- **I. Correctness Before Performance (NON-NEGOTIABLE)**: PASS, central to
  this feature's design -- FR-001 requires genuinely new correctness
  coverage (existing tests, including the misleadingly-named
  `test_coopmat_attention_bench.cpp`, cover neither shape nor shader) before
  any timing is trusted.
- **II. Samsung RDNA3 iGPU Is the Target, Not a Fallback**: PASS with scope
  note -- `rocky-ryzen` MiniPC only, consistent with every prior
  microbenchmark-tier feature; on-device Samsung/Xclipse validation is a
  future feature, not silently skipped. Notably, `SDPA.cpp`'s coopmat gate
  already omits the `!is_integrated_gpu()` exclusion specifically because
  Xclipse is the real target (existing code, unchanged by this feature).
- **III. Explicit Eligibility Gating, Safe Fallback Always**: PASS. This
  feature observes and verifies the existing `sdpa_coopmat_device_ok`/
  `sdpa_buf_half`/`sdpa_cm_aligned` gates; it does not modify eligibility
  logic (unless User Story 1 finds a real bug requiring one, per
  Assumptions, applied only with explicit authorization).
- **IV. Two-Tier, Statistically Sound Benchmarking**: PASS, tier-1 (shader
  microbenchmark) only by explicit scope -- every reported number carries
  iteration count and stdev (FR-004); tier-2 (model-level) validation of
  this same path is explicitly out of scope, mirroring how `007` preceded
  `009` for the linear-coopmat path.
- **V. Document Every Driver Workaround at the Point of Use**: N/A --
  no new driver workaround anticipated; the existing Xclipse-specific
  workarounds already documented in `SDPA.cpp`'s imported code (e.g. the
  `num_k_chunks` spec-constant loop bound, matching the linear coopmat
  path's established pattern) are unchanged by this feature.
- **VI. Verify With Tools, Never Assume**: PASS, central to this feature --
  the correctness check (FR-001), SPIR-V inspection (FR-002), and
  kernel-name dispatch confirmation (FR-003) are exactly this principle's
  three clauses, applied for the first time to the SDPA coopmat path in
  this workstream.

No violations identified. Complexity Tracking is not needed -- this is
small, additive test coverage for already-implemented dispatch code, not
new architecture.

*Post-Phase-1 re-check*: Phase 1's data model keeps `dispatch_status`,
`correctness_status`, and `spirv_verified` as separate, explicit fields
(never folded into a single pass/fail or into the timing number itself), so
Principles I and VI stay enforced by the data structure, not just by
convention -- same discipline `007`/`009` already established.

## Project Structure

### Documentation (this feature)

```text
specs/010-sdpa-coopmat-microbench/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── sdpa-coopmat-microbench-schema.md
└── tasks.md              # Phase 2 output (/speckit-tasks, not this command)
```

### Source Code (repository root)

```text
backends/vulkan/test/op_tests/sdpa_test.cpp   # MODIFIED: one new tile-aligned,
  # Buffer-storage VulkanSDPATest case (research.md Decision 2) -- genuinely
  # new coverage, no existing case is tile-aligned to the coopmat gate
backends/vulkan/test/op_tests/CMakeLists.txt  # existing, unmodified; this
  # feature configures it as a new sub-build (research.md Decision 7),
  # doesn't change its content

backends/vulkan/test/custom_ops/   # NEW: test_sdpa_coopmat_bench.cpp --
  # timed microbenchmark harness (research.md Decision 3), toggling
  # ET_VK_SDPA_COOPMAT per the existing SDPA.cpp opt-in mechanism, at each
  # target model's real prefill shape
backends/vulkan/test/custom_ops/CMakeLists.txt  # MODIFIED: new
  # add_operator_prototype(test_sdpa_coopmat_bench) entry

specs/010-sdpa-coopmat-microbench/
├── scripts/
│   └── compare_sdpa_coopmat.py   # new: loads the microbenchmark raw log,
│                                   # computes per-model speedup/significance,
│                                   # renders the report
└── results/
    ├── raw/                # new microbenchmark capture log
    ├── spirv/              # spirv-dis output for both coopmat shaders
    └── sdpa-coopmat-microbench-report.md

# Explicitly out of scope, not modified:
backends/vulkan/test/custom_ops/test_coopmat_attention_bench.cpp  # tests an
  # unrelated shader family (research.md Decision 1); left as-is, unwired
backends/vulkan/runtime/graph/ops/impl/GemmCoopmat.{h,cpp}  # the matmul
  # tile-sweep thread that arrived in the same import; untouched here
```

**Structure Decision**: Same lightweight structure as `007`/`009`: one new
Python analysis script under this feature's own `scripts/`, plus small,
targeted test-code additions -- one new case in an existing correctness
suite, one new test-owned microbenchmark file. No new production dispatch
code is planned; the SDPA coopmat path already exists and this feature's
job is to verify and measure it, mirroring exactly how `007` treated the
already-implemented quantized-linear coopmat shaders.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations -- this section is intentionally empty.
