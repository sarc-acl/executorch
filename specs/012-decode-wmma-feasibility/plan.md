# Implementation Plan: Decode Shader WMMA Acceleration

**Branch**: `012-decode-wmma-feasibility` | **Date**: 2026-07-05 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/012-decode-wmma-feasibility/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Determine whether decode's per-token linear GEMV kernel (`linear_q4gsw_coop`,
`M=1`) is compute-bound or memory-bandwidth-bound on the `rocky-ryzen`
MiniPC, via a roofline analysis (spec.md Clarifications, Option A), before
committing to designing a new WMMA-capable decode shader. Planning already
performed this analysis (research.md Decisions 1-3), using published
device specs (AMD Radeon 780M / RDNA3, confirmed via `vulkaninfo`: 17.8
TFLOPS peak FP16, 89.6 GB/s peak bandwidth) and the kernel's actual weight
-packing format (confirmed by reading `linear_q4gsw_coop.glsl` directly: 4
-bit packed weight, 4 FLOPs/byte base intensity). The result is decisive:
a 12-50x margin below this device's machine balance point (~199
FLOPs/byte) -- decode's linear layer is unambiguously memory-bandwidth
-bound, not a close call. Per FR-003, this means the feature's expected
deliverable is the roofline finding and a recommendation of what would
actually help (more aggressive weight quantization, or batching/
speculative decoding to create a real `M>1` opportunity) -- not a new
shader. User Story 2 (shader design) and User Story 3 (benchmarking) are
retained in the task list as the contingent path FR-004 requires if this
finding is somehow overturned, but are not expected to execute.

## Technical Context

**Language/Version**: Python 3.10+ (`uv`-managed `.venv`) for the roofline
calculation and report rendering. C++/GLSL only enter the picture if the
contingent User Story 2/3 path executes (not expected).

**Primary Dependencies**:
- Published device specs for the `rocky-ryzen` MiniPC's AMD Radeon 780M
  iGPU and Ryzen 9 7940HS platform (research.md Decision 1) -- no new
  profiling tooling.
- `backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coop.glsl` and its
  included `.glslh` headers, read (not modified) to confirm the weight
  -packing format driving the kernel's arithmetic-intensity calculation.
- `010`'s established correctness-harness and benchmark methodology,
  reused unchanged if the contingent path executes (research.md Decision
  5) -- no new mechanism designed for that case.

**Storage**: A single flat report file,
`specs/012-decode-wmma-feasibility/results/decode-wmma-feasibility-report.md`.
No raw capture logs are expected (the primary deliverable is a
calculation, not a device capture) unless the contingent path executes.

**Testing**: N/A for the expected (roofline-only) path -- there is no code
to test, only a calculation to verify against its cited sources. If the
contingent path executes, `010`'s correctness-harness discipline applies
(CPU/ATen reference, dtype-appropriate tolerance, SPIR-V inspection)
before any performance number is trusted (constitution Principle I).

**Target Platform**: `rocky-ryzen` MiniPC (AMD Radeon 780M / RDNA3),
matching every prior tier-1 feature -- confirmed via `vulkaninfo` during
planning (`deviceName: AMD Radeon 780M Graphics (RADV PHOENIX)`).

**Project Type**: Primarily an analytical feasibility investigation, not a
build/capture-heavy feature -- unique in this workstream for having its
main deliverable be a citable calculation rather than a device capture.

**Performance Goals**: N/A -- this feature determines whether pursuing a
performance goal for decode is even warranted; it does not carry one of
its own to hit.

**Constraints**:
- No new WMMA decode shader is designed unless the roofline finding
  (Decision 3) is overturned by new information (FR-003/FR-004) -- the
  12-50x margin found means this is not expected.
- No sudo/`dmidecode` access on `rocky-ryzen` to confirm exact installed
  RAM speed -- published platform specs are used instead (research.md
  Decision 1), a conservative choice since slower-than-assumed RAM would
  only strengthen the bandwidth-bound conclusion, never weaken it.
- Decode SDPA's two GEMV kernels remain explicitly out of scope (FR-008).

**Scale/Scope**: One kernel (`linear_q4gsw_coop`, covering both `4w`/
`8da4w` via its existing `DYNAMIC_QUANT_VARIANT` parameter), one device.
The contingent User Story 2/3 path, if it executes, would scale to the
three target models for the benchmark stage only.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against `.specify/memory/constitution.md` (v1.4.0):

- **I. Correctness Before Performance (NON-NEGOTIABLE)**: PASS -- this
  feature's whole point is preventing a performance investment
  (a new shader) before its premise is verified; no performance number is
  produced without a correctness check preceding it (FR-004, contingent
  path only).
- **II. Samsung RDNA3 iGPU Is the Target, Not a Fallback**: PASS with
  scope note -- `rocky-ryzen` MiniPC only, consistent with every prior
  tier-1 feature; both are RDNA3, so this finding is expected to transfer,
  though Samsung/Xclipse validation remains a separate future feature.
- **III. Explicit Eligibility Gating, Safe Fallback Always**: N/A -- this
  feature does not modify any eligibility-gating logic; it only decides
  whether new gated logic is worth designing in the first place.
- **IV. Two-Tier, Statistically Sound Benchmarking**: PASS -- if the
  contingent path executes, it follows `007`/`010`'s tier-1 microbenchmark
  discipline (iteration count and variance on every timing) exactly; the
  expected (roofline-only) path produces no timing numbers at all, so this
  principle's benchmarking requirements don't apply to it.
- **V. Document Every Driver Workaround at the Point of Use**: N/A -- no
  driver workaround is anticipated; this feature doesn't touch dispatch or
  shader code in its expected path.
- **VI. Verify With Tools, Never Assume**: PASS, central to this feature --
  the roofline finding itself is this principle applied at the *design
  -decision* level, not just the dispatch-confirmation level: rather than
  assuming decode's "no implementation exists" gap is fixable by adding a
  WMMA shader, this feature verifies (via published specs and the actual
  kernel's own weight format, both confirmed during planning, not assumed)
  whether that premise even holds before any shader design work begins.

No violations identified. Complexity Tracking is not needed.

*Post-Phase-1 re-check*: Phase 1's data model keeps the Roofline Finding as
a standalone, first-class entity with its own cited sources and margin
figure (data-model.md), and makes the Correctness Case and Microbenchmark
Case entities explicitly contingent on its verdict -- so Principle I stays
structurally enforced (no performance entity can be populated without the
correctness gate ahead of it), not just enforced by convention.

## Project Structure

### Documentation (this feature)

```text
specs/012-decode-wmma-feasibility/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── decode-wmma-feasibility-schema.md
└── tasks.md              # Phase 2 output (/speckit-tasks, not this command)
```

### Source Code (repository root)

No production code changes are expected in this feature's primary path --
the deliverable is an analytical report, not a shader:

```text
specs/012-decode-wmma-feasibility/
└── results/
    └── decode-wmma-feasibility-report.md   # the Roofline Finding, and
                                              # (if reached) the contingent
                                              # Correctness/Microbenchmark
                                              # sections

# Read-only references (not modified in the expected path):
backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coop.glsl        # weight-format source
backends/vulkan/runtime/graph/ops/glsl/linear_int4_weight_tile_load.glslh
backends/vulkan/runtime/graph/ops/glsl/linear_fp_weight_scales_load.glslh

# Only touched if the contingent User Story 2/3 path executes (not expected):
backends/vulkan/runtime/graph/ops/glsl/                               # new WMMA decode shader + .yaml
backends/vulkan/test/op_tests/                                        # new correctness case, 010-style
backends/vulkan/test/custom_ops/                                      # new microbenchmark harness, 007/010-style
```

**Structure Decision**: Lightest-weight structure possible for the
expected path -- a single report file, no build targets, no new shader
code. The contingent path (only if the roofline finding is overturned)
would follow `010`'s exact project structure precedent, not a new one.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations -- this section is intentionally empty.
