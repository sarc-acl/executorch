# Implementation Plan: 8da4w Coopmat Tile/Subgroup Parameter Sweep

**Branch**: `008-8da4w-parameter-sweep` | **Date**: 2026-07-04 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/008-8da4w-parameter-sweep/spec.md`

## Summary

Sweep a curated set of 11 candidate workgroup tile shape
(`WG_TILE_M`/`WG_TILE_N`/`WG_TILE_K`) and subgroup size combinations
(plus 1 deliberate negative-test configuration, `/speckit-analyze` finding
G1) for the `8da4w` (`linear_dq8ca_q4gsw`) coopmat shader on `rocky-ryzen`,
to find whether any configuration closes or reverses the 10-22% regression
`007` found for the currently-shipped (Xclipse-tuned) configuration. Every
variant is built as a
**fully separate, test-owned shader + harness** under
`backends/vulkan/test/custom_ops/` -- zero production files are touched,
satisfying FR-008 unambiguously (this repo already has precedent for
test-owned shader ports, e.g. `coopmat_mm_ref.glsl`).

## Technical Context

**Language/Version**: GLSL (new test-owned shader template, parameterized
like the production one) + C++ (new test-owned dispatch/benchmark harness)
+ Python 3.10+ (`uv`-managed `.venv`) for the comparison/report script,
matching every prior feature in this workstream.

**Primary Dependencies**:
- A new, test-owned shader template
  `backends/vulkan/test/custom_ops/glsl/dq8ca_q4gsw_coopmat_sweep.{glsl,yaml}`
  -- a copy of the production `linear_dq8ca_qw_coopmat.glsl`'s
  double-buffered int8-coopmat logic (already parameterized by
  `WG_TILE_M`/`WG_TILE_N`/`WG_TILE_K`/`SUBGROUP_SIZE`/`SG_GRID_X`/`SG_GRID_Y`),
  with its own `shader_variants` list covering the swept combinations.
  Precedented: this exact directory already carries
  `q4gsw_linear_gemv__w_4x8.glsl`, a test-owned shader distinct from
  production; `coopmat_mm_ref.glsl` is an existing test-owned *port* of a
  reference implementation -- the same pattern this feature follows.
- Already-exposed production helpers, reused (not duplicated): `QuantizedLinear.h`'s
  `prepack_quantized_linear_weight()`/`quantized_linear_local_wg_size()`
  and `QuantizeDequantize.h`'s
  `add_quantize_and_pack_4h4w_with_group_sums_node()` -- the dynamic
  activation quantization/packing step every `8da4w` case needs regardless
  of which GEMM kernel runs.
- A new test-only `DynamicDispatchNode` builder (in the new harness .cpp,
  not in `QuantizedLinear.cpp`) that mirrors `add_linear_dqa_qw_node`'s
  structure but takes a **fixed** kernel name per swept variant instead of
  calling `pick_linear_dqa_qw_shader` -- the only structural difference
  from production dispatch, and it lives entirely in test code.
- `test_coopmat_linear_bench.cpp`'s existing `bench_reference()` pattern,
  reused for correctness verification of each new variant.

**Storage**: Flat files -- new raw capture log, SPIR-V dumps for each
swept variant, and the report under this feature's own `results/`.

**Testing**: No separate automated suite -- the correctness check (CPU
reference comparison, reused pattern) and kernel-dispatch verification
(FR-005) together ARE the verification, matching `007`.

**Target Platform**: `rocky-ryzen` MiniPC (AMD Radeon 780M, RADV/Mesa
driver) only -- explicitly a device-specific tuning study (spec
Assumptions).

**Project Type**: New test-only shader + harness code under
`backends/vulkan/test/custom_ops/`. **Zero changes to any file under
`backends/vulkan/runtime/graph/ops/` or `backends/vulkan/op_registry.py`**
-- this is the one hard constraint (FR-008) shaping every technical choice
below.

**Performance Goals**: N/A -- this feature measures and compares
performance; it does not carry a target of its own.

**Constraints**:
- FR-008 (no production changes) is satisfied structurally, not by
  discipline alone: the new shader lives under a new filename stem
  (`dq8ca_q4gsw_coopmat_sweep`) in the test-only shader directory
  (`backends/vulkan/test/custom_ops/glsl/`), so ExecuTorch's shader
  codegen (`gen_vulkan_spv.py`, keyed by filename stem, errors if the same
  `template_name` is declared twice) cannot conflict with or overwrite the
  production template regardless of what this feature does.
- Correctness confidence per variant mirrors `007`'s Clarification Q1 bar:
  kernel-dispatch check (FR-005) + SPIR-V cooperative-matrix-instruction
  presence + a CPU-reference correctness comparison for that specific
  variant (extending `007`'s "cite existing test coverage" approach --
  here, each *new* variant needs its own correctness check since none of
  them exist in any prior correctness test).
- Sweep-phase measurement uses a reduced, representative shape set --
  one square (`wq`) and one rectangular (`w1_gate`) shape per model, 6
  total (not the full 3-model x 7-op `8da4w` catalog) -- to bound device
  time across the candidate variants; the winning configuration(s) are then
  validated against the full `007` shape catalog before being recommended
  (research.md Decision 3, revised during `/speckit-analyze` remediation
  to add the rectangular shape).
- One additional variant (config 12) is a deliberate negative test
  (`WG_TILE_K=64`, mathematically incompatible with `group_size=32`),
  measured at one shape only, to prove the correctness check actually
  catches a broken kernel rather than assuming it would (research.md
  Decision 4, `/speckit-analyze` finding G1).
- **Implementation-time revision**: of the original 11 candidates, the 6
  subgroup-32 variants (1, 3, 5, 7, 9, 11) were dropped after config 1's
  T010 run showed a real, reproducible correctness bug (wrong output at
  the second M-subgroup's tile boundary) -- a user decision, not an
  assumption, per research.md Decision 4. **5 candidates remain active**
  (2, 4, 6, 8, 10), all subgroup 64.
- No concurrent GPU load during any capture (established workstream
  discipline).

**Scale/Scope**: 5 active candidate configurations x 6 representative
shapes + 1 negative-test configuration x 1 shape (31 total sweep-phase
measurements) for US2, then the winning candidate(s) validated against
all 3 models' `8da4w` shapes (US3) before the final
recommendation.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against `.specify/memory/constitution.md` (v1.4.0):

- **I. Correctness Before Performance (NON-NEGOTIABLE)**: PASS. Every
  swept variant's performance number is gated on a CPU-reference
  correctness check specific to that variant (not inherited from an
  unrelated existing test, since none of these variants exist yet) --
  FR-003.
- **II. Samsung RDNA3 iGPU Is the Target, Not a Fallback**: PASS with
  scope note -- this study is explicitly `rocky-ryzen`-only device tuning
  (spec Assumptions); it does not claim anything about Xclipse and does
  not change what ships to it (FR-008).
- **III. Explicit Eligibility Gating, Safe Fallback Always**: N/A/PASS --
  this study's harness bypasses eligibility gating by construction (each
  variant is invoked directly by a fixed test-only kernel name); it does
  not add, remove, or alter any production eligibility gate.
- **IV. Two-Tier, Statistically Sound Benchmarking**: PASS, tier-1
  (shader microbenchmark) only -- every timing carries iteration count and
  stdev (FR-002).
- **V. Document Every Driver Workaround at the Point of Use**: Applies if
  any swept variant crashes at pipeline creation on this device (the same
  failure class documented for Xclipse in commit `49a51b1776`) -- any such
  crash gets the same documentation discipline, naming this device/driver.
- **VI. Verify With Tools, Never Assume**: PASS, central to this feature's
  design -- every variant's coopmat dispatch is confirmed via kernel name
  (FR-005) and SPIR-V instruction presence, exactly the discipline that
  caught `007`'s dispatch-wiring gap in the first place.

No violations. Complexity Tracking is not needed.

## Project Structure

### Documentation (this feature)

```text
specs/008-8da4w-parameter-sweep/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── sweep-report-schema.md
└── tasks.md              # Phase 2 output (/speckit-tasks, not this command)
```

### Source Code (repository root)

New, test-only files only -- no production files touched:

```text
backends/vulkan/test/custom_ops/
├── glsl/
│   ├── dq8ca_q4gsw_coopmat_sweep.glsl   # new: copy of the production double-buffered int8-coopmat template
│   └── dq8ca_q4gsw_coopmat_sweep.yaml   # new: shader_variants list, one per swept (tile, subgroup) combination
├── test_dq8ca_tile_sweep.cpp            # new: builds each variant's DynamicDispatchNode with a fixed kernel name, runs correctness + timed benchmark
└── CMakeLists.txt                        # new add_operator_prototype(test_dq8ca_tile_sweep) entry

specs/008-8da4w-parameter-sweep/
├── scripts/
│   └── compare_sweep.py   # new: loads sweep results + 007's shipped-config and 004's tiled-baseline numbers, ranks variants, renders the report
└── results/
    ├── raw/                # sweep capture log
    ├── spirv/              # spirv-dis output per swept kernel
    └── sweep-report.md
```

**Structure Decision**: Same lightweight structure as `007` for the
documentation/analysis side (one Python script, one report). The
substantive difference from every prior feature in this workstream: this
one requires new *test-only* GLSL + C++ source, because sweeping tile
parameters means building shader variants that don't exist yet -- but that
new code is entirely contained under `backends/vulkan/test/custom_ops/`,
never under `backends/vulkan/runtime/` or `op_registry.py`.
