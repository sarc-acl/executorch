# Implementation Plan: 8da4w Int8 WMMA Double-Buffer Variant Sweep

**Branch**: `023-8da4w-int8-dbuf-sweep` | **Date**: 2026-07-07 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/023-8da4w-int8-dbuf-sweep/spec.md`

## Summary

Port each of the four `shmem_double_buf{,2,3,4}.comp` loop structures onto the `8da4w`
int8 coopmat linear op (`linear_dq8ca_q4gsw_coopmat`, currently shipping dbuf4), following
the exact env-var-gated, separately-registered-shader-per-variant pattern this workstream
already built and proved for the **fp16** `4w` shader (`linear_q4gsw_coopmat_dbuf{1..4}` +
`ET_VK_Q4GSW_COOPMAT_VARIANT` in `QuantizedLinear.cpp`, uncommitted in the sibling
`.tmp-origcm` worktree) — reused here per constitution Development Workflow's explicit
mandate to check that tooling before building anything new. Each variant is verified to
compile, dispatch the int8 coopmat kernel, and pass the existing `dq8ca_q4gsw` correctness
check before being timed (one process per variant, isolating any Xclipse PAL
pipeline-creation crash); each is then measured across 6 representative shapes (`wq` +
`w1_gate` x {1B, 3B, 8B}, per spec Clarifications) with pinned, verified clocks on M5 EVT1.
A report states the fastest variant per shape and overall, confirms or refutes the
dbuf3-is-faster-for-int8 hypothesis, and compares the winner to the shipped dbuf4 baseline.
Tier-1 (microbenchmark) only — no e2e validation is required (per spec Clarifications).

## Technical Context

**Language/Version**: GLSL (`#version 450 core`, `GL_KHR_cooperative_matrix`) for the
shader variants; C++17 for the dispatch/eligibility code (`QuantizedLinear.cpp`) and the
benchmark harness (`backends/vulkan/test/custom_ops`)

**Primary Dependencies**: Vulkan 1.x + `VK_KHR_cooperative_matrix`, ExecuTorch's
`vkcompute`/Vulkan backend, the existing `BenchmarkResult` microbench harness, the existing
`dq8ca_q4gsw` correctness test suite (`test_*_linear` / `op_tests`)

**Storage**: N/A — no persisted data beyond markdown result reports and raw log files under
`specs/023-8da4w-int8-dbuf-sweep/results/`

**Testing**: on-device C++ benchmark binary (adb-pushed, run on M5 EVT1) for timing; the
existing correctness test suite for per-variant validation; `spirv-dis`/`spirv-cross` (or
equivalent) for SPIR-V inspection per constitution Principle VI

**Target Platform**: Samsung M5 EVT1 (Exynos 2500 / Xclipse 970), Android, Vulkan backend
(this workstream's sole active target, constitution Principle II)

**Project Type**: single project — an experimental extension of the existing ExecuTorch
Vulkan backend, developed in a new git worktree dedicated to this feature (see Structure
Decision)

**Performance Goals**: for each of the 6 representative shapes, produce a 3-run mean + CoV
execution time for all four dbuf variants of `linear_dq8ca_q4gsw_coopmat` on M5 EVT1, and
identify the fastest

**Constraints**: clocks pinned and verified bound (Principle VII); on-device driver
identity re-verified before measuring (Principle VIII); default production dispatch
behavior for `8da4w` MUST be unchanged (the dbuf-variant switch is opt-in via a new env
var, never on by default); a variant that fails to build/dispatch/pass correctness MUST
still be reported, never silently dropped (spec FR-004); one process per variant, so a
pipeline-creation crash on one variant cannot corrupt another's results

**Scale/Scope**: 4 variants x 6 shapes x 3 runs = 72 timed runs, plus one correctness pass
per variant (4 total) and one SPIR-V inspection per variant (4 total)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. Correctness Before Performance** — PASS. Every variant's timing is gated on it
  first passing the existing `dq8ca_q4gsw` correctness check (User Story 1 / FR-002-003);
  no timing is trusted from an unverified variant.
- **II. Samsung M5 EVT1 Is the Only Active Target** — PASS. All measurements run on M5
  EVT1 exclusively.
- **III. Explicit Eligibility Gating, Safe Fallback Always** — PASS. The new dbuf-variant
  switch is an explicit, opt-in env var (`ET_VK_DQ8CA_COOPMAT_VARIANT`, see research.md
  Decision 3); when unset, dispatch is unchanged from today's shipped `dbuf4` behavior.
- **IV. Two-Tier, Statistically Sound Benchmarking** — PARTIAL BY DESIGN, justified. This
  feature is Tier-1 (microbenchmark) only, per the spec's own Clarifications session — it
  does not claim a Tier-2 (e2e tok/s) result. This matches Principle IV's own framing of
  Tier-1 as "gates correctness... and explains a Tier-2 result after the fact," i.e. this
  sweep is explicitly the Tier-1 groundwork for a *future*, separate Tier-2 adoption
  decision, not a substitute for one. No Complexity Tracking entry needed: this is a valid,
  spec-documented scope, not an unjustified violation.
- **V. Document Every Driver Workaround at the Point of Use** — PASS (by requirement). Any
  Xclipse PAL restructuring needed to make a dbuf1/2/3 port compile against the int8
  shader's nested-loop/ping-pong structure (research.md Decision 4) MUST carry an inline
  comment per this principle, matching the existing shader's own workaround comments.
- **VI. Verify With Tools, Never Assume** — PASS (by requirement). Dispatch is confirmed
  via the bench harness's kernel-name capture (research.md Decision 6); each variant's
  compiled SPIR-V is disassembled once to confirm genuine int8 coopmat instructions are
  present.
- **VII. Clock Discipline** — PASS (by requirement). Pinned by default, pin verified bound
  via GFLOP/s cross-check, per spec FR-005.
- **VIII. Verify the Driver Before Every Coopmat Measurement** — PASS (by requirement).
  Driver identity re-verified before measuring, per spec FR-006.
- **IX. Never Disclose Samsung-Internal Specifics Upstream** — N/A. This feature's output
  stays on `sarc-acl/executorch`; no upstream PR is implied or prepared by this work.
- **X. Consult `instruction-for-ai` Before Acting** — PASS (by requirement). Build/export
  is N/A (no `.pte`, no model export); device access, clock pinning, and driver
  verification steps consult `.shared-context/instruction-for-ai/` per usual.

No unjustified gate violations. Proceeding to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/023-8da4w-int8-dbuf-sweep/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── results/             # Phase 3+ output (/speckit-implement): raw logs + report markdown
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

This feature's spec/plan/tasks documents are authored in the current
`quant-perf-optimization` worktree (where `/speckit-specify` was invoked) and must be
committed to the `quant-perf-optimization` branch before the new worktree below is created,
so that worktree's checkout includes them (research.md Decision 7).

### Source Code (repository root)

No new top-level project is created; this feature adds files to the existing ExecuTorch
Vulkan backend, inside a new git worktree dedicated to this experiment (paths below are
relative to that worktree's `executorch/` checkout, identical in layout to this one):

```text
backends/vulkan/runtime/graph/ops/glsl/
├── linear_dq8ca_qw_coopmat.glsl                 # production shader -- NOT modified
├── linear_dq8ca_qw_coopmat.yaml                 # production dispatch -- NOT modified
├── linear_dq8ca_q4gsw_coopmat_dbuf1.glsl        # new: dbuf1 ("prefetch-first") port
├── linear_dq8ca_q4gsw_coopmat_dbuf1.yaml
├── linear_dq8ca_q4gsw_coopmat_dbuf2.glsl        # new: dbuf2 ("store-first") port
├── linear_dq8ca_q4gsw_coopmat_dbuf2.yaml
├── linear_dq8ca_q4gsw_coopmat_dbuf3.glsl        # new: dbuf3 (peeled, no-conditional) port
├── linear_dq8ca_q4gsw_coopmat_dbuf3.yaml
├── linear_dq8ca_q4gsw_coopmat_dbuf4.glsl        # new: dbuf4 -- reference copy of the
├── linear_dq8ca_q4gsw_coopmat_dbuf4.yaml        #   already-shipped loop structure, built
│                                                 #   under this harness for an apples-to-
│                                                 #   apples in-sweep baseline measurement
└── ...

backends/vulkan/runtime/graph/ops/impl/
└── QuantizedLinear.cpp     # add one opt-in branch (ET_VK_DQ8CA_COOPMAT_VARIANT env var)
                             # to the existing kernel_name selection for dq8ca_q4gsw_coopmat;
                             # default (env var unset) behavior unchanged

backends/vulkan/test/custom_ops/
└── test_coopmat_linear_bench.cpp   # extend (or reuse, if already shape-generic) to time
                                     # dq8ca_q4gsw_coopmat shapes, one variant per process
```

**Structure Decision**: single project, extending the existing ExecuTorch Vulkan backend
in place. Per the user's explicit instruction, code changes and on-device measurement work
happen in a **new git worktree**, branched from the tip of `quant-perf-optimization`
(after this feature's spec/plan/tasks are committed there) so the new worktree inherits all
prior committed specs/history but none of the current worktree's uncommitted changes from
other in-flight specs. The new worktree is bootstrapped per constitution "Environment &
Build Bootstrap" (`uv venv .venv --seed`, `./install_executorch.sh --minimal`) before any
build is attempted.

## Complexity Tracking

*No entries — no unjustified Constitution Check violations (see above).*
