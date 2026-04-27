# Agent Plan 4: int8 Cooperative-Matrix Exploration

## Status

Planned. Start this only after Phase 3 has a production-quality fp16
texture-backed linear coopmat path or an explicit decision is made to pause
Phase 3.

## Objective

Explore whether `VK_KHR_cooperative_matrix` int8/uint8 paths on RDNA3 can
produce real ExecuTorch model speedups, and determine whether int8 should move
to a later production integration phase.

This is a research phase. The expected output is a decision backed by kernel
benchmarks, correctness results, and at least one end-to-end model study.

## Required Inputs

Read:

```text
yanwen_docs/agent_plans/0_overview.md
yanwen_docs/agent_plans/1_kernel_sweep_fp16_rdna3.md
yanwen_docs/agent_plans/2_real_llama_e2e_storage_study.md
yanwen_docs/agent_plans/3_production_integration_design.md
yanwen_docs/agent_plans/current_findings_after_kernel_sweep.md
yanwen_docs/agent_plans/current_findings_after_phase2_e2e.md
yanwen_docs/agent_reports/kernel_sweep_fp16_rdna3.md
yanwen_docs/agent_reports/real_llama_e2e_storage_study.md
yanwen_docs/lesson_learned/README.md
```

Also inspect the current Vulkan quantized linear, packing, and export paths
before writing new shaders.

## Non-Goals

Do not block Phase 3 fp16 production integration on int8.

Do not assume a microbenchmark win is sufficient. Prior fp16 work showed that
storage layout and CPU/GPU copy behavior can dominate real model wallclock.

Do not productionize int8 in this phase unless explicitly requested after the
research report is complete.

Do not redesign ExecuTorch quantization broadly. Keep the study scoped to the
minimum quantized linear/matmul path needed to evaluate cooperative matrix
value.

## Initial Hardware Hypothesis

The target RDNA3/RADV device has reported cooperative-matrix configurations
including:

```text
16x16x16 int8/uint8 variants -> int32/uint32
```

Confirm this from `vulkaninfo` or runtime capability queries in this phase.
Treat the exact supported signedness, accumulator type, scope, and subgroup
requirements as measured inputs, not assumptions.

## Research Questions

Answer these before recommending production work:

- Which int8/uint8 cooperative matrix configurations are exposed on the target
  device?
- What operand signedness does ExecuTorch's relevant quantized path need?
- Does the existing Vulkan quantized linear path use layouts that can feed
  cooperative matrix kernels without expensive repacking?
- Should the first kernel target be `linear`, `matmul`, or only weight-only
  linear?
- What accumulator and output path is needed: int32 accumulation, dequantized
  fp16/fp32 output, or quantized output?
- Are scale, zero-point, bias, and activation fusion costs small enough that
  int8 kernel speedups survive end to end?
- Does int8 help real LLaMA-style workloads more than the fp16 Phase 3 path?
- Does storage choice create the same texture-vs-buffer trap observed in fp16?

## Work Plan

### 1. Capability and Existing-Path Audit

Collect and report:

```text
VK_KHR_cooperative_matrix support
all int8/uint8 cooperative matrix shapes
component types
result types
scope
subgroup-size constraints
required shader capabilities/extensions
```

Inspect existing ExecuTorch Vulkan quantized operators and packing code:

```text
backends/vulkan/runtime/graph/ops/
backends/vulkan/runtime/graph/ops/glsl/
backends/vulkan/_passes/
backends/vulkan/vulkan_preprocess.py
```

Record the current shader names, storage assumptions, packed weight layout,
scale/zero-point handling, and fallback behavior.

### 2. Choose the Narrow First Kernel Target

Preferred first target:

```text
int8/uint8 linear with int32 accumulation and dequantized fp16/fp32 output
```

Reason: LLaMA and transformer workloads expose large linear layers where Phase
1/2 already built the benchmarking and ETDump methodology.

Only switch to matmul first if the existing quantized linear path is blocked by
packing or export constraints and matmul has a cleaner path to measurement.

### 3. Microbenchmark Prototype

Create benchmark-only shader variants before touching normal production
dispatch.

Suggested dimensions:

```text
HW tile: whatever int8 cooperative matrix shape the device exposes
M/N/K shape classes:
  BERT-like linears
  LLaMA prefill linears
  LLaMA decode M=1 fallbacks
  M=32 and M=64 batch/decode cases
  square stress sizes
  4096^3-style stress cases
storage:
  buffer input/output + packed weights
  texture-backed activation path only if compatible and worth testing
```

Compare against:

```text
existing Vulkan quantized shader path
fp16 Stephen baseline
fp16 coopmat Phase 3 candidate
CPU reference for correctness
```

Use full CPU reference for small cases and sampled correctness for large cases,
matching the Phase 1 large-shape methodology.

### 4. Quantization Semantics and Correctness

Correctness must cover:

```text
signed int8 and/or uint8, depending on supported operand path
per-tensor scale
per-channel weight scale if used by the export path
zero-point handling
bias
accumulation range
requantize or dequantize output
padding and non-multiple shapes
M=1 fallback
```

Report tolerances separately for:

```text
integer accumulator correctness
dequantized fp16/fp32 output correctness
model-output tolerance
```

### 5. End-to-End Study

If the microbenchmark prototype wins on relevant full-tile shapes, run an E2E
study analogous to Phase 2.

Minimum variants:

```text
current quantized Vulkan baseline
int8 coopmat candidate
fp16 texture coopmat Phase 3 candidate
CPU or non-Vulkan reference where practical
```

Collect:

```text
wallclock
ETDump category breakdown
shader routing
copy/packing overhead
CPU fallback count
peak memory if available
```

The key comparison is not only int8 coopmat vs old int8. It must also compare
against the fp16 coopmat path, because fp16 may already be fast enough that int8
packing/dequantization overhead erases the benefit.

## Deliverables

Write the main report here:

```text
yanwen_docs/agent_reports/int8_coopmat_exploration_rdna3.md
```

Write raw results under:

```text
yanwen_docs/agent_results/int8_coopmat_exploration_rdna3/
```

Write scripts under:

```text
yanwen_docs/agent_results/script/
```

Write lessons under:

```text
yanwen_docs/lesson_learned/phase4_int8_coopmat_exploration/
```

The final report should include:

- cooperative-matrix int8 capability table
- current ExecuTorch quantized Vulkan path summary
- selected kernel target and why
- benchmark shape table
- correctness results
- performance figures
- ETDump breakdown if E2E runs
- comparison against fp16 Phase 3 path
- recommendation: stop, continue research, or open a production Phase 5

## Go / No-Go Criteria

Recommend moving to a production int8 phase only if all are true:

- The hardware exposes a usable int8/uint8 cooperative matrix configuration.
- A kernel prototype beats the existing Vulkan quantized path on relevant
  shapes.
- Correctness is understood for scale, zero-point, bias, and accumulation.
- E2E results show a real model win after packing, dequantization, storage, and
  CPU/GPU copy costs.
- The implementation can reuse or minimally extend existing ExecuTorch
  quantized export/runtime contracts.

If any of these fail, document the blocker and keep int8 as research rather
than production work.

## Risks

- RDNA3 may expose int8 cooperative matrix shapes that do not match the signed
  operand layout needed by the model/export path.
- Dequantization, scale handling, or packing overhead may dominate the shader
  speedup.
- Buffer storage may look fast in microbenchmarks while hurting E2E wallclock.
- Existing quantized graph export may not route enough work to Vulkan to make
  shader optimization useful.
- int8 accuracy requirements may force paths that are slower than fp16 coopmat.
- Large-shape correctness can be too expensive without sampled reference logic.
