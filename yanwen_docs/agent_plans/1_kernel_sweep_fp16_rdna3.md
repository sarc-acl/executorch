# Agent Plan 1: RDNA3 fp16 WMMA Kernel Sweep

## Status

Complete. Phase 1 concluded with:

```text
yanwen_docs/agent_reports/kernel_sweep_fp16_rdna3.md
yanwen_docs/agent_plans/current_findings_after_kernel_sweep.md
yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/analysis/kernel_sweep_web_report.html
```

Do not restart a broad kernel sweep unless a new device, dtype, or production
blocker requires it. Follow-up work should use the completed parameter table:

```text
HW WMMA tile: 16x16x16 fixed by hardware
input type: fp16 x fp16
accumulator: fp16, fp32 -> use fp32
subgroup size: 32, 64 -> default 64
macro tile: 16x16, 16x32, 32x16, 32x32, 16x64, 64x16, 32x64, 64x32, default 64x64 -> default 64x64
K-step: 16, 32, 64 -> use 32
storage: buffer input/output + buffer weights; texture3D input/output + buffer weights for linear only
ops: linear, matmul
correctness: full CPU reference for small shapes; sampled CPU reference for large shapes
```

## Objective

Build a comprehensive kernel-side report for fp16 cooperative-matrix
matmul/linear on AMD Radeon 780M / RADV Phoenix. The goal is to identify when a
KHR WMMA shader beats Stephen Jia's existing fp16 GLSL shaders, independent of
full-model graph overhead.

This session should answer:

```text
For LLaMA-shaped fp16 matrix multiplications on RDNA3 iGPU, what WMMA shader
configuration is fastest, and when should we fall back to Stephen's shader?
```

## Inputs

Agents may read anything under `yanwen_docs/`, including background notes,
previous results, future diffs, and reports. Read first:

```text
yanwen_docs/background/0_shader_baseline.md
yanwen_docs/background/1_previous_story.md
yanwen_docs/agent_plans/0_overview.md
```

Use the tested Vulkan build commands from
`yanwen_docs/agent_plans/0_overview.md` unless the task explicitly requires a
different build.

If provided, inspect the previous coopmat branch diff:

```text
yanwen_docs/background/coopmat_previous_work.diff
```

Do not assume those files exist yet. If absent, create a minimal plan/report
stub that says what is blocked.

## Baselines

Compare against Stephen's fp16 shaders:

```text
backends/vulkan/runtime/graph/ops/glsl/linear_vec.glsl
backends/vulkan/runtime/graph/ops/glsl/linear_scalar.glsl
backends/vulkan/runtime/graph/ops/glsl/matmul_vec.glsl
backends/vulkan/runtime/graph/ops/glsl/matmul_scalar.glsl
backends/vulkan/runtime/graph/ops/glsl/pack_fp_linear_weight.glsl
```

The baseline is fp32/fp16 only. Do not include int8/q4 in this phase.

## Scope

Implement or port benchmarkable fp16 WMMA variants for:

```text
linear
matmul
```

Preferred shader candidates:

```text
linear_coopmat_buffer.glsl
matmul_coopmat_buffer.glsl
linear_coopmat_texture.glsl
matmul_coopmat_texture.glsl
```

If time is limited, prioritize `linear` before `matmul`, because LLaMA wallclock
is dominated by linear projections and MLP layers.

## Parameters To Sweep

### Cooperative Matrix Type

The RDNA3 iGPU exposes `16x16x16` cooperative matrix shapes. Sweep:

```text
fp16 x fp16 -> fp16
fp16 x fp16 -> fp32
```

Record:

- correctness error
- runtime
- GFLOP/s
- whether fp32 accumulation is worth the cost

### Subgroup Size

RDNA3/RADV reports:

```text
minSubgroupSize = 32
maxSubgroupSize = 64
subgroupSizeControl = true
default subgroupSize = 64
```

Sweep:

```text
subgroup size 64
subgroup size 32
```

If subgroup size cannot be controlled through the current shader/build path,
document the blocker and only measure the default.

### Macro Tile Shape

Sweep workgroup/output tile shapes built from 16x16x16 cooperative matrix
operations:

```text
16x16
16x32
32x16
32x32
64x16
16x64
```

Optional if time permits:

```text
32x64
64x32
```

Measure register pressure and shared memory impact where possible.

### K Blocking / Loop Unroll

Sweep:

```text
K step = 16
K step = 32
K step = 64
```

This may map to unrolling multiple `coopMatMulAdd` operations per loop.
Document whether increased unroll helps or hurts occupancy/register pressure.

### Storage Mode

Measure these separately:

```text
buffer input/output, buffer packed weights
texture3d input/output, buffer packed weights
texture3d input/output, texture2d or buffer packed weights if supported
```

Important: previous experiments showed whole-graph buffer storage is dangerous
on RDNA3 iGPU. Kernel-only benchmarks should still measure buffer storage, but
the report must not conclude that buffer is safe for E2E.

For texture-input WMMA, the likely design is:

```text
texelFetch texture3d activation -> shared memory tile -> coopMatLoad
buffer packed weight -> shared memory tile or direct coopMatLoad
coopMatMulAdd
store to texture3d output
```

### LLaMA Shape Classes

Measure at least:

```text
M=1,    K=4096, N=4096      // decode Q/K/V-ish
M=1,    K=4096, N=11008     // decode FFN up/gate
M=1,    K=11008, N=4096     // decode FFN down
M=32,   K=4096, N=4096
M=64,   K=4096, N=4096
M=128,  K=4096, N=4096
M=256,  K=4096, N=4096
M=512,  K=4096, N=4096
M=2048, K=4096, N=4096      // prefill projection
M=2048, K=4096, N=11008     // prefill FFN up/gate
M=2048, K=11008, N=4096     // prefill FFN down
```

Also include at least two non-multiple/padding cases if the shader claims to
handle them.

## Correctness

For every measured shape/config:

- compare against Stephen baseline or CPU reference
- report max absolute error
- report max relative error
- define pass/fail threshold for fp16

Correctness failures should be kept in the table. Do not silently drop them.

## Deliverables

Raw results:

```text
yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/
```

Lessons learned:

```text
yanwen_docs/lesson_learned/phase1_kernel_sweep/
```

If a shader variant, subgroup setting, build path, benchmark harness, or device
capability behaves unexpectedly, write a short lesson file there so Yanwen can
review it.

Final report:

```text
yanwen_docs/agent_reports/kernel_sweep_fp16_rdna3.md
```

The report must include:

- device info
- exact commit/local diff summary
- build commands
- benchmark commands
- correctness table
- per-shape fastest config
- speedup over Stephen baseline
- GFLOP/s
- subgroup 32 vs 64 conclusion
- fp16 accumulation vs fp32 accumulation conclusion
- buffer vs texture conclusion
- recommended dispatch gate

## Expected Output Table Columns

Use CSV or Markdown table with at least:

```text
shape_id
M
K
N
dtype
accumulator_type
storage_in_out
storage_weight
subgroup_size
macro_tile_m
macro_tile_n
k_step
baseline_us
coopmat_us
speedup
gflops
max_abs_error
max_rel_error
pass_fail
notes
```

## Decision Criteria

Recommend WMMA only for shape/configs that:

- pass correctness
- beat Stephen by a meaningful margin
- do not require whole-graph buffer storage for future E2E use
- are compatible with RDNA3 cooperative matrix properties

If decode `M=1` regresses or only breaks even, explicitly recommend a decode
fallback to Stephen or a separate GEMV path.
