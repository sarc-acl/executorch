# Agent Plan 3: Production Integration Design for RDNA3 fp16 WMMA

## Objective

Design the production path for substituting Stephen Jia's fp16 matmul/linear
Vulkan shaders with `VK_KHR_cooperative_matrix` shaders on RDNA3 iGPUs, while
preserving correctness and fallback behavior.

This plan should be executed after the kernel sweep and real LLaMA E2E study
produce enough data.

## Required Inputs

Agents may read anything under `yanwen_docs/`, including background notes,
previous results, future diffs, and reports. Read:

```text
yanwen_docs/background/0_shader_baseline.md
yanwen_docs/background/1_previous_story.md
yanwen_docs/agent_plans/0_overview.md
yanwen_docs/agent_plans/current_findings_after_kernel_sweep.md
yanwen_docs/agent_plans/current_findings_after_phase2_e2e.md
yanwen_docs/agent_reports/kernel_sweep_fp16_rdna3.md
yanwen_docs/agent_reports/real_llama_e2e_storage_study.md
```

Use the tested Vulkan build commands from
`yanwen_docs/agent_plans/0_overview.md` when validating buildability. If the
design requires extra targets or flags, document them separately.

Phase 1 and Phase 2 are complete. If the real LLaMA E2E report is missing in a
future checkout, write a design with open questions rather than guessing.

## Target

Initial target:

```text
AMD Radeon 780M / RADV Phoenix-class RDNA3 iGPU
fp16 linear/matmul first
fp32 secondary
int8 deferred to Phase 4 exploration
q4 future work
```

The production integration must preserve fallback to Stephen's existing shaders.

The current fp16 design must not force whole-graph buffer storage. Prior real
LLaMA results showed buffer storage can make the WMMA kernels faster while
making end-to-end wallclock much worse on AMD 780M / RADV Phoenix.

Phase 2 showed that texture3D linear coopmat does transfer to real LLaMA E2E at
seq=256:

```text
real_tex_stephen: 442.63 ms
real_tex_coopmat: 250.43 ms  (1.77x)
real_buf_coopmat: 205.29 ms  (2.16x vs texture Stephen at seq=256)
```

The buffer result should remain a small-sequence data point, not a production
default. The requested seq=2048 run OOM-killed on the 28 GB host, while the
previous-story seq=2048 evidence showed whole-graph buffer storage regressing
badly.

Phase 1 kernel sweep is concluded. The swept software parameters were:

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

## Candidate Architecture

Recommended starting design: **dual storage WMMA path**.

Add separate shader variants for:

```text
linear_coopmat_buffer.glsl
linear_coopmat_texture.glsl
matmul_coopmat_buffer.glsl
matmul_coopmat_texture.glsl
```

The picker selects based on:

```text
device supports VK_KHR_cooperative_matrix
device exposes required 16x16x16 fp16 cooperative matrix config
dtype is half
shape passes dispatch gate
storage type matches available shader
environment/compile option does not disable coopmat
```

The buffer variants already have kernel-side evidence. Linear texture3D
input/output + buffer-weight coopmat has both kernel-side evidence and Phase 2
real-LLaMA E2E evidence. Texture matmul and texture-packed weight variants are
not implemented.

Fallback:

```text
Stephen linear_vec/matmul_vec/scalar path
```

## Shader Selection Gate

Implement a conservative gate first.

Inputs:

```text
M
K
N
dtype
input storage
output storage
weight storage
has bias
device cooperative matrix properties
subgroup size support
```

Current conservative kernel-side gate from the fp16 RDNA3 sweep:

```text
use WMMA only when:
  dtype == half
  device supports the required fp16 cooperative matrix config
  op is 2D linear or non-constant 2D matmul
  storage path is implemented and E2E-known-good
  M % 64 == 0
  N % 64 == 0
  K % 32 == 0
  fp32 cooperative accumulator
  subgroup size 64 unless a tuning table selects otherwise
  64x64x32 software macro tile unless a tuning table selects otherwise
  K-step 32
```

Phase 3 starting production candidate:

```text
linear texture3D coopmat:
  dtype == half
  device exposes required fp16 KHR cooperative matrix config
  input storage == texture3D
  output storage == texture3D
  M % 64 == 0
  N % 64 == 0
  K % 32 == 0
  fp32 accumulator
  K-step 32
  fallback to Stephen for M=1 / LM head
```

Likely initial fallback:

```text
M == 1 -> Stephen shader
M == 32 without an enabled 32x64-style tuned variant -> Stephen shader
unsupported storage -> Stephen shader
missing cooperative matrix config -> Stephen shader
correctness-risk padding case -> Stephen shader
```

The final E2E threshold must come from the real LLaMA report, not from
microbenchmarks alone.

## Device Capability Query

Required capability checks:

```text
VK_KHR_cooperative_matrix extension present
cooperativeMatrix == true
SHADER_STAGE_COMPUTE_BIT supported
16x16x16 fp16 x fp16 -> fp16 available
16x16x16 fp16 x fp16 -> fp32 available if used
subgroup size 32/64 support if shader depends on it
```

Add debug logging that can explain selection:

```text
selected linear_coopmat_texture_half because ...
fallback to linear_vec_texture3d_texture2d_half because ...
```

## Storage Strategy

The previous iGPU story showed that whole-graph buffer storage can regress real
LLaMA badly. Therefore, production integration should avoid making the whole
graph buffer-backed just to use WMMA.

Preferred default direction:

```text
texture3d activation input/output
buffer packed weights
shared-memory staging for coopMatLoad where required
```

Keep buffer WMMA only for cases where the graph is already buffer-backed for
reasons other than coopmat. Phase 2 showed buffer coopmat wins at seq=256, but
the seq=2048 risk remains unresolved on this host.

Do not force `CONTIGUOUS_BUFFER` globally for all linear/matmul unless the E2E
study proves it is safe.

When designing the texture path, explicitly verify that graph storage tagging
does not silently propagate buffer storage from CPU-fallback boundaries into
linear/matmul. Previous work found that `CONTIGUOUS_ANY` inputs and
`sync_primary_io_repr=True` can keep linears buffer-backed even when texture
storage is desired.

## Files Likely To Change

Shader files:

```text
backends/vulkan/runtime/graph/ops/glsl/linear_coopmat_*.glsl
backends/vulkan/runtime/graph/ops/glsl/linear_coopmat_*.yaml
backends/vulkan/runtime/graph/ops/glsl/matmul_coopmat_*.glsl
backends/vulkan/runtime/graph/ops/glsl/matmul_coopmat_*.yaml
```

C++ runtime:

```text
backends/vulkan/runtime/graph/ops/impl/Linear.cpp
backends/vulkan/runtime/graph/ops/impl/Matmul.cpp
backends/vulkan/runtime/graph/ops/impl/Common.cpp
backends/vulkan/runtime/graph/ops/impl/Common.h
backends/vulkan/runtime/vk_api/Adapter.cpp
backends/vulkan/runtime/vk_api/Adapter.h
backends/vulkan/runtime/vk_api/Device.cpp
backends/vulkan/runtime/vk_api/Device.h
```

Possibly:

```text
backends/vulkan/op_registry.py
backends/vulkan/_passes/tag_memory_meta_pass.py
backends/vulkan/vulkan_preprocess.py
```

The Phase 2 real LLaMA texture path did not require partitioner changes: the
existing texture3D `.pte` routed 28 prefill linears to
`linear_coopmat_texture3d_buffer` under the env hook. Only touch
partitioner/storage rules if needed for future texture matmul or production
feature-flag integration.

Build-system note: after adding new GLSL/YAML variants, rerun the top-level
CMake configure before rebuilding. Otherwise the generated Vulkan shader
registry may not contain the new shader names.

## Feature Flags

Add or preserve controls:

```text
disable coopmat entirely
force coopmat for testing
log shader selection
possibly restrict to RDNA3/RADV during experimental phase
```

Example environment names can be decided later, but the behavior should exist.

The benchmark-only hooks from the sweep are not production designs:

```text
VK_COOPMAT_ACCUM_FP16
VK_COOPMAT_REQUIRED_SUBGROUP_SIZE
VK_COOPMAT_MACRO_TILE
```

If any of these become production behavior, convert them into normal capability
checks, picker inputs, or tuning-table entries.

## Tests

Unit/kernel tests:

```text
fp16 linear correctness for LLaMA shapes against Stephen baseline or CPU ref
fp16 matmul correctness for attention shapes against Stephen baseline or CPU ref
bias and no-bias
texture3d and buffer storage paths
padding/non-multiple dimensions
fallback when coopmat unsupported
fallback when dtype != half
fallback for M=1 decode
fallback for large shapes without verified correctness
```

Benchmark tests:

```text
microbench against Stephen baseline
reduced real LLaMA wallclock
ETDump shader selection confirmation
```

Regression tests:

```text
device without VK_KHR_cooperative_matrix still uses Stephen shaders
fp32 path still works
buffer fallback still works
existing quantized linear shaders unchanged
```

## Deliverables

Final design report:

```text
yanwen_docs/agent_reports/production_integration_design_rdna3_fp16_wmma.md
```

Lessons learned:

```text
yanwen_docs/lesson_learned/phase3_production_integration/
```

If design work uncovers non-obvious integration risks, fallback requirements,
build-system problems, or storage/partitioner constraints, document them there
for Yanwen.

The report should include:

- final selected architecture
- shader selection decision tree
- exact files to modify
- capability query design
- storage strategy
- fallback behavior
- test matrix
- implementation risks
- estimated implementation stages

Optional implementation patch:

```text
backends/vulkan/... changes
```

Do not land broad refactors as part of this design session unless explicitly
requested.

## Risks To Call Out

- Kernel speedup may not transfer to E2E if storage propagation is wrong.
- Buffer storage may create large host-side overhead on RDNA3 iGPU.
- Texture-input WMMA may require careful shared-memory staging.
- fp16 accumulator was not a consistent speedup and may be less accurate than
  fp32 accumulator.
- Decode `M=1` may regress and should likely fall back.
- Large benchmark rows that skipped CPU reference are not final correctness
  evidence.
- Explicit shader variant names in YAML do not automatically take dtype suffixes.
- Subgroup-size selection currently exists only through a benchmark hook.
- Device cooperative matrix properties are not portable across vendors.
- Quantized LLaMA remains future work.
