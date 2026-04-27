# Subgroup-size control runtime hook

## What was attempted

Looked for an existing ExecuTorch Vulkan runtime hook to request subgroup size
32 versus 64 for a compute shader pipeline, then added a narrow benchmark hook.

## What happened

Searches under `backends/vulkan/runtime` and `backends/vulkan/tools` did not
find a use of `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`,
`requiredSubgroupSize`, or `subgroupSizeControl` in runtime pipeline creation.
The workaround was to add `VK_COOPMAT_REQUIRED_SUBGROUP_SIZE`, which applies a
required subgroup size only to shader names containing `coopmat`.

## Why it matters

The RDNA3/RADV device reports subgroup size control support. A real subgroup 32
vs 64 sweep needs runtime pipeline support for required subgroup size selection,
not just a GLSL edit.

## Commands

```bash
VK_COOPMAT_REQUIRED_SUBGROUP_SIZE=64 cmake-out-vk/backends/vulkan/test/custom_ops/matmul_coopmat_bench > yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/matmul_coopmat_bench_subgroup64.log 2>&1
VK_COOPMAT_REQUIRED_SUBGROUP_SIZE=32 cmake-out-vk/backends/vulkan/test/custom_ops/matmul_coopmat_bench > yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/matmul_coopmat_bench_subgroup32.log 2>&1
VK_COOPMAT_REQUIRED_SUBGROUP_SIZE=64 cmake-out-vk/backends/vulkan/test/custom_ops/linear_coopmat_bench > yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/linear_coopmat_bench_subgroup64.log 2>&1
VK_COOPMAT_REQUIRED_SUBGROUP_SIZE=32 cmake-out-vk/backends/vulkan/test/custom_ops/linear_coopmat_bench > yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/linear_coopmat_bench_subgroup32.log 2>&1
```

## Result

Both subgroup 64 and subgroup 32 runs completed. Subgroup 32 was not a universal
win; it helped a few large cases and regressed others. Keep subgroup size as a
measured dispatch/tuning parameter rather than a global default.
