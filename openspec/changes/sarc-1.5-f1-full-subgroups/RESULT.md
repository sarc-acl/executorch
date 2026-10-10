# F1 on the Arc B580: pipelines created with the full-subgroups flag

Task `f1-b580`, branch `topic/f1-full-subgroups` (not pushed). Arc B580 of `fedora`, Vulkan device 0, ANV Mesa
26.2.3, kernel 7.2.9-200.fc44, 2026-10-09 23:07 to 2026-10-10 03:18 UTC. Old = build `topic7` (`e1e450530`, the
fused-port campaign's final build), new = build `f1` (`bf9ae6ba3`). Running record: `LOG.md`; raw data: `data/`.

**Answer.** Creating the cooperative-matrix pipelines with `REQUIRE_FULL_SUBGROUPS` changes nothing measurable on
this card. Correctness: every gate item of both configurations reads the same on the new build as on the old
one, and every dumped value (444 logits vectors, the SDPA error records) is identical. Speed: **no effect** by
the pre-registered rule in both configurations (`b580-fused1`: geomean -0.04 %, largest cell -0.88 %; no
environment: +0.00 % in every cell). The validation layer reports VUID 10770 on the old build and no longer on
the new one. Two things a reader should know before relying on this: the first session with `b580-fused1` was
incomplete because the desktop was in use (its one permitted repeat is the table below), and the validation
layer reports seven other VUIDs on both builds that this task did not touch (section 3).

## 1. The change (`bf9ae6ba3`, "Release-zone change (owner decision 2026-10-10): full-subgroups flag")

```
backends/vulkan/runtime/vk_api/Adapter.cpp  | 10 ++++++--
 backends/vulkan/runtime/vk_api/Pipeline.cpp | 37 +++++++++++++++++++++++++----
 backends/vulkan/runtime/vk_api/Pipeline.h   |  9 +++++--
 sarc/HOOKS                                  |  5 ++--
 4 files changed, 50 insertions(+), 11 deletions(-)
```

Form B: both compute stage creation sites of `Pipeline.cpp` set the flag when a required subgroup size is in
force, the device reports `computeFullSubgroups`, and the local size in X (specialization constant 0) is a
multiple of the required size. The third condition is not needed by any SARC variant (section 4); it keeps one
existing pipeline valid, the 4 x 4 x 1 inspection pipeline of `Context::get_shader_pipeline`
(`ETVK_INSPECT_PIPELINES` builds). The cache key is unchanged: the flag is a function of the key and of a device
constant, which the pipeline cache now carries (hence `Pipeline.h` and two constructor calls in `Adapter.cpp`;
`sarc/HOOKS` lists them). `Device.cpp` is not changed: the feature was already enabled, not only reported
(`Adapter.cpp` `create_logical_device` chains the queried `VkPhysicalDeviceSubgroupSizeControlFeatures` whenever
`subgroupSizeControl` is reported); the layer confirms it (no VUID `...-flags-02785` on the new build). No
shader, tile, yaml or table is changed.

```diff
diff --git a/backends/vulkan/runtime/vk_api/Adapter.cpp b/backends/vulkan/runtime/vk_api/Adapter.cpp
index 1a7a8907f..083f3cd9c 100644
--- a/backends/vulkan/runtime/vk_api/Adapter.cpp
+++ b/backends/vulkan/runtime/vk_api/Adapter.cpp
@@ -321,3 +321,6 @@ Adapter::Adapter(
       pipeline_layout_cache_(device_.handle),
-      compute_pipeline_cache_(device_.handle, cache_data_path),
+      compute_pipeline_cache_(
+          device_.handle,
+          cache_data_path,
+          supports_compute_full_subgroups()),
       sampler_cache_(device_.handle),
@@ -345,3 +348,6 @@ Adapter::Adapter(
       pipeline_layout_cache_(device_.handle),
-      compute_pipeline_cache_(device_.handle, cache_data_path),
+      compute_pipeline_cache_(
+          device_.handle,
+          cache_data_path,
+          supports_compute_full_subgroups()),
       sampler_cache_(device_.handle),
diff --git a/backends/vulkan/runtime/vk_api/Pipeline.cpp b/backends/vulkan/runtime/vk_api/Pipeline.cpp
index 15151e2d8..35dcc8d94 100644
--- a/backends/vulkan/runtime/vk_api/Pipeline.cpp
+++ b/backends/vulkan/runtime/vk_api/Pipeline.cpp
@@ -268,2 +268,24 @@ void swap(PipelineLayout& lhs, PipelineLayout& rhs) noexcept {
 
+namespace {
+
+// Cooperative-matrix shaders below SPIR-V 1.6 need full subgroups
+// (VUID-RuntimeSpirv-OpTypeCooperativeMatrixKHR-10770). The flag is only valid
+// when the local size in X, specialization constant 0, is a multiple of the
+// required subgroup size (VUID-VkPipelineShaderStageCreateInfo-pNext-02757).
+VkPipelineShaderStageCreateFlags full_subgroups_flag(
+    const ComputePipeline::Descriptor& descriptor,
+    const bool full_subgroups) {
+#ifdef VK_EXT_subgroup_size_control
+  const SpecVarList& spec_vars = descriptor.specialization_constants;
+  const uint32_t subgroup_size = descriptor.required_subgroup_size;
+  if (full_subgroups && subgroup_size > 0u && spec_vars.size() > 0u &&
+      spec_vars.at(0).value.as_uint32 % subgroup_size == 0u) {
+    return VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT;
+  }
+#endif /* VK_EXT_subgroup_size_control */
+  return 0u;
+}
+
+} // namespace
+
 //
@@ -278,3 +300,4 @@ ComputePipeline::ComputePipeline(
     const ComputePipeline::Descriptor& descriptor,
-    VkPipelineCache pipeline_cache)
+    VkPipelineCache pipeline_cache,
+    const bool full_subgroups)
     : device_(device), handle_{VK_NULL_HANDLE} {
@@ -304,3 +327,3 @@ ComputePipeline::ComputePipeline(
       shader_stage_pnext, // pNext
-      0u, // flags
+      full_subgroups_flag(descriptor, full_subgroups), // flags
       VK_SHADER_STAGE_COMPUTE_BIT, // stage
@@ -419,5 +442,7 @@ ComputePipelineCache::ComputePipelineCache(
     VkDevice device,
-    const std::string& cache_data_path)
+    const std::string& cache_data_path,
+    const bool full_subgroups)
     : cache_mutex_{},
       device_(device),
+      full_subgroups_(full_subgroups),
       pipeline_cache_{VK_NULL_HANDLE},
@@ -445,2 +470,3 @@ ComputePipelineCache::ComputePipelineCache(
       device_(other.device_),
+      full_subgroups_(other.full_subgroups_),
       pipeline_cache_(other.pipeline_cache_),
@@ -539,3 +565,3 @@ void ComputePipelineCache::create_pipelines(
         shader_stage_pnext, // pNext
-        0u, // flags
+        full_subgroups_flag(key, full_subgroups_), // flags
         VK_SHADER_STAGE_COMPUTE_BIT, // stage
@@ -587,3 +613,4 @@ VkPipeline ComputePipelineCache::retrieve(
                  {key,
-                  ComputePipelineCache::Value(device_, key, pipeline_cache_)})
+                  ComputePipelineCache::Value(
+                      device_, key, pipeline_cache_, full_subgroups_)})
              .first;
diff --git a/backends/vulkan/runtime/vk_api/Pipeline.h b/backends/vulkan/runtime/vk_api/Pipeline.h
index 5e286ad77..12455558c 100644
--- a/backends/vulkan/runtime/vk_api/Pipeline.h
+++ b/backends/vulkan/runtime/vk_api/Pipeline.h
@@ -174,3 +174,4 @@ class ComputePipeline final {
       const Descriptor& descriptor,
-      VkPipelineCache pipeline_cache);
+      VkPipelineCache pipeline_cache,
+      const bool full_subgroups);
 
@@ -246,3 +247,4 @@ class ComputePipelineCache final {
       VkDevice device,
-      const std::string& cache_data_path);
+      const std::string& cache_data_path,
+      const bool full_subgroups);
 
@@ -308,2 +310,5 @@ class ComputePipelineCache final {
   VkDevice device_;
+  // computeFullSubgroups of the device: pipelines with a required subgroup
+  // size are created with REQUIRE_FULL_SUBGROUPS.
+  bool full_subgroups_;
   VkPipelineCache pipeline_cache_;
```

## 2. Specification sentences relied on (`vulkan-docs` MCP server, refpages `latest`)

- VUID-RuntimeSpirv-OpTypeCooperativeMatrixKHR-10770 (`RuntimeSpirv`): "Any pipeline containing a shader with
  OpTypeCooperativeMatrixKHR or OpCooperativeMatrix*KHR instructions must be created with the
  VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT flag or the shader module must be version 1.6 or
  greater" (as quoted in the fused-port `STATUS.md` from the same source; the server's exact-VUID lookup
  returned nothing for it in this session, the page search finds it on `RuntimeSpirv`).
- VUID-VkPipelineShaderStageCreateInfo-pNext-02757: "If a VkPipelineShaderStageRequiredSubgroupSizeCreateInfo
  structure is included in the pNext chain, and flags has the
  VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT flag set, the local workgroup size in the X
  dimension of the pipeline must be a multiple of
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo::requiredSubgroupSize".
- VUID-VkPipelineShaderStageCreateInfo-flags-02785: "If flags has the
  VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT flag set, the computeFullSubgroups feature must be
  enabled"; -flags-08988: "... stage must be one of VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_STAGE_TASK_BIT_EXT, or
  VK_SHADER_STAGE_COMPUTE_BIT".
- `VkPipelineShaderStageCreateFlagBits`: "VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT specifies
  that the subgroup sizes must be launched with all invocations active in the task, mesh, or compute stage."
- VUID-VkPipelineShaderStageCreateInfo-module-08987: "If module uses the OpTypeCooperativeMatrixKHR instruction
  with a Scope equal to Subgroup, then the local workgroup size in the X dimension of the pipeline must be a
  multiple of the effective subgroup size" (already required without the flag; consistent with section 4).

## 3. Validation layer, before and after (`data/validation/`)

`VK_LAYER_KHRONOS_validation` 1.4.341 is installed (`vulkan-validation-layers-1.4.341.0-2.fc44`). One linear
suite (`--linear --regime=prefill --scheme=4w --storage=texture3d`) and one SDPA tier
(`--sdpa-correctness-only --sdpa-tier=all`) of `test_llama_microbench` under `b580-fused1`, layer on:

| message | old `topic7` | new `f1` |
|---|---:|---:|
| VUID-RuntimeSpirv-OpTypeCooperativeMatrixKHR-10770 | **28** | **0** |
| VUID-RuntimeSpirv-cooperativeMatrixFlexibleDimensionsMaxDimension-10167 | 40 | 40 |
| VUID-RuntimeSpirv-vulkanMemoryModel-06266 | 28 | 28 |
| VUID-vkCmdDispatch-imageLayout-00344 | 20 | 20 |
| VUID-vkCmdDraw-None-09600 | 20 | 20 |
| VUID-VkImageCreateInfo-extent-02252 | 8 | 8 |
| VUID-VkShaderModuleCreateInfo-pCode-08740 / -08742 | 4 / 4 | 4 / 4 |

The old build's message: "vkCreateComputePipelines(): pCreateInfos[0].stage SPIR-V (VK_SHADER_STAGE_COMPUTE_BIT)
contains SPV_KHR_cooperative_matrix which requires SPIR-V 1.6 (Vulkan 1.3). In order to use it with older
versions, you need to use VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT (which requires
VK_EXT_subgroup_size_control)." It is gone with the flag, and the flag raised no new message (no 02757, no
02785). **The seven other VUIDs are identical in count on both builds and are not part of F1**: an 8 x 16
cooperative-matrix type against `cooperativeMatrixFlexibleDimensionsMaxDimension` 0 (10167), the
`VulkanMemoryModel` capability / QueueFamily scope without the `vulkanMemoryModel` feature enabled (06266,
08740, 08742), image-layout tracking (00344, 09600) and an image wider than the format's maximum extent in a
test (02252). So the new pipelines are no longer invalid for F1's reason, but the layer does not call them clean.
These are reported, not investigated.

## 4. Local size against required subgroup size (`data/local-size.csv`, `tools/local_size.py`)

The local size in X is a specialization constant set by the op from the selection row
(`sg_grid_x * sg_grid_y * subgroup_size`), or S x G for the B580 fused kernel. Over every generated SARC variant
with a required subgroup size, joined with all 250 rows and sweep candidates (`data/rows.csv`): 401 lines, 33
release zone and 368 development zone, **0 violators**; every row's subgroup size equals its shader's. Nothing
had to be listed as work left, and no tile was changed.

## 5. SPIR-V identity (`data/spirv-identity.txt`)

6304 `.spv` files in each build tree (backend, llama, tests, traced excluded), sha256 lists identical: **0
differences**. Shipped-SPIR-V golden of `f1`: PASS (53 variants).

## 6. Correctness on the new build, both configurations

| item | `b580-fused1` (`sA-fused1`) | no environment (`sB-noenv`) |
|---|---|---|
| unmodified `verify.sh` against the parent snapshot (`gate_check.py`) | PASS, same status as `s0-parent-verify` | PASS, same status as `s0-topic7-noenv` |
| `verify.out` line by line against the old build's, rates removed, kernel names included (`verify_diff.py`) | VERIFY_SAME, 34 lines (old: `s6-final`) | VERIFY_SAME, 34 lines (old: `s0-topic7-noenv`) |
| production-diff | 12 of 12 cases ALL PASSED | 12 of 12 cases ALL PASSED |
| SDPA tiers `all` / `extended` / `full`, 12 passes each | 0 mismatches, finite error, `pairing=ok` (fused kernel) | 0 mismatches and finite error records in all 36 passes; the stock SDPA kernels run (no released SDPA row for this card), so the test's return status is 1 as on the old build: device status, `gate-control.txt` |
| SDPA records old against new (`sdpa-records.txt`; the test does not dump outputs) | IDENTICAL in 4 tiers; all 36 gate passes equal the old build's records | IDENTICAL in 4 tiers; all 36 passes equal |
| next token, three prompts, six cells | SAME in 18 of 18 (and 18 of 18 in the repeat) | SAME in 18 of 18 |
| last-position logits, old against new, byte for byte (`logits-identity.txt`) | 222 vectors, 0 differ | 222 vectors, 0 differ |
| gate verdict | 34 PASS, 3 FAIL: **GATE_FAIL on timing completeness only** (below) | **GATE_PASS**, 34 PASS, 0 FAIL |

`verify.sh` reports `correctness rc=1` in both configurations exactly as the old build and the parent do (28 of
28 numeric cases and 4 of 4 rank-3 cases PASSED; the campaign's known device status). No pipeline failed to be
created.

## 7. Speed, old against new

Pre-registered in `LOG.md` (commit `8368f2b22`, before the first timed run): "no effect" = every cell inside
+-2 % and the geomean inside +-1 %. Tok/s, median of the first 7 valid runs per arm, arms interleaved; tables
printed by `tools/session_table.py <runs.csv>`.

**Session A, `b580-fused1`.** The first session (`sA-fused1`, 00:09 to 00:40 UTC) ran while the desktop was in
use: 166 timed runs, 78 valid, and the cells 3B 4w (5 / 7 valid) and 3B 8da4w (4 / 2) did not reach 7 valid
runs within the 12 extra pairs; its four complete cells read -0.72 / +0.00 / +0.22 / +1.35 %. That is the
gate's three FAIL lines. By the task's rule the session was repeated once, 70 minutes later (`sA2-fused1`,
01:50 to 02:13 UTC, timed session only, desktop idle), `E2E5_OK`:

| cell | old `topic7` | new `f1` | new / old | difference | spread old / new | valid old / new | invalid | foreign share median / max |
|---|---:|---:|---:|---:|---|---|---|---|
| 1B 4w | 14948.90 | 15058.80 | 1.00735 | +0.74 % | 0.7 / 0.7 % | 7 / 7 | none | 0.00 / 0.00 % |
| 1B 8da4w | 18285.70 | 18123.90 | 0.99115 | -0.88 % | 0.9 / 1.8 % | 7 / 7 | none | 0.00 / 0.00 % |
| 3B 4w | 5461.33 | 5446.81 | 0.99734 | -0.27 % | 0.0 / 0.3 % | 7 / 7 | none | 0.00 / 0.00 % |
| 3B 8da4w | 6826.67 | 6826.67 | 1.00000 | +0.00 % | 0.3 / 0.7 % | 7 / 7 | none | 0.00 / 0.00 % |
| 8B 4w | 2381.40 | 2381.40 | 1.00000 | +0.00 % | 0.2 / 0.2 % | 7 / 7 | none | 0.00 / 0.00 % |
| 8B 8da4w | 3141.10 | 3145.93 | 1.00154 | +0.15 % | 0.3 / 0.2 % | 7 / 7 | none | 0.00 / 0.00 % |

Geomean over 6 cells: 0.99955 (-0.04 %); largest cell difference -0.88 %; 84 timed runs, 84 valid. Verdict: NO EFFECT (every cell inside +-2 %, geomean inside +-1 %)

**Session B, no environment** (`sB-noenv`, 01:35 to 01:46 UTC, desktop idle), `E2E5_OK`:

| cell | old `topic7` | new `f1` | new / old | difference | spread old / new | valid old / new | invalid | foreign share median / max |
|---|---:|---:|---:|---:|---|---|---|---|
| 1B 4w | 8641.35 | 8641.35 | 1.00000 | +0.00 % | 0.8 / 0.4 % | 7 / 7 | none | 0.00 / 0.00 % |
| 1B 8da4w | 8827.59 | 8827.59 | 1.00000 | +0.00 % | 0.4 / 0.4 % | 7 / 7 | none | 0.00 / 0.00 % |
| 3B 4w | 3442.02 | 3442.02 | 1.00000 | +0.00 % | 0.2 / 0.2 % | 7 / 7 | none | 0.00 / 0.00 % |
| 3B 8da4w | 3518.90 | 3518.90 | 1.00000 | +0.00 % | 0.3 / 0.0 % | 7 / 7 | none | 0.00 / 0.00 % |
| 8B 4w | 1726.81 | 1726.81 | 1.00000 | +0.00 % | 0.2 / 0.1 % | 7 / 7 | none | 0.00 / 0.00 % |
| 8B 8da4w | 1845.05 | 1845.05 | 1.00000 | +0.00 % | 0.3 / 0.2 % | 7 / 7 | none | 0.00 / 0.00 % |

Geomean over 6 cells: 1.00000 (+0.00 %); largest cell difference +0.00 %; 84 timed runs, 84 valid. Verdict: NO EFFECT (every cell inside +-2 %, geomean inside +-1 %)

The medians of an arm are often equal because a 2048-token prefill is timed in whole milliseconds. Both idle
sessions read a foreign engine share of 0.00 % in every run (the desktop session reported idle, 22 DRM clients
present); the earlier campaign's idle sessions read 0.25 to 0.46 %. The reading is taken as recorded; the
absolute rates agree with that campaign's (`s6-final` final arm 15058.80 / 18285.70 / 5461.33 / 6849.50 /
2381.40 / 3145.93; `s7-pristine` pristine arm 8677.97 / 8865.80 / 3442.02 / 3524.96 / 1726.81 / 1845.05).

**Kernel level** (`data/kernel/kernel.csv`; 3 processes per arm, interleaved; speed new / old per case from the
median of the process medians):

| environment | suite | cases | geomean | min | max | outside +-2 % |
|---|---|---:|---:|---:|---:|---:|
| `b580-fused1` | linear-4w | 12 | -0.02 % | -0.08 % | +0.08 % | 0 |
| `b580-fused1` | linear-8da4w | 12 | +0.01 % | -0.03 % | +0.14 % | 0 |
| `b580-fused1` | sdpa | 27 | -0.01 % | -0.43 % | +0.53 % | 0 |
| `no environment` | linear-4w | 12 | -0.02 % | -0.10 % | +0.11 % | 0 |
| `no environment` | linear-8da4w | 12 | +0.01 % | -0.05 % | +0.05 % | 0 |
| `no environment` | sdpa | 36 | -0.06 % | -0.70 % | +0.14 % | 0 |

`--storage=texture3d` gives 12 linear cases per scheme. Under `b580-fused1` 9 of the 36 SDPA cases are the
per-stage placeholders of the fused kernel (time 0) and are not counted; the fused attention kernel itself is
the `total` / `coopmat` case: 3.1-8b 1463.0 -> 1466.7 us (-0.26 %); 3.2-3b 1142.2 -> 1142.6 us (-0.04 %); 3.2-1b 725.5 -> 721.7 us (+0.53 %). The microbench's return status (1 for the linear suites and for SDPA
without a profile: its dispatch-expectation report on this card) is the same in both arms in all 36 runs.

**Verdict: no effect**, in both sessions and at kernel level.

## 8. What this means for the other devices

The static rule holds for every selected variant of every device, so the flag can be set everywhere without
touching a tile (local size X / required subgroup size, from `data/local-size.csv`):

| device row | variants selected | local size X / required size | multiple |
|---|---:|---|---|
| 4070 ti super | 7 | 256 / 32, 512 / 32 | all |
| 780m | 7 | 256 / 32, 256 / 64 | all |
| 7900 xtx | 5 | 256 / 32, 256 / 64 | all |
| adreno | 3 | 128 / 64 | all |
| bmg g21 | 6 | 256 / 16, 512 / 16 | all |
| bmg g31 | 6 | 256 / 16, 512 / 16 | all |
| mali-g1 | 3 | 256 / 16 | all |
| rx 7600 | 5 | 256 / 32, 256 / 64 | all |
| tegra orin | 5 | 128 / 32, 256 / 32, 512 / 32 | all |
| xclipse | 7 | 128 / 32, 256 / 32, 256 / 64 | all |

What each device still needs is its own evidence, since this task measured one driver: (1) that it reports
`computeFullSubgroups` (the flag is only set then; a device that does not report it keeps F1 and would need
SPIR-V 1.6 on a Vulkan 1.3 instance instead; here only the B580 was seen to report it; the RADV iGPU of this workstation does too but is not a device of
the tables); (2) its gate and its timed sessions on a build with this commit, because a driver may
compile a full-subgroups pipeline differently even where ANV does not (the SPIR-V is unchanged, the pipeline
creation is not); (3) a validation-layer run, which on this card also shows the seven other messages of
section 3 that apply to the shared shaders and runtime. The upstream cooperative-matrix shaders (10 with a
required subgroup size) get the flag by the same code; their local sizes are set in upstream op code and were
not derived here, the run-time multiple check covers them.
