# Required 8B model image extents exceed the queried limit

No model GPU process, correctness gate or timed session has run. The pristine parent
build and shipped-SPIR-V check passed; this is an allocation-legality prerequisite
finding, not a measured kernel failure.

Both approved 8B PTE files serialize fp16 texture3d, width-packed linear outputs
`[1, 2048, 14336]` (w1/w3) and inputs of the same shape (w2).
The loader's `VulkanBackend.cpp:335` passes these serialized dimensions/storage/layout
to `ComputeGraph::add_tensor`, using its default axis map.
`api/containers/Tensor.cpp:43` gives width-packed textures a four-element width block;
`:338` gives the default axis map `{0,1,2,2}`; `:423-470` derives the physical extents.
After padding to rank four, these are `[3584,2048,1]` texels. No alternative physical
axis mapping or image split is specified by the serialized layout/loader path.

Current read-only properties return maximum 3D extent `[2048,2048,2048]` for rgba16f,
rgba32i, rgba8i and rgba32f, in both linear and optimal tiling, usage sampled+storage+
transfer-src+transfer-dst and flags zero. The exact runtime rgba32f linear-tiling probe
(usage sampled+storage) succeeds, so this base selects linear tiling; its queried
extent bound is also 2048. The requirement exceeds the width bound by 1536 texels.
Raw model metadata, module declarations and query data are in the artifact preflight
records, including `models.json`, `model-summary.json`, `storage-properties-tiling.json`
and `parent-modules.json`.

The pinned runtime allocates those extents (`Tensor.cpp:705-737`). Its explicit extent
check at `:632-650` is compiled only with `VULKAN_DEBUG`; the parent runner/backend was
built Release without that definition. The nearby existing TODO records anecdotal
success over advertised limits on other drivers. That comment and historical successful
B580 runs do not establish queried legality or supply an exception for this stage.

Vulkan's [image creation requirements](https://docs.vulkan.org/refpages/latest/refpages/source/VkImageCreateInfo.html)
define imageCreateMaxExtent from matching image-format queries and require width,
height and depth to fit it (VUIDs 02252, 02253, 02254). The local knowledge base's Vulkan
rule 5 also describes matching exported texture limits to the device (source issue
#17299); its generic export advice does not authorize re-exporting these fixed models.

All 12 inventoried shader modules match their source-declared shared storage and fit
queried subgroup/workgroup/shared/MMA limits. This does not resolve the whole-model
resource violation. The two required 8B cells and texture production-diff cases at
N=14336 cannot be marked legal from those checks. The six-cell unmodified gate,
parent snapshot, A/A and finite matrix therefore remain blocked before GPU launch.
No family acceptance, measured margin, clock calibration or timed result is claimed.

The stage forbids re-exporting models, release-zone/runtime fixes, and extending
historical owner exceptions by analogy. The single owner question is in STATUS.md.
Independent work remaining is the CPU dispatch-name audit and bounded blocker review;
then the run waits for the owner without repeated queries, recomputation or reviews.
