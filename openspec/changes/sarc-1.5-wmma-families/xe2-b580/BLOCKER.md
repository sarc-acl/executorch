# Model image allocation prerequisites are blocked

Recovery audit, 2026-10-15. The original 8B finding is genuine, but incomplete:
the fixed RoPE prepack tensors also violate the matching image limit in **all six
models**. There are no legal whole-model cells to evaluate with the supplied inputs.
No model forward method, GPU gate or timed session has been started. This is not
a measured kernel failure or a six-cell result.

## Actual allocation path

References below are under `backends/vulkan/runtime/` at pristine base `6178971845`.
All twelve allocation-path/schema source files were byte-compared with both the
base git objects and the preserved parent export. Hashes are in the private
`preflight/recovery/allocation-source-hashes.json` record.

1. `VulkanBackend.cpp:272` resolves serialized storage/layout and treats negative
   constant IDs as actual tensors. `:335` passes dimensions, dtype, storage, layout
   and shared-object ID to `ComputeGraph::add_tensor`. `:454` adds **every value
   before operators**, so a later operator rewrite cannot avoid initial creation.
   `ResolveLayouts.cpp:184` only resolves convolution/dynamic int8 layouts; none
   of these explicit fp16 width-packed tensors is changed.
2. `graph/ComputeGraph.h:739` defaults the axis-map argument. `ComputeGraph.cpp:488`
   constructs a vTensor with `allocate_memory=false`, which still constructs the
   image object. This flag defers memory binding, not `vkCreateImage`.
3. `api/containers/Tensor.cpp:43`, `:338`, `:383`, and `:423` use width blocks of
   four, default axes `{0,1,2,2}`, leading dimension padding to rank four, and
   texel extents `[W/4,H,C*B]`. `:598` selects `VK_IMAGE_TYPE_3D` and 3D view for
   texture3d, maps half to RGBA16F, and enables transfers. `:705` constructs that
   storage immediately, before the tensor can be virtually resized.
4. `vk_api/Adapter.cpp:267` probes RGBA32F, 3D, linear tiling, sampled+storage,
   flags zero. Its preserved query returns success. `api/Context.cpp:68` therefore
   selects linear tiling. This probe chooses tiling; it does not establish the
   extent limit for the eventual half image with transfers.
5. `vk_api/memory/Allocator.cpp:117` sets usage sampled+storage+transfer-src+
   transfer-dst (`0x0f`). `memory/Image.cpp:151` creates the image with flags zero,
   null pNext, one mip, one array layer, one sample, exclusive sharing, undefined
   initial layout. `:181` calls `vkCreateImage` even without allocated memory.
   No external-memory, sparse, array, extended-usage or alternate-axis exemption
   is involved.

The preserved **matching** query is format RGBA16F (97), image type 3D (2), tiling
linear (1), usage `0x0f`, flags zero: success with maximum extent
`[2048,2048,2048]`. The query helper's ctypes signature/field layout matches the
pinned Vulkan headers. RGBA32F has the same bound if half were promoted to float;
the queried fp16 buffer/arithmetic features support half here. Optimal tiling also
reports the same extent bound and would not resolve these dimensions.

The parent's backend compile flags have no `VULKAN_DEBUG`; its CMake build type
is unset (the old description calling the backend Release was inaccurate).
The runner is Release. The debug-only extent guard in `Tensor.cpp:632` is therefore
absent. A disabled guard and the adjacent historical anecdote do not confer legality.
Vulkan requires extents to fit the matching image-format limits; see the official
[image creation requirements](https://docs.vulkan.org/refpages/latest/refpages/source/VkImageCreateInfo.html),
VUIDs 02252–02254. The KB's texture-limit guidance is background, not authorization
to change these fixed exports.

## Independent CPU and serialized evidence

`image_extents.cpp` calls the pure extent functions linked from the **already built
parent** archive, without constructing a Context, vTensor or Vulkan device:

| Serialized shape | Physical extent | Conflict |
|---|---|---|
| `[1,2048,14336]` | `[3584,2048,1]` | 8B width exceeds 2048 |
| `[5120,32]` | `[8,5120,1]` | 1B RoPE height exceeds 2048 |
| `[5120,64]` | `[16,5120,1]` | 3B/8B RoPE height exceeds 2048 |

`audit_allocations.py` enumerates serialized nonconstant texture3d values, records
their IDs, dimensions, shared-object IDs and consuming calls, and compares extents
with the saved matching query. Each model contains two offending RoPE prepack
outputs used by slice operations. The 8B 4w/8da4w graphs additionally contain
224/192 offending wide tensors. The audit establishes definite violations; it
does not claim a complete legality proof for operator-created temporary resources.
It does not execute or attempt any invalid allocation.

`runtime_constants.cpp` links only existing parent runtime/data-loader archives,
skips `forward`, and executes the nine constant methods per PTE on CPU. All 60
outputs match the serialized records, including context 2560 in every model.
Linear group 128 and q4 embedding group 32 are serialized graph facts; these PTEs
do not expose them as separate constant methods. No substitute runtime claim is made.

Exact private evidence is under this run's `.artifacts/preflight/recovery/`:
`allocation-audit.json`, `image-extents.out`, `runtime-constants.csv`,
`runtime-constant-validation.json`, compiler command/log records and `SHA256SUMS`.
`preflight/storage-properties-tiling.json` supplies the original query;
`preflight/model-hashes.json` pins the fixed model inputs.
Small performance coverage is in `results/cell-coverage.csv`; it contains no timings.

## Remaining dependency

Every candidate profile inherits these allocations. All twelve inventoried module
declarations fit the queried kernel limits, and the CPU selector audit covers all
twelve names, but neither fact makes these models legal. The parent snapshot,
A/A, full gate, runtime dispatch evidence and measured matrix remain **BLOCKED**.

The single owner dependency in STATUS.md now covers both resource classes. A
conforming resolution must supply a separately authorized, reviewed model/runtime
path that fits the queried limits for both the shared RoPE tables and 8B linear
tensors, or an owner-provided environment whose exact matching queries actually
support them. Any changed inputs/runtime need new provenance and revalidation
before baseline/gating. Merely transposing axes cannot fit a 5120 axis into a
2048 cube. No allocation-limit waiver, driver change, model re-export or runtime
patch is performed within this stage. No current legal cell is being left unrun.
