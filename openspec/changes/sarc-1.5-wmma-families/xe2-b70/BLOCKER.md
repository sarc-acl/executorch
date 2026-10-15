# B70 model resource-limit blocker

2026-10-15. No model GPU job or detached baseline queue has launched.
The six-cell matrix is incomplete; no candidate pass or measured margin exists.

The assigned primary unit was confirmed by PCI and UUID in private query evidence.
`vkGetPhysicalDeviceImageFormatProperties` returned success with maxExtent
[2048,2048,2048] for `VK_FORMAT_R16G16B16A16_SFLOAT`, `VK_IMAGE_TYPE_3D`,
`VK_IMAGE_TILING_LINEAR`, flags=0 and usage=15 (sampled, storage, transfer source,
transfer destination). Samples=1, one mip and one array layer are supported.
The same format with optimal tiling also reports maximum width 2048.
The fp32 linear sampled/storage query used by the adapter returned success,
so the unchanged runtime chooses linear tiling. These were physical-device
queries only: no logical device, image allocation or dispatch was created.

Both supplied 8B schemes explicitly serialize FLOAT16=4, TEXTURE_3D=1,
TENSOR_WIDTH_PACKED=0 linear outputs of shape [1,2048,14336]. The runtime's
default axis map and four-element width packing require image extent
[14336/4,2048,1] = [3584,2048,1]. Required width 3584 exceeds supported width
2048. One witness per serialized graph (four witnesses across both schemes)
is preserved in the private blocker JSON; the larger model graph inspection
remains available. A single required over-limit resource blocks the full gate.
This finding does not assert that smaller models have completed a resource audit.

The exact topic export is `2fcc1c7e96a7774d80b66095aadb79700d3b48e3`.
The source derivation, also checked against that immutable export, is:

| Source | Relevant behavior |
|---|---|
| backends/vulkan/serialization/schema.fbs | Serialized dtype/storage/layout enum values |
| backends/vulkan/runtime/api/containers/Tensor.cpp:43 | Width packing uses blocks of four |
| backends/vulkan/runtime/api/containers/Tensor.cpp:310 | Default axis map is {0,1,2,2} |
| backends/vulkan/runtime/api/containers/Tensor.cpp:423 | Physical image extent calculation |
| backends/vulkan/runtime/api/containers/Tensor.cpp:598 | fp16 format, 3D type, preferred tiling, transfers enabled |
| backends/vulkan/runtime/vk_api/Types.h:35 | Half maps to R16G16B16A16_SFLOAT |
| backends/vulkan/runtime/vk_api/memory/Allocator.cpp:117 | Exact image usage flags |
| backends/vulkan/runtime/vk_api/memory/Image.cpp:151 | Zero image flags; one mip/layer/sample |
| backends/vulkan/runtime/api/Context.cpp:68 | Preferred tiling selection |
| backends/vulkan/runtime/vk_api/Adapter.cpp:267 | Adapter's fp32 linear-tiling support query |

Private evidence under the artifact root:

| Evidence | SHA256 |
|---|---|
| preflight/image-format-limits.json | 710aded9a075e8a299558508bfedebe88e44677725002ce01a596f31b8945aed |
| preflight/query-image-limits.py | 640553012351945ae4503430ba69b636d2bd9783a472302168ef1caa294f805e |
| preflight/model-vk-graphs.json | 86abe5b47d86529b36ba1850af87ffb1209fd008319a56fee00d74bdd5ecc648 |

`preflight/resource-limit-blocker.json` records the matching query, unit identity,
serialized witnesses, exact source hashes and derivation. No confidential host
or model location is copied here. The artifact-root RESOURCE_LIMIT_BLOCKED marker
prevents run-local GPU/build launches. The prepared baseline queue has not started.
Preserved builds, metadata, module checks and all source input evidence remain
available; changing hardware policy, model export or release code is not authorized.
The owner question appears once in STATUS.md. Work is parked pending that answer.
