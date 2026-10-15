# Logical-device feature prerequisite

Pinned runtime: `617897184529ff180af491ee0d5d7de778cbb6c6`.
The 11 reviewed NVIDIA manifest modules were compiled with the recorded pinned
compiler. Every exact module declares `OpCapability VulkanMemoryModel`,
`OpExtension "SPV_KHR_vulkan_memory_model"` and `OpMemoryModel Logical Vulkan`.
The parent golden passes; standalone module validation passes. Neither check
supplies a feature absent from logical-device creation.

At the pinned source, `backends/vulkan/runtime/vk_api/Adapter.cpp:100-251`
enumerates requested extensions and creates the complete feature chain. There
is no memory-model extension, `VkPhysicalDeviceVulkanMemoryModelFeatures` or
`VkPhysicalDeviceVulkan12Features`. `Device.cpp` and `Device.h` contain no
memory-model feature query/storage. The physical unit supports the feature,
but the runtime does not enable it. This conclusion is from the exact source
and modules, not a validation-layer GPU job; none was started.

The [feature definition](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceVulkanMemoryModelFeatures.html)
requires a corresponding feature structure with desired members set true in
the device-create chain. The [SPIR-V capability table](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html)
maps `VulkanMemoryModel` to `vulkanMemoryModel`.
[Shader-module validation](https://docs.vulkan.org/refpages/latest/refpages/source/VkShaderModuleCreateInfo.html)
VUID `VkShaderModuleCreateInfo-pCode-08740` requires satisfying the capability's
corresponding requirement. Repository Vulkan knowledge-base articles do not
describe this feature-enablement omission; the conclusion uses the source and
primary specification.

The stage prohibits release/runtime edits and fixes to pinned sources without
a new owner ruling. STATUS.md records one concrete question about a revised
base or narrowly authorized enablement. No waiver is inferred from historical
measurements, driver tolerance, hardware support or the golden PASS.
The matrix remains unlaunched and incomplete. Reuse the successful parent and
CPU evidence where applicable; a revised runtime would require a new exact
build and gates before its measurement.
