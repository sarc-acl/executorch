# Memory-model feature enablement

The owner decision of 2026-10-15 03:57 UTC authorizes the narrowly scoped runtime
change identified in `runtime-feature-blocker.md`. The original base and its
recovered build remain preserved; they are not silently relabelled as a fixed
control. Both future measurement arms must use the new exported source commit,
including the same runtime change and finite dev selectors. Control remains
`orin-fused1`; candidates change only their registered linear choice. Attention
configuration is unchanged.

`vk_api/Device.h` stores a zero-initialized memory-model feature structure.
`Device.cpp` queries it only on the existing Vulkan 1.1 query path when the
extension is advertised. `Adapter.cpp` requests that extension only when the
queried base feature is supported, rechecks the filtered extension list, and
chains a fresh structure enabling only `vulkanMemoryModel` at `vkCreateDevice`.
The query structure's existing pNext links are never copied into device creation.
Device scope and multi-element availability/visibility chains stay disabled:
none of the 11 exact manifest modules declares those additional capabilities.
No shader, compiler, pipeline flag, selection table, golden or gate is edited.

The [extension specification](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_vulkan_memory_model.html)
requires Vulkan 1.1 or the properties2 extension and records promotion to 1.2.
Runtime.cpp requests Vulkan 1.1. This patch deliberately uses the advertised KHR
extension path on 1.1 and newer devices, rather than changing the application's
API version or relying on physical 1.2 support alone. The existing 1.0 path stays
unchanged. Devices without the extension or base feature receive no new enabled
extension or feature-chain entry. The
[feature definition](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceVulkanMemoryModelFeatures.html)
requires the enabled structure at device creation. Prior read-only target queries
confirm the extension and feature support; raw facts remain private artifacts.

`tools/test_memory_model.sh <artifact-output>` compiles the real Device.cpp and
Adapter.cpp against intercepted Vulkan calls, with no Vulkan loader or GPU.
Sixteen cases cross physical API 1.0/1.1/1.2/1.4, extension present/absent and
feature true/false. Assertions check actual query and device-create pNext chains,
filtered extension names, optional bits left disabled and queue creation.
Result: PASS, 16 cases. Build holds the shared desktop lock.

The unmodified `check.sh --no-build` reports FAIL solely for Device.cpp and
Device.h outside its zones. Adapter.cpp is already a listed hook. These are
covered by the new owner decision, not a clean check PASS. Twin checks and both
selector suites pass: release 1,240 checks / 31 rows; dev 1,921 checks / 43 rows /
533 candidates with no matrix profile. Raw output is in
`.artifacts/memory-model-tests/check.txt`. `git diff --check` passes.

Independent review, a new exact exported-commit cross-build, its golden check,
and normal GPU gates remain required. CPU mocks are not physical-device,
correctness, dispatch, A/A or timing evidence. The finite matrix is incomplete.
