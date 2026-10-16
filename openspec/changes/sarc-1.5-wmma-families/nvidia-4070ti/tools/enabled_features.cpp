#include <cstdio>
#include <cstring>

#include <executorch/backends/vulkan/runtime/vk_api/Runtime.h>

namespace {
PFN_vkGetInstanceProcAddr loader = nullptr;
PFN_vkCreateDevice create_device = nullptr;
bool observed = false;
bool enabled = false;

VKAPI_ATTR VkResult VKAPI_CALL record_create_device(
    VkPhysicalDevice physical,
    const VkDeviceCreateInfo* info,
    const VkAllocationCallbacks* allocator,
    VkDevice* device) {
  bool extension = false;
  for (uint32_t i = 0; i < info->enabledExtensionCount; ++i) {
    extension |= std::strcmp(
                     info->ppEnabledExtensionNames[i],
                     VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME) == 0;
  }
  VkBool32 memory_model = VK_FALSE;
  VkBool32 device_scope = VK_FALSE;
  VkBool32 chains = VK_FALSE;
  for (auto* node = static_cast<const VkBaseInStructure*>(info->pNext); node;
       node = node->pNext) {
    if (node->sType ==
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES_KHR) {
      const auto* features = reinterpret_cast<
          const VkPhysicalDeviceVulkanMemoryModelFeaturesKHR*>(node);
      memory_model = features->vulkanMemoryModel;
      device_scope = features->vulkanMemoryModelDeviceScope;
      chains = features->vulkanMemoryModelAvailabilityVisibilityChains;
    }
  }
  const auto result = create_device(physical, info, allocator, device);
  observed = true;
  enabled = extension && memory_model && !device_scope && !chains &&
      result == VK_SUCCESS;
  std::printf(
      "{\"extension_enabled\":%s,\"vulkanMemoryModel\":%u,"
      "\"vulkanMemoryModelDeviceScope\":%u,"
      "\"vulkanMemoryModelAvailabilityVisibilityChains\":%u,"
      "\"vkCreateDevice_result\":%d}\n",
      extension ? "true" : "false", memory_model, device_scope, chains, result);
  return result;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL record_proc(
    VkInstance instance,
    const char* name) {
  auto proc = loader(instance, name);
  if (std::strcmp(name, "vkCreateDevice") == 0 && proc) {
    create_device = reinterpret_cast<PFN_vkCreateDevice>(proc);
    return reinterpret_cast<PFN_vkVoidFunction>(record_create_device);
  }
  return proc;
}
} // namespace

int main() {
  if (volkInitialize() != VK_SUCCESS) {
    return 2;
  }
  loader = vkGetInstanceProcAddr;
  vkGetInstanceProcAddr = record_proc;
  vkcompute::vkapi::Runtime runtime(
      {false, true, vkcompute::vkapi::AdapterSelector::Auto, 1, ""});
  runtime.get_adapter_p();
  return observed && enabled ? 0 : 3;
}
