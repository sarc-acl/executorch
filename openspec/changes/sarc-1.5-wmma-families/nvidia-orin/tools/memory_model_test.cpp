// CPU-only interception of the real feature-query and device-create paths.
#include <executorch/backends/vulkan/runtime/vk_api/Adapter.cpp>
#include <executorch/backends/vulkan/runtime/vk_api/Device.cpp>

#include <cassert>
#include <iostream>

namespace {
uint32_t api_version;
bool extension_available;
bool feature_available;
bool queried;
bool enabled;
bool extension_enabled;
} // namespace

extern "C" {
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties(
    VkPhysicalDevice,
    VkPhysicalDeviceProperties* properties) {
  *properties = {};
  properties->apiVersion = api_version;
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice,
    VkPhysicalDeviceMemoryProperties* properties) {
  *properties = {};
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties(
    VkPhysicalDevice,
    uint32_t* count,
    VkQueueFamilyProperties* properties) {
  if (properties) {
    properties[0] = {};
    properties[0].queueFlags = VK_QUEUE_COMPUTE_BIT;
    properties[0].queueCount = 1;
  }
  *count = 1;
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures(
    VkPhysicalDevice,
    VkPhysicalDeviceFeatures* features) {
  *features = {};
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr(VkInstance, const char*) {
  return nullptr;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(
    VkPhysicalDevice,
    const char*,
    uint32_t* count,
    VkExtensionProperties* properties) {
  if (properties && extension_available) {
    properties[0] = {};
    std::strcpy(
        properties[0].extensionName, VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME);
    properties[0].specVersion = 3;
  }
  *count = extension_available ? 1 : 0;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(
    VkPhysicalDevice,
    VkPhysicalDeviceFeatures2* features) {
  for (auto* p = static_cast<VkBaseOutStructure*>(features->pNext); p;
       p = p->pNext) {
    if (p->sType ==
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES_KHR) {
      assert(extension_available);
      queried = true;
      auto* memory =
          reinterpret_cast<VkPhysicalDeviceVulkanMemoryModelFeaturesKHR*>(p);
      memory->vulkanMemoryModel = feature_available;
      memory->vulkanMemoryModelDeviceScope = feature_available;
      memory->vulkanMemoryModelAvailabilityVisibilityChains = feature_available;
    }
  }
}

VKAPI_ATTR void VKAPI_CALL
vkGetPhysicalDeviceProperties2(VkPhysicalDevice, VkPhysicalDeviceProperties2*) {
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(
    VkPhysicalDevice,
    const VkDeviceCreateInfo* info,
    const VkAllocationCallbacks*,
    VkDevice* device) {
  for (uint32_t i = 0; i < info->enabledExtensionCount; ++i) {
    if (std::strcmp(
            info->ppEnabledExtensionNames[i],
            VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME) == 0) {
      extension_enabled = true;
    }
  }
  for (auto* p = static_cast<const VkBaseInStructure*>(info->pNext); p;
       p = p->pNext) {
    if (p->sType ==
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES_KHR) {
      assert(!enabled);
      auto* memory =
          reinterpret_cast<const VkPhysicalDeviceVulkanMemoryModelFeaturesKHR*>(
              p);
      assert(memory->vulkanMemoryModel == VK_TRUE);
      assert(memory->vulkanMemoryModelDeviceScope == VK_FALSE);
      assert(memory->vulkanMemoryModelAvailabilityVisibilityChains == VK_FALSE);
      enabled = true;
    }
  }
  *device = VK_NULL_HANDLE;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vkGetDeviceQueue(VkDevice, uint32_t, uint32_t, VkQueue* queue) {
  *queue = VK_NULL_HANDLE;
}
}

int main() {
  int cases = 0;
  for (uint32_t version :
       {VK_API_VERSION_1_0,
        VK_API_VERSION_1_1,
        VK_API_VERSION_1_2,
        VK_API_VERSION_1_4}) {
    for (bool extension : {false, true}) {
      for (bool feature : {false, true}) {
        api_version = version;
        extension_available = extension;
        feature_available = feature;
        queried = enabled = extension_enabled = false;
        vkcompute::vkapi::PhysicalDevice physical(
            VK_NULL_HANDLE, VK_NULL_HANDLE);
        const bool expected_query = version >= VK_API_VERSION_1_1 && extension;
        assert(queried == expected_query);
        assert(
            physical.vulkan_memory_model_features.vulkanMemoryModel ==
            (expected_query && feature));
        std::vector<vkcompute::vkapi::Adapter::Queue> queues;
        std::vector<uint32_t> usage;
        vkcompute::vkapi::create_logical_device(physical, 1, queues, usage);
        assert(enabled == (expected_query && feature));
        assert(extension_enabled == enabled);
        assert(queues.size() == 1 && usage.size() == 1);
        ++cases;
      }
    }
  }
  std::cout << "memory-model query/create: PASS (" << cases
            << " cases; CPU mocks, no Vulkan loader or GPU)\n";
}
