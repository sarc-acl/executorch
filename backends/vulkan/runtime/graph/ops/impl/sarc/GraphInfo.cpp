/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/GraphInfo.h>

#include <executorch/backends/vulkan/runtime/vk_api/Runtime.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <mutex>

namespace vkcompute {
namespace sarc {

namespace {

struct FamilyKey {
  uint32_t vendor_id = 0;
  uint32_t driver_id = 0;
  std::vector<CoopmatShape> coopmat_shapes;
};

#ifdef VK_KHR_cooperative_matrix
CoopmatType to_coopmat_type(const VkComponentTypeKHR t) {
  switch (t) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR:
      return CoopmatType::kFp16;
    case VK_COMPONENT_TYPE_FLOAT32_KHR:
      return CoopmatType::kFp32;
    case VK_COMPONENT_TYPE_SINT8_KHR:
      return CoopmatType::kSint8;
    case VK_COMPONENT_TYPE_SINT32_KHR:
      return CoopmatType::kSint32;
    default:
      return CoopmatType::kOther;
  }
}
#endif /* VK_KHR_cooperative_matrix */

// Vendor ID, driver ID and the subgroup-scope cooperative-matrix shapes of a
// physical device, queried once. Entry points are resolved through the
// instance, as Device.cpp does, for builds that link Vulkan without volk.
FamilyKey query_family_key(const VkPhysicalDevice handle) {
  FamilyKey key;
  const VkInstance instance = vkapi::runtime()->instance();
  auto get_properties2 =
      (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(
          instance, "vkGetPhysicalDeviceProperties2");
  if (get_properties2 == nullptr) {
    get_properties2 =
        (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(
            instance, "vkGetPhysicalDeviceProperties2KHR");
  }
  if (get_properties2 != nullptr) {
    VkPhysicalDeviceDriverProperties driver{};
    driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &driver;
    get_properties2(handle, &properties);
    key.vendor_id = properties.properties.vendorID;
    key.driver_id = static_cast<uint32_t>(driver.driverID);
  }
#ifdef VK_KHR_cooperative_matrix
  auto get_coopmat_properties =
      (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
          vkGetInstanceProcAddr(
              instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
  uint32_t count = 0;
  if (get_coopmat_properties != nullptr &&
      get_coopmat_properties(handle, &count, nullptr) == VK_SUCCESS &&
      count > 0) {
    std::vector<VkCooperativeMatrixPropertiesKHR> props(count);
    for (auto& p : props) {
      p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
      p.pNext = nullptr;
    }
    if (get_coopmat_properties(handle, &count, props.data()) == VK_SUCCESS) {
      for (uint32_t i = 0; i < count; ++i) {
        const auto& p = props[i];
        if (p.scope != VK_SCOPE_SUBGROUP_KHR || p.AType != p.BType ||
            p.CType != p.ResultType) {
          continue;
        }
        key.coopmat_shapes.push_back(
            {p.MSize,
             p.NSize,
             p.KSize,
             to_coopmat_type(p.AType),
             to_coopmat_type(p.ResultType)});
      }
    }
  }
#endif /* VK_KHR_cooperative_matrix */
  return key;
}

const FamilyKey& family_key(const VkPhysicalDevice handle) {
  static std::mutex mutex;
  static std::map<VkPhysicalDevice, FamilyKey> keys;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = keys.find(handle);
  if (it == keys.end()) {
    it = keys.emplace(handle, query_family_key(handle)).first;
  }
  return it->second;
}

} // namespace

Storage to_storage(const utils::StorageType s) {
  switch (s) {
    case utils::kBuffer:
      return Storage::kBuffer;
    case utils::kTexture2D:
      return Storage::kTexture2D;
    case utils::kTexture3D:
      return Storage::kTexture3D;
    default:
      return Storage::kOther;
  }
}

DeviceInfo device_info(ComputeGraph* graph) {
  const auto* adapter = graph->context()->adapter_ptr();
  DeviceInfo d;
  d.name = graph->device_name();
  std::transform(d.name.begin(), d.name.end(), d.name.begin(), [](char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  });
  d.is_amd = graph->device_is_amd();
  d.subgroup_size = adapter->subgroup_size();
  d.min_subgroup_size = adapter->min_subgroup_size();
  d.max_subgroup_size = adapter->max_subgroup_size();
  d.subgroup_size_control = adapter->supports_subgroup_size_control();
  d.coopmat = adapter->supports_cooperative_matrix();
  d.int8_coopmat = adapter->supports_int8_cooperative_matrix();
  d.max_shared_bytes = adapter->max_compute_shared_memory_size();
  const FamilyKey& key = family_key(adapter->physical_handle());
  d.vendor_id = key.vendor_id;
  d.driver_id = key.driver_id;
  d.coopmat_shapes = key.coopmat_shapes;
  return d;
}

utils::StorageType predicted_q4_weight_storage(
    ComputeGraph& graph,
    const ValueRef weight_data) {
  const std::vector<int64_t> sizes = graph.sizes_of(weight_data);
  const int64_t K = utils::val_at(-1, sizes) * 2;
  const int64_t N = utils::val_at(-2, sizes);
  const int64_t height = utils::div_up(N, int64_t(8));
  const int64_t width = utils::div_up(K, int64_t(4)) * 4;
  const uint32_t max_extent =
      graph.context()->adapter_ptr()->max_texture2d_dim();
  return (width > int64_t(max_extent) * 4 || height > int64_t(max_extent))
      ? utils::kBuffer
      : utils::kTexture2D;
}

} // namespace sarc
} // namespace vkcompute
