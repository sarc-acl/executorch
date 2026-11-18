// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/vk_api/Runtime.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/CaseCheckRegistry.h>

#include <cstdio>
#include <string>
#include <vector>

namespace vkcompute::shader_profiling {
namespace {

#ifdef VK_KHR_cooperative_matrix
std::string component_type_name(const VkComponentTypeKHR type) {
  switch (type) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR:
      return "float16";
    case VK_COMPONENT_TYPE_FLOAT32_KHR:
      return "float32";
    case VK_COMPONENT_TYPE_FLOAT64_KHR:
      return "float64";
    case VK_COMPONENT_TYPE_SINT8_KHR:
      return "sint8";
    case VK_COMPONENT_TYPE_SINT16_KHR:
      return "sint16";
    case VK_COMPONENT_TYPE_SINT32_KHR:
      return "sint32";
    case VK_COMPONENT_TYPE_SINT64_KHR:
      return "sint64";
    case VK_COMPONENT_TYPE_UINT8_KHR:
      return "uint8";
    case VK_COMPONENT_TYPE_UINT16_KHR:
      return "uint16";
    case VK_COMPONENT_TYPE_UINT32_KHR:
      return "uint32";
    case VK_COMPONENT_TYPE_UINT64_KHR:
      return "uint64";
    case VK_COMPONENT_TYPE_BFLOAT16_KHR:
      return "bfloat16";
    case VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT:
      return "float8_e4m3";
    case VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT:
      return "float8_e5m2";
    default:
      return "other(" + std::to_string(static_cast<int64_t>(type)) + ")";
  }
}

const char* scope_name(const VkScopeKHR scope) {
  switch (scope) {
    case VK_SCOPE_DEVICE_KHR:
      return "device";
    case VK_SCOPE_WORKGROUP_KHR:
      return "workgroup";
    case VK_SCOPE_SUBGROUP_KHR:
      return "subgroup";
    case VK_SCOPE_QUEUE_FAMILY_KHR:
      return "queue_family";
    default:
      return "other";
  }
}
#endif /* VK_KHR_cooperative_matrix */

std::string version_string(const uint32_t v) {
  return std::to_string(VK_API_VERSION_MAJOR(v)) + "." +
      std::to_string(VK_API_VERSION_MINOR(v)) + "." +
      std::to_string(VK_API_VERSION_PATCH(v));
}

Json enumerated_coopmat_properties(const vkapi::Adapter& adapter) {
  Json list = Json::array();
#ifdef VK_KHR_cooperative_matrix
  if (!adapter.supports_cooperative_matrix()) {
    return list;
  }
  const auto fn =
      reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
          vkGetInstanceProcAddr(
              vkapi::runtime()->instance(),
              "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
  if (fn == nullptr) {
    return list;
  }
  uint32_t count = 0;
  fn(adapter.physical_handle(), &count, nullptr);
  std::vector<VkCooperativeMatrixPropertiesKHR> props(count);
  for (auto& p : props) {
    p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    p.pNext = nullptr;
  }
  fn(adapter.physical_handle(), &count, props.data());
  for (const auto& p : props) {
    list.push_back(
        {{"M", p.MSize},
         {"N", p.NSize},
         {"K", p.KSize},
         {"A", component_type_name(p.AType)},
         {"B", component_type_name(p.BType)},
         {"C", component_type_name(p.CType)},
         {"result", component_type_name(p.ResultType)},
         {"saturating_accumulation", p.saturatingAccumulation == VK_TRUE},
         {"scope", scope_name(p.scope)}});
  }
#endif /* VK_KHR_cooperative_matrix */
  return list;
}

Json check_device_coopmat_properties(const Json& spec, ComputeGraph& graph) {
  (void)spec;
  vkapi::Adapter& adapter = *graph.context()->adapter_ptr();

  VkPhysicalDeviceIDProperties id{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
  VkPhysicalDeviceDriverProperties driver{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
  driver.pNext = &id;
  VkPhysicalDeviceProperties2 props2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  props2.pNext = &driver;
  vkGetPhysicalDeviceProperties2(adapter.physical_handle(), &props2);
  const VkPhysicalDeviceProperties& props = props2.properties;

  char ids[32];
  std::snprintf(
      ids, sizeof(ids), "0x%04x:0x%04x", props.vendorID, props.deviceID);
  char uuid[40];
  const uint8_t* u = id.deviceUUID;
  std::snprintf(
      uuid,
      sizeof(uuid),
      "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
      u[0],
      u[1],
      u[2],
      u[3],
      u[4],
      u[5],
      u[6],
      u[7],
      u[8],
      u[9],
      u[10],
      u[11],
      u[12],
      u[13],
      u[14],
      u[15]);

  const Json coopmat = enumerated_coopmat_properties(adapter);
  Json result = {
      {"status", "PASS"},
      {"device_name", props.deviceName},
      {"vendor_device_id", ids},
      {"device_uuid", uuid},
      {"api_version", version_string(props.apiVersion)},
      {"driver_version_raw", props.driverVersion},
      {"driver_name", driver.driverName},
      {"driver_info", driver.driverInfo},
      {"subgroup_size", adapter.subgroup_size()},
      {"min_subgroup_size", adapter.min_subgroup_size()},
      {"max_subgroup_size", adapter.max_subgroup_size()},
      {"required_subgroup_size_for_compute",
       adapter.supports_required_subgroup_size_for_compute()},
      {"supports_cooperative_matrix", adapter.supports_cooperative_matrix()},
      {"cooperative_matrix_properties", coopmat}};
  if (coopmat.empty()) {
    result["note"] = "no cooperative-matrix properties are enumerated";
  }
  return result;
}

REGISTER_CASE_CHECKS {
  REGISTER_CASE_CHECK(
      device_coopmat_properties, check_device_coopmat_properties);
}

} // namespace
} // namespace vkcompute::shader_profiling
