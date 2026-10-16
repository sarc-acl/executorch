// vk-caps: what the device exposes for the SARC kernels: subgroup sizes, compute limits and every
// VK_KHR_cooperative_matrix shape. Printed as plain text; read-only.
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <cstdio>
#include <string_view>
#include <vector>
static const char* ct(VkComponentTypeKHR t) {
  switch (t) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16"; case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
    case VK_COMPONENT_TYPE_FLOAT64_KHR: return "f64"; case VK_COMPONENT_TYPE_SINT8_KHR: return "s8";
    case VK_COMPONENT_TYPE_SINT16_KHR: return "s16"; case VK_COMPONENT_TYPE_SINT32_KHR: return "s32";
    case VK_COMPONENT_TYPE_SINT64_KHR: return "s64"; case VK_COMPONENT_TYPE_UINT8_KHR: return "u8";
    case VK_COMPONENT_TYPE_UINT16_KHR: return "u16"; case VK_COMPONENT_TYPE_UINT32_KHR: return "u32";
    case VK_COMPONENT_TYPE_UINT64_KHR: return "u64"; default: return "other";
  }
}
int main() {
  void* lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 1; }
  auto get = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib, "vkGetInstanceProcAddr"));
  auto create = reinterpret_cast<PFN_vkCreateInstance>(get(nullptr, "vkCreateInstance"));
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ci.pApplicationInfo = &app;
  VkInstance inst;
  if (!create || create(&ci, nullptr, &inst) != VK_SUCCESS) return 3;
  auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(get(inst, "vkEnumeratePhysicalDevices"));
  auto props2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(get(inst, "vkGetPhysicalDeviceProperties2"));
  auto feats2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(get(inst, "vkGetPhysicalDeviceFeatures2"));
  auto exts = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(get(inst, "vkEnumerateDeviceExtensionProperties"));
  auto coop = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
      get(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
  uint32_t n = 0; enumerate(inst, &n, nullptr); std::vector<VkPhysicalDevice> devs(n); enumerate(inst, &n, devs.data());
  for (auto d : devs) {
    VkPhysicalDeviceSubgroupSizeControlProperties sc{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES}; sg.pNext = &sc;
    VkPhysicalDeviceCooperativeMatrixPropertiesKHR cp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR}; cp.pNext = &sg;
    VkPhysicalDeviceProperties2 p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; p.pNext = &cp;
    props2(d, &p);
    const auto& l = p.properties.limits;
    printf("device %s driver 0x%x api %u.%u.%u\n", p.properties.deviceName, p.properties.driverVersion,
           VK_API_VERSION_MAJOR(p.properties.apiVersion), VK_API_VERSION_MINOR(p.properties.apiVersion), VK_API_VERSION_PATCH(p.properties.apiVersion));
    printf("subgroupSize %u min %u max %u maxComputeWorkgroupSubgroups %u requiredSubgroupSizeStages 0x%x supportedStages 0x%x\n",
           sg.subgroupSize, sc.minSubgroupSize, sc.maxSubgroupSize, sc.maxComputeWorkgroupSubgroups, sc.requiredSubgroupSizeStages, sg.supportedStages);
    printf("maxComputeSharedMemorySize %u maxComputeWorkGroupInvocations %u maxComputeWorkGroupSize %u %u %u maxComputeWorkGroupCount %u %u %u\n",
           l.maxComputeSharedMemorySize, l.maxComputeWorkGroupInvocations, l.maxComputeWorkGroupSize[0], l.maxComputeWorkGroupSize[1],
           l.maxComputeWorkGroupSize[2], l.maxComputeWorkGroupCount[0], l.maxComputeWorkGroupCount[1], l.maxComputeWorkGroupCount[2]);
    printf("maxStorageBufferRange %u maxImageDimension2D %u maxImageDimension3D %u\n", l.maxStorageBufferRange, l.maxImageDimension2D, l.maxImageDimension3D);
    uint32_t ne = 0; exts(d, nullptr, &ne, nullptr); std::vector<VkExtensionProperties> ev(ne); exts(d, nullptr, &ne, ev.data());
    for (auto& e : ev) for (const char* w : {"VK_KHR_cooperative_matrix", "VK_NV_cooperative_matrix", "VK_NV_cooperative_matrix2", "VK_EXT_subgroup_size_control",
                                             "VK_KHR_shader_float16_int8", "VK_KHR_16bit_storage", "VK_KHR_8bit_storage", "VK_KHR_shader_integer_dot_product",
                                             "VK_KHR_pipeline_executable_properties", "VK_KHR_shader_float_controls2"})
      if (std::string_view(e.extensionName) == w) printf("extension %s %u\n", e.extensionName, e.specVersion);
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f.pNext = &cf; feats2(d, &f);
    printf("cooperativeMatrix %u robustBufferAccess %u supportedStages 0x%x\n", cf.cooperativeMatrix, cf.cooperativeMatrixRobustBufferAccess, cp.cooperativeMatrixSupportedStages);
    if (!coop) { printf("no vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR\n"); continue; }
    uint32_t nc = 0; coop(d, &nc, nullptr);
    std::vector<VkCooperativeMatrixPropertiesKHR> cv(nc, VkCooperativeMatrixPropertiesKHR{VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
    coop(d, &nc, cv.data());
    for (auto& c : cv)
      printf("coopmat M %u N %u K %u A %s B %s C %s R %s saturating %u scope %d\n", c.MSize, c.NSize, c.KSize, ct(c.AType), ct(c.BType),
             ct(c.CType), ct(c.ResultType), c.saturatingAccumulation, int(c.scope));
  }
  return 0;
}
