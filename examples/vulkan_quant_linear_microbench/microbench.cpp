// Standalone Vulkan compute microbenchmark: linear_q4gsw_tiled (4w) vs
// linear_dq8ca_q4gsw_tiled (8da4w), the two BASELINE (non-coopmat/non-WMMA)
// Vulkan linear kernels from ExecuTorch's `yanwen/dev-igpu` branch
// (sarc-acl/executorch, commit c02c80254a) -- plus, under --storage=texture,
// each baseline's real KHR_cooperative_matrix/WMMA counterpart:
// linear_q4gsw_coopmat (fp16 WMMA) vs q4gsw tiled, and
// linear_dq8ca_q4gsw_coopmat (int8 WMMA) vs dq8ca tiled. See README.md's
// "coopmat (WMMA)" section for storage caveats and shape alignment
// requirements.
//
// The embedded SPIR-V blobs (shaders/shader_*_spv.h) are glslc output
// from the byte-resolved production GLSL (see shaders/*.glsl in this
// directory) -- resolved via the repo's own
// backends/vulkan/runtime/gen_vulkan_spv.py, NOT hand-transcribed, so the
// dispatched arithmetic is identical to what ExecuTorch itself runs.
//
// Two storage variants are available, selected with --storage=buffer|texture
// (default buffer):
//   buffer  -> linear_*_tiled_buffer_buffer_half   (IO + weight both SSBO)
//   texture -> linear_*_tiled_texture3d_texture2d_half
//              (IO as image3D/sampler3D, weight as isampler2D)
// Note: t_weight_scales/t_weight_sums/t_bias/t_packed_int8_input and the
// dq8ca activation scale/zero-point tensors are ALWAYS buffer/texture3d
// respectively regardless of this flag -- that's hardcoded in the shader
// source, not gated by the storage variant. See the resolved .glsl for the
// exact per-binding types.
//
// Buffer/image contents are a deterministic pseudo-random pattern, not real
// quantized data -- this measures dispatch throughput only, not numerical
// correctness. Both kernels are given the exact same tile config
// (TILE_M4=1, TILE_K4=1, TILE_N8=1 -> 4x8 output tile) and the same
// global/local workgroup-size formula as production (QuantizedLinear.cpp's
// quantized_linear_global_wg_size / pick_hw_square_wg_size), so the two are
// dispatched identically save for the shader body itself.
//
// This harness measures the linear kernel only. Production 8da4w also runs a
// separate activation-quantization pre-pass before this kernel; excluding it
// is intentional (shader-level comparison), but means the ratio here will
// differ from end-to-end tok/s ratios.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "shaders/shader_dq8ca_coopmat_sametile_spv.h"
#include "shaders/shader_dq8ca_coopmat_texture_spv.h"
#include "shaders/shader_dq8ca_coopmat_tuned_spv.h"
#include "shaders/shader_dq8ca_spv.h"
#include "shaders/shader_dq8ca_texture_spv.h"
#include "shaders/shader_q4gsw_coopmat_texture_spv.h"
#include "shaders/shader_q4gsw_coopmat_tuned_spv.h"
#include "shaders/shader_q4gsw_spv.h"
#include "shaders/shader_q4gsw_texture_spv.h"

#define VK_CHECK(x)                          \
  do {                                       \
    VkResult err__ = (x);                    \
    if (err__ != VK_SUCCESS) {               \
      fprintf(                               \
          stderr,                            \
          "Vulkan error %d at %s:%d (%s)\n", \
          (int)err__,                        \
          __FILE__,                          \
          __LINE__,                          \
          #x);                               \
      exit(1);                               \
    }                                        \
  } while (0)

namespace {

uint32_t div_up(uint32_t a, uint32_t b) {
  return (a + b - 1) / b;
}

enum class StorageMode { kBuffer, kTexture };

const char* storage_mode_name(StorageMode m) {
  return m == StorageMode::kBuffer ? "buffer" : "texture";
}

// ---------------------------------------------------------------------------
// Vulkan context
// ---------------------------------------------------------------------------

struct Ctx {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice dev = VK_NULL_HANDLE;
  uint32_t queueFamily = 0;
  VkQueue queue = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkSampler sampler = VK_NULL_HANDLE;
  float timestampPeriodNs = 1.0f;
  bool haveIntegerDotProduct = false;
  bool haveCoopMat = false;
};

VkDebugUtilsMessengerEXT g_messenger = VK_NULL_HANDLE;

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*) {
  if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    fprintf(stderr, "[VALIDATION] %s\n", data->pMessage);
  }
  return VK_FALSE;
}

bool has_layer(const char* name) {
  uint32_t count = 0;
  vkEnumerateInstanceLayerProperties(&count, nullptr);
  std::vector<VkLayerProperties> layers(count);
  vkEnumerateInstanceLayerProperties(&count, layers.data());
  for (auto& l : layers) {
    if (strcmp(l.layerName, name) == 0)
      return true;
  }
  return false;
}

bool has_device_ext(VkPhysicalDevice phys, const char* name) {
  uint32_t count = 0;
  vkEnumerateDeviceExtensionProperties(phys, nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> exts(count);
  vkEnumerateDeviceExtensionProperties(phys, nullptr, &count, exts.data());
  for (auto& e : exts) {
    if (strcmp(e.extensionName, name) == 0)
      return true;
  }
  return false;
}

Ctx create_context(bool validation) {
  Ctx ctx;

  VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  appInfo.pApplicationName = "shader_microbench";
  appInfo.apiVersion = VK_API_VERSION_1_2;

  std::vector<const char*> instLayers;
  std::vector<const char*> instExts;
  if (validation && has_layer("VK_LAYER_KHRONOS_validation")) {
    instLayers.push_back("VK_LAYER_KHRONOS_validation");
    instExts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  VkInstanceCreateInfo instInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instInfo.pApplicationInfo = &appInfo;
  instInfo.enabledLayerCount = (uint32_t)instLayers.size();
  instInfo.ppEnabledLayerNames = instLayers.data();
  instInfo.enabledExtensionCount = (uint32_t)instExts.size();
  instInfo.ppEnabledExtensionNames = instExts.data();
  VK_CHECK(vkCreateInstance(&instInfo, nullptr, &ctx.instance));

  if (!instExts.empty()) {
    auto create_dbg = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        ctx.instance, "vkCreateDebugUtilsMessengerEXT");
    if (create_dbg) {
      VkDebugUtilsMessengerCreateInfoEXT dbgInfo{
          VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
      dbgInfo.messageSeverity =
          VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      dbgInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
          VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
          VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      dbgInfo.pfnUserCallback = debug_callback;
      create_dbg(ctx.instance, &dbgInfo, nullptr, &g_messenger);
    }
  }

  uint32_t physCount = 0;
  vkEnumeratePhysicalDevices(ctx.instance, &physCount, nullptr);
  std::vector<VkPhysicalDevice> physDevices(physCount);
  vkEnumeratePhysicalDevices(ctx.instance, &physCount, physDevices.data());
  if (physCount == 0) {
    fprintf(stderr, "No Vulkan physical devices found.\n");
    exit(1);
  }

  int forceIdx = -1;
  if (const char* e = getenv("MBENCH_DEVICE_INDEX"))
    forceIdx = atoi(e);

  ctx.phys = VK_NULL_HANDLE;
  if (forceIdx >= 0 && (uint32_t)forceIdx < physCount) {
    ctx.phys = physDevices[forceIdx];
  } else {
    for (auto p : physDevices) {
      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(p, &props);
      if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        ctx.phys = p;
        break;
      }
    }
    if (ctx.phys == VK_NULL_HANDLE)
      ctx.phys = physDevices[0];
  }

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(ctx.phys, &props);
  ctx.timestampPeriodNs = props.limits.timestampPeriod;
  fprintf(
      stderr,
      "Device: %s (driverVersion=%u, apiVersion=%u.%u.%u)\n",
      props.deviceName,
      props.driverVersion,
      VK_API_VERSION_MAJOR(props.apiVersion),
      VK_API_VERSION_MINOR(props.apiVersion),
      VK_API_VERSION_PATCH(props.apiVersion));

  uint32_t qCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys, &qCount, nullptr);
  std::vector<VkQueueFamilyProperties> qProps(qCount);
  vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys, &qCount, qProps.data());
  ctx.queueFamily = UINT32_MAX;
  for (uint32_t i = 0; i < qCount; ++i) {
    if (qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      ctx.queueFamily = i;
      break;
    }
  }
  if (ctx.queueFamily == UINT32_MAX) {
    fprintf(stderr, "No compute queue family found.\n");
    exit(1);
  }

  std::vector<const char*> devExts = {
      "VK_KHR_shader_float16_int8",
      "VK_KHR_8bit_storage",
      "VK_KHR_16bit_storage",
      "VK_KHR_shader_integer_dot_product",
      "VK_KHR_cooperative_matrix",
      "VK_EXT_subgroup_size_control"};
  std::vector<const char*> enabledDevExts;
  for (auto e : devExts) {
    if (has_device_ext(ctx.phys, e))
      enabledDevExts.push_back(e);
  }
  auto has_enabled = [&](const char* name) {
    return std::find(
               enabledDevExts.begin(),
               enabledDevExts.end(),
               std::string(name)) != enabledDevExts.end();
  };
  ctx.haveIntegerDotProduct = has_enabled("VK_KHR_shader_integer_dot_product");
  if (!ctx.haveIntegerDotProduct) {
    fprintf(
        stderr,
        "WARNING: VK_KHR_shader_integer_dot_product not reported; "
        "8da4w shader (which requires GL_EXT_integer_dot_product) may "
        "fail to run.\n");
  }
  const bool haveCoopMatExts = has_enabled("VK_KHR_cooperative_matrix") &&
      has_enabled("VK_EXT_subgroup_size_control");

  VkPhysicalDeviceShaderFloat16Int8Features f16i8{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
  VkPhysicalDevice16BitStorageFeatures store16{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
  VkPhysicalDevice8BitStorageFeatures store8{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
  VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR dotProd{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR};
  VkPhysicalDeviceVulkanMemoryModelFeatures memModel{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES};
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMat{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroupSizeCtrl{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT};
  f16i8.pNext = &store16;
  store16.pNext = &store8;
  store8.pNext = &dotProd;
  dotProd.pNext = &memModel;
  memModel.pNext = &coopMat;
  coopMat.pNext = &subgroupSizeCtrl;

  VkPhysicalDeviceFeatures2 feats2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  feats2.pNext = &f16i8;
  vkGetPhysicalDeviceFeatures2(ctx.phys, &feats2);

  ctx.haveCoopMat = haveCoopMatExts && coopMat.cooperativeMatrix &&
      subgroupSizeCtrl.subgroupSizeControl;
  if (!ctx.haveCoopMat) {
    fprintf(
        stderr,
        "NOTE: cooperative matrix / subgroup-size-control not available on "
        "this device; the q4gsw_coopmat (WMMA) kernel will be skipped.\n");
  }

  float qPrio = 1.0f;
  VkDeviceQueueCreateInfo qInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qInfo.queueFamilyIndex = ctx.queueFamily;
  qInfo.queueCount = 1;
  qInfo.pQueuePriorities = &qPrio;

  VkDeviceCreateInfo devInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  devInfo.pNext = &feats2;
  devInfo.queueCreateInfoCount = 1;
  devInfo.pQueueCreateInfos = &qInfo;
  devInfo.enabledExtensionCount = (uint32_t)enabledDevExts.size();
  devInfo.ppEnabledExtensionNames = enabledDevExts.data();
  VK_CHECK(vkCreateDevice(ctx.phys, &devInfo, nullptr, &ctx.dev));

  vkGetDeviceQueue(ctx.dev, ctx.queueFamily, 0, &ctx.queue);

  VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = ctx.queueFamily;
  VK_CHECK(vkCreateCommandPool(ctx.dev, &poolInfo, nullptr, &ctx.pool));

  VkSamplerCreateInfo sampInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sampInfo.magFilter = VK_FILTER_NEAREST;
  sampInfo.minFilter = VK_FILTER_NEAREST;
  sampInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  sampInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  VK_CHECK(vkCreateSampler(ctx.dev, &sampInfo, nullptr, &ctx.sampler));

  return ctx;
}

// ---------------------------------------------------------------------------
// Buffers / images
// ---------------------------------------------------------------------------

uint32_t find_memory_type(
    VkPhysicalDevice phys,
    uint32_t typeBits,
    VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
  for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    if ((typeBits & (1u << i)) &&
        (memProps.memoryTypes[i].propertyFlags & props) == props) {
      return i;
    }
  }
  fprintf(
      stderr,
      "No suitable memory type found (bits=0x%x props=0x%x)\n",
      typeBits,
      props);
  exit(1);
}

struct Buffer {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
};

Buffer alloc_buffer(
    Ctx& ctx,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags memProps) {
  Buffer b;
  b.size = size;
  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = size;
  info.usage = usage;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(ctx.dev, &info, nullptr, &b.buf));

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(ctx.dev, b.buf, &req);
  VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  alloc.allocationSize = req.size;
  alloc.memoryTypeIndex =
      find_memory_type(ctx.phys, req.memoryTypeBits, memProps);
  VK_CHECK(vkAllocateMemory(ctx.dev, &alloc, nullptr, &b.mem));
  VK_CHECK(vkBindBufferMemory(ctx.dev, b.buf, b.mem, 0));
  return b;
}

void free_buffer(Ctx& ctx, Buffer& b) {
  if (b.buf)
    vkDestroyBuffer(ctx.dev, b.buf, nullptr);
  if (b.mem)
    vkFreeMemory(ctx.dev, b.mem, nullptr);
  b = Buffer{};
}

VkCommandBuffer begin_one_shot(Ctx& ctx) {
  VkCommandBufferAllocateInfo allocInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocInfo.commandPool = ctx.pool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  VkCommandBuffer cmd;
  VK_CHECK(vkAllocateCommandBuffers(ctx.dev, &allocInfo, &cmd));
  VkCommandBufferBeginInfo beginInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
  return cmd;
}

void end_one_shot_and_wait(Ctx& ctx, VkCommandBuffer cmd) {
  VK_CHECK(vkEndCommandBuffer(cmd));
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  VK_CHECK(vkQueueSubmit(ctx.queue, 1, &submit, VK_NULL_HANDLE));
  VK_CHECK(vkQueueWaitIdle(ctx.queue));
  vkFreeCommandBuffers(ctx.dev, ctx.pool, 1, &cmd);
}

// Deterministic pseudo-random fill -- content is irrelevant for timing, but
// fixed (not zero) in case any path is data-dependent.
void fill_pattern(std::vector<uint8_t>& data, uint32_t seed) {
  uint32_t x = seed * 2654435761u + 1u;
  for (size_t i = 0; i < data.size(); ++i) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    data[i] = (uint8_t)(x & 0xFF);
  }
}

Buffer create_filled_device_buffer(
    Ctx& ctx,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    uint32_t seed) {
  Buffer dst = alloc_buffer(
      ctx,
      size,
      usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  std::vector<uint8_t> hostData((size_t)size);
  fill_pattern(hostData, seed);

  Buffer staging = alloc_buffer(
      ctx,
      size,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  void* mapped = nullptr;
  VK_CHECK(vkMapMemory(ctx.dev, staging.mem, 0, size, 0, &mapped));
  memcpy(mapped, hostData.data(), (size_t)size);
  vkUnmapMemory(ctx.dev, staging.mem);

  VkCommandBuffer cmd = begin_one_shot(ctx);
  VkBufferCopy copy{0, 0, size};
  vkCmdCopyBuffer(cmd, staging.buf, dst.buf, 1, &copy);
  end_one_shot_and_wait(ctx, cmd);

  free_buffer(ctx, staging);
  return dst;
}

struct Image {
  VkImage img = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
};

Image create_image(
    Ctx& ctx,
    VkImageType type,
    VkExtent3D extent,
    VkFormat format,
    VkImageUsageFlags usage,
    VkImageLayout finalLayout,
    VkAccessFlags finalAccess) {
  Image im;
  VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  info.imageType = type;
  info.format = format;
  info.extent = extent;
  info.mipLevels = 1;
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = usage;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VK_CHECK(vkCreateImage(ctx.dev, &info, nullptr, &im.img));

  VkMemoryRequirements req;
  vkGetImageMemoryRequirements(ctx.dev, im.img, &req);
  VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  alloc.allocationSize = req.size;
  alloc.memoryTypeIndex = find_memory_type(
      ctx.phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK_CHECK(vkAllocateMemory(ctx.dev, &alloc, nullptr, &im.mem));
  VK_CHECK(vkBindImageMemory(ctx.dev, im.img, im.mem, 0));

  VkImageViewType viewType =
      type == VK_IMAGE_TYPE_3D ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
  VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  viewInfo.image = im.img;
  viewInfo.viewType = viewType;
  viewInfo.format = format;
  viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VK_CHECK(vkCreateImageView(ctx.dev, &viewInfo, nullptr, &im.view));

  VkCommandBuffer cmd = begin_one_shot(ctx);
  VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = finalLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = im.img;
  barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  barrier.srcAccessMask = 0;
  barrier.dstAccessMask = finalAccess;
  vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      0,
      nullptr,
      0,
      nullptr,
      1,
      &barrier);
  end_one_shot_and_wait(ctx, cmd);
  return im;
}

Image create_sampled_image3d(
    Ctx& ctx,
    uint32_t width,
    uint32_t height,
    VkFormat format) {
  return create_image(
      ctx,
      VK_IMAGE_TYPE_3D,
      {width, height, 1},
      format,
      VK_IMAGE_USAGE_SAMPLED_BIT,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VK_ACCESS_SHADER_READ_BIT);
}

Image create_sampled_image2d(
    Ctx& ctx,
    uint32_t width,
    uint32_t height,
    VkFormat format) {
  return create_image(
      ctx,
      VK_IMAGE_TYPE_2D,
      {width, height, 1},
      format,
      VK_IMAGE_USAGE_SAMPLED_BIT,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VK_ACCESS_SHADER_READ_BIT);
}

Image create_storage_image3d(
    Ctx& ctx,
    uint32_t width,
    uint32_t height,
    VkFormat format) {
  return create_image(
      ctx,
      VK_IMAGE_TYPE_3D,
      {width, height, 1},
      format,
      VK_IMAGE_USAGE_STORAGE_BIT,
      VK_IMAGE_LAYOUT_GENERAL,
      VK_ACCESS_SHADER_WRITE_BIT);
}

void free_image(Ctx& ctx, Image& im) {
  if (im.view)
    vkDestroyImageView(ctx.dev, im.view, nullptr);
  if (im.img)
    vkDestroyImage(ctx.dev, im.img, nullptr);
  if (im.mem)
    vkFreeMemory(ctx.dev, im.mem, nullptr);
  im = Image{};
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

struct Pipeline {
  VkShaderModule module = VK_NULL_HANDLE;
  VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
  VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkDescriptorPool descPool = VK_NULL_HANDLE;
  VkDescriptorSet descSet = VK_NULL_HANDLE;
};

Pipeline build_pipeline(
    Ctx& ctx,
    const uint32_t* spv,
    size_t spvBytes,
    const std::vector<VkDescriptorType>& bindingTypes,
    const std::vector<int32_t>& specConsts,
    const void* stagePNext = nullptr) {
  Pipeline p;

  VkShaderModuleCreateInfo modInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  modInfo.codeSize = spvBytes;
  modInfo.pCode = spv;
  VK_CHECK(vkCreateShaderModule(ctx.dev, &modInfo, nullptr, &p.module));

  std::vector<VkDescriptorSetLayoutBinding> bindings(bindingTypes.size());
  for (size_t i = 0; i < bindingTypes.size(); ++i) {
    bindings[i] = {
        (uint32_t)i, bindingTypes[i], 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  }
  VkDescriptorSetLayoutCreateInfo layoutInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layoutInfo.bindingCount = (uint32_t)bindings.size();
  layoutInfo.pBindings = bindings.data();
  VK_CHECK(
      vkCreateDescriptorSetLayout(ctx.dev, &layoutInfo, nullptr, &p.setLayout));

  VkPipelineLayoutCreateInfo pipeLayoutInfo{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipeLayoutInfo.setLayoutCount = 1;
  pipeLayoutInfo.pSetLayouts = &p.setLayout;
  VK_CHECK(
      vkCreatePipelineLayout(ctx.dev, &pipeLayoutInfo, nullptr, &p.pipeLayout));

  std::vector<VkSpecializationMapEntry> mapEntries(specConsts.size());
  for (uint32_t i = 0; i < specConsts.size(); ++i) {
    mapEntries[i] = {i, i * 4, 4};
  }
  VkSpecializationInfo specInfo{};
  specInfo.mapEntryCount = (uint32_t)mapEntries.size();
  specInfo.pMapEntries = mapEntries.data();
  specInfo.dataSize = specConsts.size() * sizeof(int32_t);
  specInfo.pData = specConsts.data();

  VkPipelineShaderStageCreateInfo stageInfo{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stageInfo.pNext = stagePNext;
  stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stageInfo.module = p.module;
  stageInfo.pName = "main";
  stageInfo.pSpecializationInfo = &specInfo;

  VkComputePipelineCreateInfo pipeInfo{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeInfo.stage = stageInfo;
  pipeInfo.layout = p.pipeLayout;
  VK_CHECK(vkCreateComputePipelines(
      ctx.dev, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &p.pipeline));

  std::vector<VkDescriptorPoolSize> poolSizes;
  uint32_t bufCount = 0, imgCount = 0, uboCount = 0, storageImgCount = 0;
  for (auto t : bindingTypes) {
    if (t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      bufCount++;
    else if (t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
      uboCount++;
    else if (t == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
      imgCount++;
    else if (t == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
      storageImgCount++;
  }
  if (bufCount)
    poolSizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bufCount});
  if (uboCount)
    poolSizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uboCount});
  if (imgCount)
    poolSizes.push_back({VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imgCount});
  if (storageImgCount)
    poolSizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storageImgCount});

  VkDescriptorPoolCreateInfo poolInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolInfo.maxSets = 1;
  poolInfo.poolSizeCount = (uint32_t)poolSizes.size();
  poolInfo.pPoolSizes = poolSizes.data();
  VK_CHECK(vkCreateDescriptorPool(ctx.dev, &poolInfo, nullptr, &p.descPool));

  VkDescriptorSetAllocateInfo dsAlloc{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsAlloc.descriptorPool = p.descPool;
  dsAlloc.descriptorSetCount = 1;
  dsAlloc.pSetLayouts = &p.setLayout;
  VK_CHECK(vkAllocateDescriptorSets(ctx.dev, &dsAlloc, &p.descSet));

  return p;
}

void destroy_pipeline(Ctx& ctx, Pipeline& p) {
  if (p.descPool)
    vkDestroyDescriptorPool(ctx.dev, p.descPool, nullptr);
  if (p.pipeline)
    vkDestroyPipeline(ctx.dev, p.pipeline, nullptr);
  if (p.pipeLayout)
    vkDestroyPipelineLayout(ctx.dev, p.pipeLayout, nullptr);
  if (p.setLayout)
    vkDestroyDescriptorSetLayout(ctx.dev, p.setLayout, nullptr);
  if (p.module)
    vkDestroyShaderModule(ctx.dev, p.module, nullptr);
  p = Pipeline{};
}

void bind_buffer(
    Ctx& ctx,
    Pipeline& p,
    uint32_t binding,
    VkDescriptorType type,
    VkBuffer buf,
    VkDeviceSize size) {
  VkDescriptorBufferInfo info{buf, 0, size};
  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = p.descSet;
  write.dstBinding = binding;
  write.descriptorCount = 1;
  write.descriptorType = type;
  write.pBufferInfo = &info;
  vkUpdateDescriptorSets(ctx.dev, 1, &write, 0, nullptr);
}

void bind_sampled_image(
    Ctx& ctx,
    Pipeline& p,
    uint32_t binding,
    VkImageView view) {
  VkDescriptorImageInfo info{
      ctx.sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = p.descSet;
  write.dstBinding = binding;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  write.pImageInfo = &info;
  vkUpdateDescriptorSets(ctx.dev, 1, &write, 0, nullptr);
}

void bind_storage_image(
    Ctx& ctx,
    Pipeline& p,
    uint32_t binding,
    VkImageView view) {
  VkDescriptorImageInfo info{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = p.descSet;
  write.dstBinding = binding;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  write.pImageInfo = &info;
  vkUpdateDescriptorSets(ctx.dev, 1, &write, 0, nullptr);
}

// ---------------------------------------------------------------------------
// Workgroup sizing -- mirrors QuantizedLinear.cpp's
// quantized_linear_global_wg_size / pick_hw_square_wg_size exactly, so both
// kernels are dispatched with production's own formula.
// ---------------------------------------------------------------------------

struct WgConfig {
  uint32_t numNTiles, numMTiles; // "global_wg_size" in ET's terminology
  uint32_t localX, localY, localZ;
  uint32_t groupCountX, groupCountY, groupCountZ;
};

WgConfig compute_wg_config(uint32_t M, uint32_t N) {
  WgConfig c{};
  c.numNTiles = div_up(N, 8); // N_per_tile = 8 for q4 kernels
  c.numMTiles = div_up(M, 4); // M_per_tile = 4 (non-gemv, non-coop)
  if (c.numNTiles >= 6 && c.numMTiles >= 6) {
    c.localX = 8;
    c.localY = 8;
  } else if (c.numNTiles < 6) {
    c.localX = 4;
    c.localY = 16;
  } else {
    c.localX = 16;
    c.localY = 4;
  }
  c.localZ = 1;
  c.groupCountX = div_up(c.numNTiles, c.localX);
  c.groupCountY = div_up(c.numMTiles, c.localY);
  c.groupCountZ = 1;
  return c;
}

// ---------------------------------------------------------------------------
// Timed dispatch batch
// ---------------------------------------------------------------------------

double run_timed_batch(
    Ctx& ctx,
    VkQueryPool queryPool,
    VkPipeline pipeline,
    VkPipelineLayout pipeLayout,
    VkDescriptorSet descSet,
    const WgConfig& wg,
    uint32_t iters) {
  VkCommandBuffer cmd = begin_one_shot(ctx);
  vkCmdResetQueryPool(cmd, queryPool, 0, 2);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(
      cmd,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipeLayout,
      0,
      1,
      &descSet,
      0,
      nullptr);
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
  for (uint32_t i = 0; i < iters; ++i) {
    vkCmdDispatch(cmd, wg.groupCountX, wg.groupCountY, wg.groupCountZ);
  }
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
  end_one_shot_and_wait(ctx, cmd);

  uint64_t ts[2];
  VK_CHECK(vkGetQueryPoolResults(
      ctx.dev,
      queryPool,
      0,
      2,
      sizeof(ts),
      ts,
      sizeof(uint64_t),
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
  double totalNs = (double)(ts[1] - ts[0]) * ctx.timestampPeriodNs;
  return totalNs / iters; // ns per dispatch
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

double cov(const std::vector<double>& v) {
  double mean = 0;
  for (double x : v)
    mean += x;
  mean /= v.size();
  double var = 0;
  for (double x : v)
    var += (x - mean) * (x - mean);
  var /= v.size();
  return mean > 0 ? std::sqrt(var) / mean : 0.0;
}

// ---------------------------------------------------------------------------
// Shape definitions and per-kernel resource setup
// ---------------------------------------------------------------------------

struct Shape {
  const char* label;
  uint32_t M, K, N, groupSize;
};

const std::vector<Shape> kShapes = {
    {"1B_qkvo", 2048, 2048, 2048, 128},
    {"1B_mlp_up", 2048, 2048, 8192, 128},
    {"1B_mlp_down", 2048, 8192, 2048, 128},
    {"3B_qkvo", 2048, 3072, 3072, 128},
    {"3B_mlp_up", 2048, 3072, 8192, 128},
    {"3B_mlp_down", 2048, 8192, 3072, 128},
    {"8B_qkvo", 2048, 4096, 4096, 128},
    // 8B_mlp_up (2048,4096,14336) and 8B_mlp_down (2048,14336,4096) skipped:
    // same total weight/FLOPs (just transposed K/N), both crash the M51
    // secondary board (xgpusw-debug08) into the LK bootloader every time,
    // reproducibly, in both storage=buffer and storage=texture, at both
    // floating and max-pinned clocks -- local-only workaround, not pushed.
    // M-sweep on the same down_proj shape to see how the ratio moves with
    // sequence length / prefill batch size; M2048 (== full 8B_mlp_down)
    // skipped for the same reason, M32/M128/M512 kept (4-64x less work).
    {"8B_mlp_down_M32", 32, 14336, 4096, 128},
    {"8B_mlp_down_M128", 128, 14336, 4096, 128},
    {"8B_mlp_down_M512", 512, 14336, 4096, 128},
};

struct KernelHandle {
  Pipeline pipe;
  WgConfig wg;
  std::vector<Buffer> ownedBuffers;
  std::vector<Image> ownedImages;
};

KernelHandle setup_q4gsw(Ctx& ctx, const Shape& s, StorageMode mode) {
  const uint32_t N4 = div_up(s.N, 4), N8 = div_up(s.N, 8);
  const uint32_t K4 = div_up(s.K, 4);
  const uint32_t M4 = div_up(s.M, 4);
  const uint32_t K4PerGroup = div_up(s.groupSize, 4);
  const uint32_t numGroups = K4 / K4PerGroup;

  KernelHandle h;
  h.wg = compute_wg_config(s.M, s.N);

  const bool tex = (mode == StorageMode::kTexture);
  std::vector<VkDescriptorType> bindingTypes = {
      tex ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
          : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      tex ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
          : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      tex ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
          : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // t_weight_scales -- always buffer
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // t_bias -- always buffer
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};

  h.pipe = build_pipeline(
      ctx,
      tex ? shader_q4gsw_texture_spv : shader_q4gsw_spv,
      tex ? shader_q4gsw_texture_spv_size : shader_q4gsw_spv_size,
      bindingTypes,
      {(int32_t)h.wg.localX,
       (int32_t)h.wg.localY,
       (int32_t)h.wg.localZ,
       /*applyBias=*/0,
       (int32_t)K4PerGroup});

  VkDeviceSize scalesBytes = (VkDeviceSize)numGroups * N4 * 8;
  VkDeviceSize biasBytes = (VkDeviceSize)N4 * 8;
  Buffer scalesBuf = create_filled_device_buffer(
      ctx, scalesBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 3);
  Buffer biasBuf = create_filled_device_buffer(
      ctx, biasBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 4);
  bind_buffer(
      ctx,
      h.pipe,
      3,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      scalesBuf.buf,
      scalesBytes);
  bind_buffer(
      ctx,
      h.pipe,
      4,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      biasBuf.buf,
      biasBytes);
  h.ownedBuffers = {scalesBuf, biasBuf};

  if (!tex) {
    VkDeviceSize outputBytes = (VkDeviceSize)s.M * N4 * 8;
    VkDeviceSize inputBytes = (VkDeviceSize)M4 * K4 * 8;
    VkDeviceSize weightBytes = (VkDeviceSize)N8 * K4 * 16;
    Buffer outputBuf = alloc_buffer(
        ctx,
        outputBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buffer inputBuf = create_filled_device_buffer(
        ctx, inputBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1);
    Buffer weightBuf = create_filled_device_buffer(
        ctx, weightBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 2);
    bind_buffer(
        ctx,
        h.pipe,
        0,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        outputBuf.buf,
        outputBytes);
    bind_buffer(
        ctx,
        h.pipe,
        1,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        inputBuf.buf,
        inputBytes);
    bind_buffer(
        ctx,
        h.pipe,
        2,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        weightBuf.buf,
        weightBytes);
    h.ownedBuffers.push_back(outputBuf);
    h.ownedBuffers.push_back(inputBuf);
    h.ownedBuffers.push_back(weightBuf);
  } else {
    Image outputImg =
        create_storage_image3d(ctx, N4, s.M, VK_FORMAT_R16G16B16A16_SFLOAT);
    Image inputImg =
        create_sampled_image3d(ctx, K4, M4, VK_FORMAT_R16G16B16A16_SFLOAT);
    Image weightImg =
        create_sampled_image2d(ctx, K4, N8, VK_FORMAT_R32G32B32A32_SINT);
    bind_storage_image(ctx, h.pipe, 0, outputImg.view);
    bind_sampled_image(ctx, h.pipe, 1, inputImg.view);
    bind_sampled_image(ctx, h.pipe, 2, weightImg.view);
    h.ownedImages = {outputImg, inputImg, weightImg};
  }

  int32_t sizesData[2][4] = {
      {(int32_t)s.N, (int32_t)s.M, 1, 1}, {(int32_t)s.K, (int32_t)s.M, 1, 1}};
  Buffer outputSizesUbo = alloc_buffer(
      ctx,
      sizeof(sizesData[0]),
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  Buffer inputSizesUbo = alloc_buffer(
      ctx,
      sizeof(sizesData[1]),
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  {
    void* p;
    vkMapMemory(ctx.dev, outputSizesUbo.mem, 0, sizeof(sizesData[0]), 0, &p);
    memcpy(p, sizesData[0], sizeof(sizesData[0]));
    vkUnmapMemory(ctx.dev, outputSizesUbo.mem);
    vkMapMemory(ctx.dev, inputSizesUbo.mem, 0, sizeof(sizesData[1]), 0, &p);
    memcpy(p, sizesData[1], sizeof(sizesData[1]));
    vkUnmapMemory(ctx.dev, inputSizesUbo.mem);
  }
  bind_buffer(
      ctx,
      h.pipe,
      5,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      outputSizesUbo.buf,
      sizeof(sizesData[0]));
  bind_buffer(
      ctx,
      h.pipe,
      6,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      inputSizesUbo.buf,
      sizeof(sizesData[1]));
  h.ownedBuffers.push_back(outputSizesUbo);
  h.ownedBuffers.push_back(inputSizesUbo);

  return h;
}

KernelHandle setup_dq8ca(Ctx& ctx, const Shape& s, StorageMode mode) {
  const uint32_t N4 = div_up(s.N, 4), N8 = div_up(s.N, 8);
  const uint32_t K4 = div_up(s.K, 4);
  const uint32_t M4 = div_up(s.M, 4);
  const uint32_t K4PerGroup = div_up(s.groupSize, 4);
  const uint32_t numGroups = K4 / K4PerGroup;

  KernelHandle h;
  h.wg = compute_wg_config(s.M, s.N);

  const bool tex = (mode == StorageMode::kTexture);
  std::vector<VkDescriptorType> bindingTypes = {
      tex ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
          : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 0 t_output
      tex ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
          : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 1 t_input (unused)
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 2 t_packed_int8_input -- always
                                         // buffer
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 3 t_int8_input_sums -- always buffer
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, // 4 t_int8_input_scales --
                                                 // always texture3d
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, // 5 t_int8_input_zps -- always
                                                 // texture3d
      tex ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
          : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 6 t_packed_int4_weight
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 7 t_weight_sums -- always buffer
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 8 t_weight_scales -- always buffer
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 9 t_bias -- always buffer
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};

  h.pipe = build_pipeline(
      ctx,
      tex ? shader_dq8ca_texture_spv : shader_dq8ca_spv,
      tex ? shader_dq8ca_texture_spv_size : shader_dq8ca_spv_size,
      bindingTypes,
      {(int32_t)h.wg.localX,
       (int32_t)h.wg.localY,
       (int32_t)h.wg.localZ,
       /*applyBias=*/0,
       (int32_t)K4PerGroup});

  VkDeviceSize packedInt8InputBytes = (VkDeviceSize)M4 * K4 * 16;
  VkDeviceSize int8InputSumsBytes = (VkDeviceSize)numGroups * M4 * 16;
  VkDeviceSize weightSumsBytes = (VkDeviceSize)numGroups * N4 * 16;
  VkDeviceSize scalesBytes = (VkDeviceSize)numGroups * N4 * 8;
  VkDeviceSize biasBytes = (VkDeviceSize)N4 * 8;

  Buffer packedInt8InputBuf = create_filled_device_buffer(
      ctx, packedInt8InputBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 2);
  Buffer int8InputSumsBuf = create_filled_device_buffer(
      ctx, int8InputSumsBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 3);
  Buffer weightSumsBuf = create_filled_device_buffer(
      ctx, weightSumsBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 5);
  Buffer scalesBuf = create_filled_device_buffer(
      ctx, scalesBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 6);
  Buffer biasBuf = create_filled_device_buffer(
      ctx, biasBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 7);

  bind_buffer(
      ctx,
      h.pipe,
      2,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      packedInt8InputBuf.buf,
      packedInt8InputBytes);
  bind_buffer(
      ctx,
      h.pipe,
      3,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      int8InputSumsBuf.buf,
      int8InputSumsBytes);
  bind_buffer(
      ctx,
      h.pipe,
      7,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      weightSumsBuf.buf,
      weightSumsBytes);
  bind_buffer(
      ctx,
      h.pipe,
      8,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      scalesBuf.buf,
      scalesBytes);
  bind_buffer(
      ctx,
      h.pipe,
      9,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      biasBuf.buf,
      biasBytes);
  h.ownedBuffers = {
      packedInt8InputBuf, int8InputSumsBuf, weightSumsBuf, scalesBuf, biasBuf};

  // t_int8_input_scales / t_int8_input_zps -- always texture3d, regardless of
  // the buffer/texture storage flag (hardcoded in the shader source).
  Image scalesImg =
      create_sampled_image3d(ctx, M4, 1, VK_FORMAT_R16G16B16A16_SFLOAT);
  Image zpsImg = create_sampled_image3d(ctx, M4, 1, VK_FORMAT_R8G8B8A8_SINT);
  bind_sampled_image(ctx, h.pipe, 4, scalesImg.view);
  bind_sampled_image(ctx, h.pipe, 5, zpsImg.view);
  h.ownedImages = {scalesImg, zpsImg};

  if (!tex) {
    VkDeviceSize outputBytes = (VkDeviceSize)s.M * N4 * 8;
    VkDeviceSize inputBytes = 16; // t_input: bound but unread by this shader
    VkDeviceSize weightBytes = (VkDeviceSize)N8 * K4 * 16;
    Buffer outputBuf = alloc_buffer(
        ctx,
        outputBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buffer inputBuf = create_filled_device_buffer(
        ctx, inputBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1);
    Buffer weightBuf = create_filled_device_buffer(
        ctx, weightBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 4);
    bind_buffer(
        ctx,
        h.pipe,
        0,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        outputBuf.buf,
        outputBytes);
    bind_buffer(
        ctx,
        h.pipe,
        1,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        inputBuf.buf,
        inputBytes);
    bind_buffer(
        ctx,
        h.pipe,
        6,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        weightBuf.buf,
        weightBytes);
    h.ownedBuffers.push_back(outputBuf);
    h.ownedBuffers.push_back(inputBuf);
    h.ownedBuffers.push_back(weightBuf);
  } else {
    Image outputImg =
        create_storage_image3d(ctx, N4, s.M, VK_FORMAT_R16G16B16A16_SFLOAT);
    Image inputImg = create_sampled_image3d(
        ctx, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT); // unused
    Image weightImg =
        create_sampled_image2d(ctx, K4, N8, VK_FORMAT_R32G32B32A32_SINT);
    bind_storage_image(ctx, h.pipe, 0, outputImg.view);
    bind_sampled_image(ctx, h.pipe, 1, inputImg.view);
    bind_sampled_image(ctx, h.pipe, 6, weightImg.view);
    h.ownedImages.push_back(outputImg);
    h.ownedImages.push_back(inputImg);
    h.ownedImages.push_back(weightImg);
  }

  int32_t sizesData[2][4] = {
      {(int32_t)s.N, (int32_t)s.M, 1, 1}, {(int32_t)s.K, (int32_t)s.M, 1, 1}};
  Buffer outputSizesUbo = alloc_buffer(
      ctx,
      sizeof(sizesData[0]),
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  Buffer inputSizesUbo = alloc_buffer(
      ctx,
      sizeof(sizesData[1]),
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  {
    void* p;
    vkMapMemory(ctx.dev, outputSizesUbo.mem, 0, sizeof(sizesData[0]), 0, &p);
    memcpy(p, sizesData[0], sizeof(sizesData[0]));
    vkUnmapMemory(ctx.dev, outputSizesUbo.mem);
    vkMapMemory(ctx.dev, inputSizesUbo.mem, 0, sizeof(sizesData[1]), 0, &p);
    memcpy(p, sizesData[1], sizeof(sizesData[1]));
    vkUnmapMemory(ctx.dev, inputSizesUbo.mem);
  }
  bind_buffer(
      ctx,
      h.pipe,
      10,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      outputSizesUbo.buf,
      sizeof(sizesData[0]));
  bind_buffer(
      ctx,
      h.pipe,
      11,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      inputSizesUbo.buf,
      sizeof(sizesData[1]));
  h.ownedBuffers.push_back(outputSizesUbo);
  h.ownedBuffers.push_back(inputSizesUbo);

  return h;
}

// ---------------------------------------------------------------------------
// coopmat (WMMA) tile variants.
//
// Cooperative-matrix shaders dispatch/bind very differently from the tiled
// kernels above: activation and output are forced to buffer storage
// (coopMatLoad/Store require linear memory, not images), only the weight
// tensor has a texture2d option -- "storage=texture" here means the weight,
// same as the tiled kernels' always-texture2d weight. Tile geometry is NOT
// gated inside the shader (misaligned shapes silently miscompute), so the
// harness checks M/N/K alignment before dispatching. WG_SIZE (= SG_GRID_X *
// SG_GRID_Y * SUBGROUP_SIZE) is a fixed 128 threads/workgroup across every
// variant below -- a QuantizedLinear.cpp dispatch-thread-count constraint,
// not a coincidence.
//
// Two variants per kernel: "shipped" is production's current default
// (tuned on a Samsung Xclipse 970, per each shader's own header comment);
// "780M-tuned" is dev-igpu's specs/035 e2e sweep result, done specifically
// on this GPU family (findings in specs/035-dev-igpu-tile-sweep/findings.md
// on the yanwen/dev-igpu branch) -- a fair, locally-tuned comparison point
// instead of measuring against a config tuned for different hardware.
// ---------------------------------------------------------------------------

struct CoopmatVariant {
  const char* label;
  uint32_t tileM, tileN, tileK, subgroupSize;
  const uint32_t* spv;
  size_t spvSize;
};

constexpr uint32_t kCoopmatWgSize = 128;

bool coopmat_tile_aligned(const Shape& s, const CoopmatVariant& v) {
  return s.M % v.tileM == 0 && s.N % v.tileN == 0 && s.K % v.tileK == 0;
}

KernelHandle
setup_q4gsw_coopmat(Ctx& ctx, const Shape& s, const CoopmatVariant& variant) {
  const uint32_t N4 = div_up(s.N, 4), N8 = div_up(s.N, 8);
  const uint32_t K4 = div_up(s.K, 4);
  const uint32_t K4PerGroup = div_up(s.groupSize, 4);
  const uint32_t numGroups = K4 / K4PerGroup;

  KernelHandle h;
  h.wg.localX = kCoopmatWgSize;
  h.wg.localY = 1;
  h.wg.localZ = 1;
  h.wg.groupCountX = div_up(s.N, variant.tileN);
  h.wg.groupCountY = div_up(s.M, variant.tileM);
  h.wg.groupCountZ = 1;

  std::vector<VkDescriptorType> bindingTypes = {
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 0 t_output
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 1 t_input
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, // 2 t_packed_weight
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 3 t_weight_scales
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 4 t_bias -- bound but unread
                                         // (apply_bias=0 -> HAS_BIAS unset)
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // 5 output_sizes
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // 6 input_sizes
  };

  VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroupSizeInfo{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT};
  subgroupSizeInfo.requiredSubgroupSize = variant.subgroupSize;

  h.pipe = build_pipeline(
      ctx,
      variant.spv,
      variant.spvSize,
      bindingTypes,
      {(int32_t)kCoopmatWgSize,
       1,
       1,
       /*apply_bias=*/0,
       (int32_t)K4PerGroup,
       (int32_t)numGroups,
       (int32_t)s.N},
      &subgroupSizeInfo);

  VkDeviceSize outputBytes = (VkDeviceSize)s.M * s.N * 2; // float16_t scalar
  VkDeviceSize inputBytes = (VkDeviceSize)s.M * K4 * 8; // f16vec4
  VkDeviceSize scalesBytes = (VkDeviceSize)numGroups * N4 * 8; // f16vec4
  VkDeviceSize biasBytes = (VkDeviceSize)s.N * 2; // float16_t scalar, unread

  Buffer outputBuf = alloc_buffer(
      ctx,
      outputBytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  Buffer inputBuf = create_filled_device_buffer(
      ctx, inputBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1);
  Buffer scalesBuf = create_filled_device_buffer(
      ctx, scalesBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 3);
  Buffer biasBuf = create_filled_device_buffer(
      ctx, biasBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 4);

  bind_buffer(
      ctx,
      h.pipe,
      0,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      outputBuf.buf,
      outputBytes);
  bind_buffer(
      ctx,
      h.pipe,
      1,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      inputBuf.buf,
      inputBytes);
  bind_buffer(
      ctx,
      h.pipe,
      3,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      scalesBuf.buf,
      scalesBytes);
  bind_buffer(
      ctx,
      h.pipe,
      4,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      biasBuf.buf,
      biasBytes);
  h.ownedBuffers = {outputBuf, inputBuf, scalesBuf, biasBuf};

  // Weight block layout (K4 x N8 blocks) is identical to the tiled kernels'
  // t_packed_int4_weight -- same image dims/format.
  Image weightImg =
      create_sampled_image2d(ctx, K4, N8, VK_FORMAT_R32G32B32A32_SINT);
  bind_sampled_image(ctx, h.pipe, 2, weightImg.view);
  h.ownedImages = {weightImg};

  int32_t sizesData[2][4] = {
      {(int32_t)s.N, (int32_t)s.M, 1, 1}, {(int32_t)s.K, (int32_t)s.M, 1, 1}};
  Buffer outputSizesUbo = alloc_buffer(
      ctx,
      sizeof(sizesData[0]),
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  Buffer inputSizesUbo = alloc_buffer(
      ctx,
      sizeof(sizesData[1]),
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  {
    void* p;
    vkMapMemory(ctx.dev, outputSizesUbo.mem, 0, sizeof(sizesData[0]), 0, &p);
    memcpy(p, sizesData[0], sizeof(sizesData[0]));
    vkUnmapMemory(ctx.dev, outputSizesUbo.mem);
    vkMapMemory(ctx.dev, inputSizesUbo.mem, 0, sizeof(sizesData[1]), 0, &p);
    memcpy(p, sizesData[1], sizeof(sizesData[1]));
    vkUnmapMemory(ctx.dev, inputSizesUbo.mem);
  }
  bind_buffer(
      ctx,
      h.pipe,
      5,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      outputSizesUbo.buf,
      sizeof(sizesData[0]));
  bind_buffer(
      ctx,
      h.pipe,
      6,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      inputSizesUbo.buf,
      sizeof(sizesData[1]));
  h.ownedBuffers.push_back(outputSizesUbo);
  h.ownedBuffers.push_back(inputSizesUbo);

  return h;
}

// ---------------------------------------------------------------------------
// dq8ca_q4gsw coopmat (WMMA, int8).
//
// Same binding order/types as setup_dq8ca's tiled kernel (see
// add_linear_dqa_qw_node in QuantizedLinear.cpp), so buffer-size formulas are
// reused as-is even though a few tensors (t_weight_sums, t_int8_input_sums)
// are scalar int[] here vs ivec4[] in the tiled shader -- the tiled vec4
// formulas are a safe superset since N % 4 == 0 / M % 4 == 0 for every shape
// that reaches this kernel. Runs matrix multiply in INT8 (coopmat<int8> x
// coopmat<int8> -> coopmat<int32>, dequantized once per group).
// ---------------------------------------------------------------------------

KernelHandle
setup_dq8ca_coopmat(Ctx& ctx, const Shape& s, const CoopmatVariant& variant) {
  const uint32_t N4 = div_up(s.N, 4), N8 = div_up(s.N, 8);
  const uint32_t K4 = div_up(s.K, 4);
  const uint32_t M4 = div_up(s.M, 4);
  const uint32_t K4PerGroup = div_up(s.groupSize, 4);
  const uint32_t numGroups = K4 / K4PerGroup;

  KernelHandle h;
  h.wg.localX = kCoopmatWgSize;
  h.wg.localY = 1;
  h.wg.localZ = 1;
  h.wg.groupCountX = div_up(s.N, variant.tileN);
  h.wg.groupCountY = div_up(s.M, variant.tileM);
  h.wg.groupCountZ = 1;

  std::vector<VkDescriptorType> bindingTypes = {
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 0 t_output
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 1 t_input -- unused
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 2 t_packed_int8_input
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 3 t_int8_input_sums
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, // 4 t_int8_input_scales
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, // 5 t_int8_input_zps
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, // 6 t_packed_weight
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 7 t_weight_sums
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 8 t_weight_scales
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 9 t_bias -- bound but unread
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // 10 output_sizes
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // 11 input_sizes
  };

  VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroupSizeInfo{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT};
  subgroupSizeInfo.requiredSubgroupSize = variant.subgroupSize;

  h.pipe = build_pipeline(
      ctx,
      variant.spv,
      variant.spvSize,
      bindingTypes,
      {(int32_t)kCoopmatWgSize,
       1,
       1,
       /*apply_bias=*/0,
       (int32_t)K4PerGroup,
       (int32_t)numGroups,
       (int32_t)s.N},
      &subgroupSizeInfo);

  VkDeviceSize outputBytes = (VkDeviceSize)s.M * s.N * 2; // float16_t scalar
  VkDeviceSize inputBytes = 16; // t_input: bound but unread
  VkDeviceSize packedInt8InputBytes = (VkDeviceSize)M4 * K4 * 16;
  VkDeviceSize int8InputSumsBytes = (VkDeviceSize)numGroups * M4 * 16;
  VkDeviceSize weightSumsBytes = (VkDeviceSize)numGroups * N4 * 16;
  VkDeviceSize scalesBytes = (VkDeviceSize)numGroups * N4 * 8;
  VkDeviceSize biasBytes = (VkDeviceSize)s.N * 2; // float16_t scalar, unread

  Buffer outputBuf = alloc_buffer(
      ctx,
      outputBytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  Buffer inputBuf = create_filled_device_buffer(
      ctx, inputBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1);
  Buffer packedInt8InputBuf = create_filled_device_buffer(
      ctx, packedInt8InputBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 2);
  Buffer int8InputSumsBuf = create_filled_device_buffer(
      ctx, int8InputSumsBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 3);
  Buffer weightSumsBuf = create_filled_device_buffer(
      ctx, weightSumsBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 5);
  Buffer scalesBuf = create_filled_device_buffer(
      ctx, scalesBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 6);
  Buffer biasBuf = create_filled_device_buffer(
      ctx, biasBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 7);

  bind_buffer(
      ctx,
      h.pipe,
      0,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      outputBuf.buf,
      outputBytes);
  bind_buffer(
      ctx,
      h.pipe,
      1,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      inputBuf.buf,
      inputBytes);
  bind_buffer(
      ctx,
      h.pipe,
      2,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      packedInt8InputBuf.buf,
      packedInt8InputBytes);
  bind_buffer(
      ctx,
      h.pipe,
      3,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      int8InputSumsBuf.buf,
      int8InputSumsBytes);
  bind_buffer(
      ctx,
      h.pipe,
      7,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      weightSumsBuf.buf,
      weightSumsBytes);
  bind_buffer(
      ctx,
      h.pipe,
      8,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      scalesBuf.buf,
      scalesBytes);
  bind_buffer(
      ctx,
      h.pipe,
      9,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      biasBuf.buf,
      biasBytes);
  h.ownedBuffers = {
      outputBuf,
      inputBuf,
      packedInt8InputBuf,
      int8InputSumsBuf,
      weightSumsBuf,
      scalesBuf,
      biasBuf};

  // t_int8_input_scales / t_int8_input_zps -- always texture3d, same as the
  // tiled kernel.
  Image scalesImg =
      create_sampled_image3d(ctx, M4, 1, VK_FORMAT_R16G16B16A16_SFLOAT);
  Image zpsImg = create_sampled_image3d(ctx, M4, 1, VK_FORMAT_R8G8B8A8_SINT);
  bind_sampled_image(ctx, h.pipe, 4, scalesImg.view);
  bind_sampled_image(ctx, h.pipe, 5, zpsImg.view);

  // Weight block layout (K4 x N8 blocks) is identical to the tiled kernels'
  // t_packed_int4_weight -- same image dims/format.
  Image weightImg =
      create_sampled_image2d(ctx, K4, N8, VK_FORMAT_R32G32B32A32_SINT);
  bind_sampled_image(ctx, h.pipe, 6, weightImg.view);
  h.ownedImages = {scalesImg, zpsImg, weightImg};

  int32_t sizesData[2][4] = {
      {(int32_t)s.N, (int32_t)s.M, 1, 1}, {(int32_t)s.K, (int32_t)s.M, 1, 1}};
  Buffer outputSizesUbo = alloc_buffer(
      ctx,
      sizeof(sizesData[0]),
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  Buffer inputSizesUbo = alloc_buffer(
      ctx,
      sizeof(sizesData[1]),
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  {
    void* p;
    vkMapMemory(ctx.dev, outputSizesUbo.mem, 0, sizeof(sizesData[0]), 0, &p);
    memcpy(p, sizesData[0], sizeof(sizesData[0]));
    vkUnmapMemory(ctx.dev, outputSizesUbo.mem);
    vkMapMemory(ctx.dev, inputSizesUbo.mem, 0, sizeof(sizesData[1]), 0, &p);
    memcpy(p, sizesData[1], sizeof(sizesData[1]));
    vkUnmapMemory(ctx.dev, inputSizesUbo.mem);
  }
  bind_buffer(
      ctx,
      h.pipe,
      10,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      outputSizesUbo.buf,
      sizeof(sizesData[0]));
  bind_buffer(
      ctx,
      h.pipe,
      11,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      inputSizesUbo.buf,
      sizeof(sizesData[1]));
  h.ownedBuffers.push_back(outputSizesUbo);
  h.ownedBuffers.push_back(inputSizesUbo);

  return h;
}

void destroy_handle(Ctx& ctx, KernelHandle& h) {
  for (auto& b : h.ownedBuffers)
    free_buffer(ctx, b);
  for (auto& im : h.ownedImages)
    free_image(ctx, im);
  destroy_pipeline(ctx, h.pipe);
}

struct Row {
  std::string label;
  uint32_t M, K, N, groupSize;
  double q4MedianUs, q4CovPct, dqMedianUs, dqCovPct, speedup;
};

void run_suite(
    Ctx& ctx,
    VkQueryPool queryPool,
    StorageMode mode,
    uint32_t rounds,
    uint32_t itersPerRound,
    std::vector<Row>& outRows) {
  for (const Shape& s : kShapes) {
    KernelHandle q4 = setup_q4gsw(ctx, s, mode);
    KernelHandle dq = setup_dq8ca(ctx, s, mode);

    // Warmup each once, untimed.
    run_timed_batch(
        ctx,
        queryPool,
        q4.pipe.pipeline,
        q4.pipe.pipeLayout,
        q4.pipe.descSet,
        q4.wg,
        itersPerRound);
    run_timed_batch(
        ctx,
        queryPool,
        dq.pipe.pipeline,
        dq.pipe.pipeLayout,
        dq.pipe.descSet,
        dq.wg,
        itersPerRound);

    std::vector<double> q4Ns, dqNs;
    for (uint32_t r = 0; r < rounds; ++r) {
      // Interleave W(q4gsw), T(dq8ca) per round -- avoids blocked-sampling
      // artifacts on unpinned/floating clocks.
      q4Ns.push_back(run_timed_batch(
          ctx,
          queryPool,
          q4.pipe.pipeline,
          q4.pipe.pipeLayout,
          q4.pipe.descSet,
          q4.wg,
          itersPerRound));
      dqNs.push_back(run_timed_batch(
          ctx,
          queryPool,
          dq.pipe.pipeline,
          dq.pipe.pipeLayout,
          dq.pipe.descSet,
          dq.wg,
          itersPerRound));
    }

    Row row;
    row.label = s.label;
    row.M = s.M;
    row.K = s.K;
    row.N = s.N;
    row.groupSize = s.groupSize;
    row.q4MedianUs = median(q4Ns) / 1000.0;
    row.dqMedianUs = median(dqNs) / 1000.0;
    row.q4CovPct = cov(q4Ns) * 100.0;
    row.dqCovPct = cov(dqNs) * 100.0;
    row.speedup = row.dqMedianUs > 0 ? row.q4MedianUs / row.dqMedianUs : 0.0;

    printf(
        "%s,%u,%u,%u,%u,%.3f,%.2f,%.3f,%.2f,%.4f\n",
        row.label.c_str(),
        row.M,
        row.K,
        row.N,
        row.groupSize,
        row.q4MedianUs,
        row.q4CovPct,
        row.dqMedianUs,
        row.dqCovPct,
        row.speedup);
    fflush(stdout);
    outRows.push_back(row);

    destroy_handle(ctx, q4);
    destroy_handle(ctx, dq);
  }
}

// ---------------------------------------------------------------------------
// q4gsw coopmat (WMMA) vs q4gsw tiled, weight=texture2d on both sides.
// ---------------------------------------------------------------------------

struct CoopmatRow {
  std::string label;
  uint32_t M, K, N;
  double tiledMedianUs;
  std::vector<bool> aligned; // per variant
  std::vector<double> medianUs, covPct, speedup; // per variant
};

using TiledSetupFn = KernelHandle (*)(Ctx&, const Shape&, StorageMode);
using CoopmatSetupFn =
    KernelHandle (*)(Ctx&, const Shape&, const CoopmatVariant&);

// Compares the tiled baseline against N coopmat tile variants at once
// (e.g. "shipped" vs "780M-tuned"), round-robin interleaved per round across
// all N+1 participants so no single config is biased by thermal/clock drift
// relative to the others.
void run_coopmat_suite(
    Ctx& ctx,
    VkQueryPool queryPool,
    uint32_t rounds,
    uint32_t itersPerRound,
    const char* kernelLabel,
    TiledSetupFn setupTiled,
    CoopmatSetupFn setupCoopmat,
    const std::vector<CoopmatVariant>& variants,
    std::vector<CoopmatRow>& outRows) {
  printf("\nlabel,M,K,N,%s_tiled_texture_us_median", kernelLabel);
  for (const CoopmatVariant& v : variants)
    printf(
        ",%s_us_median,%s_cov_pct,%s_speedup_over_tiled",
        v.label,
        v.label,
        v.label);
  printf("\n");

  for (const Shape& s : kShapes) {
    CoopmatRow row{};
    row.label = s.label;
    row.M = s.M;
    row.K = s.K;
    row.N = s.N;

    KernelHandle tiled = setupTiled(ctx, s, StorageMode::kTexture);
    std::vector<KernelHandle> coop(variants.size());
    std::vector<bool> aligned(variants.size());
    for (size_t i = 0; i < variants.size(); ++i) {
      aligned[i] = coopmat_tile_aligned(s, variants[i]);
      if (aligned[i])
        coop[i] = setupCoopmat(ctx, s, variants[i]);
    }

    run_timed_batch(
        ctx,
        queryPool,
        tiled.pipe.pipeline,
        tiled.pipe.pipeLayout,
        tiled.pipe.descSet,
        tiled.wg,
        itersPerRound);
    for (size_t i = 0; i < variants.size(); ++i) {
      if (aligned[i])
        run_timed_batch(
            ctx,
            queryPool,
            coop[i].pipe.pipeline,
            coop[i].pipe.pipeLayout,
            coop[i].pipe.descSet,
            coop[i].wg,
            itersPerRound);
    }

    std::vector<double> tiledNs;
    std::vector<std::vector<double>> coopNs(variants.size());
    for (uint32_t r = 0; r < rounds; ++r) {
      tiledNs.push_back(run_timed_batch(
          ctx,
          queryPool,
          tiled.pipe.pipeline,
          tiled.pipe.pipeLayout,
          tiled.pipe.descSet,
          tiled.wg,
          itersPerRound));
      for (size_t i = 0; i < variants.size(); ++i) {
        if (aligned[i])
          coopNs[i].push_back(run_timed_batch(
              ctx,
              queryPool,
              coop[i].pipe.pipeline,
              coop[i].pipe.pipeLayout,
              coop[i].pipe.descSet,
              coop[i].wg,
              itersPerRound));
      }
    }

    row.tiledMedianUs = median(tiledNs) / 1000.0;
    row.aligned = aligned;
    printf(
        "%s,%u,%u,%u,%.3f",
        row.label.c_str(),
        row.M,
        row.K,
        row.N,
        row.tiledMedianUs);
    for (size_t i = 0; i < variants.size(); ++i) {
      if (!aligned[i]) {
        row.medianUs.push_back(0.0);
        row.covPct.push_back(0.0);
        row.speedup.push_back(0.0);
        printf(
            ",-,-,skipped(%ux%ux%u)",
            variants[i].tileM,
            variants[i].tileN,
            variants[i].tileK);
        continue;
      }
      double m = median(coopNs[i]) / 1000.0;
      double c = cov(coopNs[i]) * 100.0;
      double sp = m > 0 ? row.tiledMedianUs / m : 0.0;
      row.medianUs.push_back(m);
      row.covPct.push_back(c);
      row.speedup.push_back(sp);
      printf(",%.3f,%.2f,%.4f", m, c, sp);
    }
    printf("\n");
    fflush(stdout);
    outRows.push_back(row);

    destroy_handle(ctx, tiled);
    for (size_t i = 0; i < variants.size(); ++i)
      if (aligned[i])
        destroy_handle(ctx, coop[i]);
  }
}

void print_coopmat_summary(
    const std::vector<CoopmatRow>& rows,
    const char* deviceName,
    const char* title,
    const char* kernelLabel,
    const std::vector<CoopmatVariant>& variants) {
  printf("\n============ Summary (%s) ============\n", title);
  if (deviceName)
    printf("Device: %s\n", deviceName);
  printf("%-18s %6s %7s %7s | %10s |", "shape", "M", "K", "N", "tiled(us)");
  for (const CoopmatVariant& v : variants)
    printf(" %-13s %6s %8s |", v.label, "CoV%", "speedup");
  printf("\n");
  const std::string divider(52 + variants.size() * 32, '-');
  printf("%s\n", divider.c_str());

  std::vector<std::vector<double>> speedups(variants.size());
  for (const CoopmatRow& r : rows) {
    printf(
        "%-18s %6u %7u %7u | %10.2f |",
        r.label.c_str(),
        r.M,
        r.K,
        r.N,
        r.tiledMedianUs);
    for (size_t i = 0; i < variants.size(); ++i) {
      if (!r.aligned[i]) {
        printf(" %-13s %6s %8s |", "-", "-", "skipped");
        continue;
      }
      printf(
          " %13.2f %5.2f%% %7.2fx |", r.medianUs[i], r.covPct[i], r.speedup[i]);
      speedups[i].push_back(r.speedup[i]);
    }
    printf("\n");
  }
  printf("%s\n", divider.c_str());

  for (size_t i = 0; i < variants.size(); ++i) {
    double logSum = 0.0;
    for (double x : speedups[i])
      logSum += std::log(x);
    double geomean =
        speedups[i].empty() ? 0.0 : std::exp(logSum / speedups[i].size());
    printf(
        "%s_coopmat[%s]-over-%s_tiled speedup, geomean across %zu aligned "
        "shapes: %.2fx (tile %ux%ux%u, subgroup %u)\n",
        kernelLabel,
        variants[i].label,
        kernelLabel,
        speedups[i].size(),
        geomean,
        variants[i].tileM,
        variants[i].tileN,
        variants[i].tileK,
        variants[i].subgroupSize);
  }
}

double compute_geomean(const std::vector<Row>& rows) {
  if (rows.empty())
    return 0.0;
  double logSum = 0.0;
  for (const Row& r : rows)
    logSum += std::log(r.speedup);
  return std::exp(logSum / rows.size());
}

void print_human_summary(
    const std::vector<Row>& rows,
    StorageMode mode,
    const char* deviceName) {
  printf(
      "\n==================== Summary (storage=%s) ====================\n",
      storage_mode_name(mode));
  if (deviceName)
    printf("Device: %s\n", deviceName);
  printf(
      "%-20s %6s %7s %7s %6s | %11s %8s | %11s %8s | %8s\n",
      "shape",
      "M",
      "K",
      "N",
      "grp",
      "4w(us)",
      "CoV%",
      "8da4w(us)",
      "CoV%",
      "speedup");
  printf("%s\n", std::string(100, '-').c_str());
  for (const Row& r : rows) {
    printf(
        "%-20s %6u %7u %7u %6u | %11.2f %7.2f%% | %11.2f %7.2f%% | %7.2fx\n",
        r.label.c_str(),
        r.M,
        r.K,
        r.N,
        r.groupSize,
        r.q4MedianUs,
        r.q4CovPct,
        r.dqMedianUs,
        r.dqCovPct,
        r.speedup);
  }
  printf("%s\n", std::string(100, '-').c_str());
  double geomean = compute_geomean(rows);
  printf(
      "8da4w-over-4w speedup, geomean across %zu shapes: %.2fx\n",
      rows.size(),
      geomean);
  printf(
      "(baseline/tiled kernels only -- no coopmat/WMMA; dq8ca excludes its\n"
      " activation-quantization pre-pass, so this will differ from e2e ratios)\n");
}

} // namespace

int main(int argc, char** argv) {
  bool validation = getenv("MBENCH_VALIDATION") != nullptr;
  uint32_t rounds = 7;
  uint32_t itersPerRound = 20;
  if (const char* e = getenv("MBENCH_ROUNDS"))
    rounds = (uint32_t)atoi(e);
  if (const char* e = getenv("MBENCH_ITERS"))
    itersPerRound = (uint32_t)atoi(e);

  std::vector<StorageMode> modes = {StorageMode::kBuffer};
  if (const char* e = getenv("MBENCH_STORAGE")) {
    std::string v = e;
    if (v == "texture")
      modes = {StorageMode::kTexture};
    else if (v == "both")
      modes = {StorageMode::kBuffer, StorageMode::kTexture};
    else
      modes = {StorageMode::kBuffer};
  }
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    const std::string prefix = "--storage=";
    if (arg.rfind(prefix, 0) == 0) {
      std::string v = arg.substr(prefix.size());
      if (v == "texture")
        modes = {StorageMode::kTexture};
      else if (v == "both")
        modes = {StorageMode::kBuffer, StorageMode::kTexture};
      else if (v == "buffer")
        modes = {StorageMode::kBuffer};
      else {
        fprintf(
            stderr,
            "Unknown --storage=%s (expected buffer|texture|both)\n",
            v.c_str());
        return 1;
      }
    } else if (arg == "--help" || arg == "-h") {
      printf(
          "usage: %s [--storage=buffer|texture|both]\n"
          "env: MBENCH_STORAGE=buffer|texture|both (default buffer)\n"
          "     MBENCH_ROUNDS=<n> (default 7), MBENCH_ITERS=<n> (default 20)\n"
          "     MBENCH_VALIDATION=1, MBENCH_DEVICE_INDEX=<n>\n",
          argv[0]);
      return 0;
    }
  }

  Ctx ctx = create_context(validation);

  VkPhysicalDeviceProperties devProps;
  vkGetPhysicalDeviceProperties(ctx.phys, &devProps);

  VkQueryPoolCreateInfo qpInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qpInfo.queryCount = 2;
  VkQueryPool queryPool;
  VK_CHECK(vkCreateQueryPool(ctx.dev, &qpInfo, nullptr, &queryPool));

  std::vector<Row> allRows;
  std::vector<StorageMode> ranModes;
  std::vector<double> modeGeomeans;

  for (StorageMode mode : modes) {
    fprintf(stderr, "\n=== Running storage=%s ===\n", storage_mode_name(mode));
    printf("storage=%s\n", storage_mode_name(mode));
    printf(
        "label,M,K,N,group_size,q4gsw_us_median,q4gsw_cov_pct,dq8ca_us_median,dq8ca_cov_pct,dq8ca_speedup_over_q4gsw\n");

    std::vector<Row> rows;
    run_suite(ctx, queryPool, mode, rounds, itersPerRound, rows);
    print_human_summary(rows, mode, devProps.deviceName);

    ranModes.push_back(mode);
    modeGeomeans.push_back(compute_geomean(rows));
    allRows.insert(allRows.end(), rows.begin(), rows.end());
  }

  printf("\n%s\n", std::string(60, '=').c_str());
  printf("GEOMEAN SPEEDUP (8da4w vs 4w, dispatch throughput only)\n");
  printf("%s\n", std::string(60, '=').c_str());
  for (size_t i = 0; i < ranModes.size(); ++i) {
    printf(
        "  storage=%-8s %.2fx\n",
        storage_mode_name(ranModes[i]),
        modeGeomeans[i]);
  }
  if (ranModes.size() > 1) {
    printf(
        "  overall (%zu modes)  %.2fx\n",
        ranModes.size(),
        compute_geomean(allRows));
  }
  printf("%s\n", std::string(60, '=').c_str());

  const bool wantCoopmat =
      std::find(modes.begin(), modes.end(), StorageMode::kTexture) !=
      modes.end();
  if (wantCoopmat && ctx.haveCoopMat) {
    const std::vector<CoopmatVariant> q4Variants = {
        {"shipped(Xclipse)",
         128,
         128,
         16,
         32,
         shader_q4gsw_coopmat_texture_spv,
         shader_q4gsw_coopmat_texture_spv_size},
        {"780M-tuned",
         128,
         64,
         32,
         32,
         shader_q4gsw_coopmat_tuned_spv,
         shader_q4gsw_coopmat_tuned_spv_size},
    };
    fprintf(stderr, "\n=== Running q4gsw_coopmat (WMMA, fp16) ===\n");
    std::vector<CoopmatRow> q4CoopRows;
    run_coopmat_suite(
        ctx,
        queryPool,
        rounds,
        itersPerRound,
        "q4gsw",
        setup_q4gsw,
        setup_q4gsw_coopmat,
        q4Variants,
        q4CoopRows);
    print_coopmat_summary(
        q4CoopRows,
        devProps.deviceName,
        "q4gsw_coopmat(WMMA,fp16) vs q4gsw_tiled, weight=texture2d",
        "q4gsw",
        q4Variants);

    const std::vector<CoopmatVariant> dqVariants = {
        {"shipped(Xclipse)",
         64,
         32,
         32,
         64,
         shader_dq8ca_coopmat_texture_spv,
         shader_dq8ca_coopmat_texture_spv_size},
        {"780M-tuned",
         64,
         128,
         32,
         32,
         shader_dq8ca_coopmat_tuned_spv,
         shader_dq8ca_coopmat_tuned_spv_size},
        // Same tile/subgroup config as q4gsw's "780M-tuned" entry above
        // (128x64x32, g2x2, s32) -- both kernels' tsweep grids happen to
        // include this exact token, so this isolates the kernel/math delta
        // (fp16 MMA + no correction pass vs int8 MMA + per-group dequant
        // correction) from tile-shape choice.
        // Best dq8ca coopmat config found by this microbench's own isolated-
        // dispatch methodology (~1.0x vs tiled) -- beats both production
        // defaults despite carrying a 32-VGPR spill. A follow-up attempt at
        // a smaller tile (64x64x32) cleared that spill entirely but measured
        // ~26% SLOWER (0.74x): halving the tile area doubles the workgroup
        // count, and each workgroup pays a largely fixed tax (barrier sync
        // for the double-buffered shared-memory staging, the per-group
        // correction epilogue) independent of how much useful compute it
        // does -- that lost tile-reuse outweighed the spill fix. Register
        // pressure is not the dominant remaining cost here.
        {"same-tile(128x64x32)",
         128,
         64,
         32,
         32,
         shader_dq8ca_coopmat_sametile_spv,
         shader_dq8ca_coopmat_sametile_spv_size},
    };
    fprintf(stderr, "\n=== Running dq8ca_q4gsw_coopmat (WMMA, int8) ===\n");
    std::vector<CoopmatRow> dqCoopRows;
    run_coopmat_suite(
        ctx,
        queryPool,
        rounds,
        itersPerRound,
        "dq8ca",
        setup_dq8ca,
        setup_dq8ca_coopmat,
        dqVariants,
        dqCoopRows);
    print_coopmat_summary(
        dqCoopRows,
        devProps.deviceName,
        "dq8ca_q4gsw_coopmat(WMMA,int8) vs dq8ca_q4gsw_tiled, weight=texture2d",
        "dq8ca",
        dqVariants);
  } else if (wantCoopmat && !ctx.haveCoopMat) {
    fprintf(
        stderr,
        "\n=== Skipping q4gsw_coopmat (WMMA): device lacks cooperative "
        "matrix / subgroup-size-control support ===\n");
  }

  vkDestroyQueryPool(ctx.dev, queryPool, nullptr);
  return 0;
}
