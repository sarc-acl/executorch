// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2d.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/RouteRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/Conv2dRouteCommon.h>

namespace vkcompute::shader_profiling {
using namespace convolution;
namespace {

// Forward, ungrouped, batch-1 1x1 convolution with floating-point
// channels-packed texture3d input and output and constant weights.
ConvShape pointwise_shape(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  if (args.size() != 10u && args.size() != 12u && args.size() != 17u)
    throw std::invalid_argument(
        "Convolution route received wrong argument count");
  ConvShape shape = conv_shape_base(
      graph, args[0], args[1], args.back(), graph.get_int(args[8]), 4);
  supported(
      !graph.get_bool(args[6]),
      "Pointwise routes support forward convolution only");
  supported(shape.groups == 1, "Pointwise routes require groups 1");
  supported(shape.input[0] == 1, "Pointwise routes require batch 1");
  supported(
      shape.weight[1] == shape.input[1] && shape.weight[2] == 1 &&
          shape.weight[3] == 1,
      "Pointwise routes require a 1x1 kernel over every input channel");
  spatial_parameter(graph, args[7], false);
  shape.kernel = {1, 1};
  check_conv_output(graph, shape, args[3], args[4], args[5]);
  const vkapi::ScalarType dtype = graph.dtype_of(args[0]);
  supported(
      (dtype == vkapi::kFloat || dtype == vkapi::kHalf) &&
          graph.dtype_of(args.back()) == dtype &&
          graph.dtype_of(args[1]) == dtype,
      "Pointwise routes require matching floating input/output/weight dtypes");
  supported(
      graph.storage_type_of(args[0]) == utils::kTexture3D &&
          graph.storage_type_of(args.back()) == utils::kTexture3D &&
          graph.packed_dim_of(args[0]) == WHCN::kChannelsDim &&
          graph.packed_dim_of(args.back()) == WHCN::kChannelsDim,
      "Pointwise routes require channels-packed texture3d IO");
  if (graph.val_is_not_none(args[2])) {
    constant_tensor(graph, args[2], "Pointwise convolution bias");
    supported(
        graph.dtype_of(args[2]) == dtype &&
            graph.sizes_of(args[2]) == std::vector<int64_t>{shape.output[1]},
        "Pointwise bias must match the dtype and output channels");
  }
  return shape;
}

void add_pointwise(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Conv2dPwRequest& request) {
  conv2d_pw_impl(
      graph,
      args[0],
      args[1],
      args[2],
      args[3],
      args[4],
      args.back(),
      /*transposed_val=*/false,
      conv2d_epilogue_from_args(graph, args),
      request);
}

// Throws UnsupportedRoute naming the first device requirement of the variant
// that is not met.
void device_offers(ComputeGraph& graph, const Conv2dPwCoopmatVariant& v) {
  vkapi::Adapter* const adapter = graph.context()->adapter_ptr();
  supported(
      adapter->supports_cooperative_matrix(),
      "The device does not support VK_KHR_cooperative_matrix");
#ifdef VK_KHR_cooperative_matrix
  const VkComponentTypeKHR acc = v.fp32_accumulation
      ? VK_COMPONENT_TYPE_FLOAT32_KHR
      : VK_COMPONENT_TYPE_FLOAT16_KHR;
  supported(
      adapter->supports_cooperative_matrix_shape(
          v.mma_m,
          v.mma_n,
          v.mma_k,
          VK_COMPONENT_TYPE_FLOAT16_KHR,
          VK_COMPONENT_TYPE_FLOAT16_KHR,
          acc,
          acc),
      std::string("The device does not enumerate the cooperative matrix ") +
          std::to_string(v.mma_m) + "x" + std::to_string(v.mma_n) + "x" +
          std::to_string(v.mma_k) + " fp16 -> " +
          (v.fp32_accumulation ? "fp32" : "fp16") + " at subgroup scope of " +
          v.name);
#endif /* VK_KHR_cooperative_matrix */
  supported(
      adapter->supports_required_subgroup_size_for_compute() &&
          v.subgroup_size >= adapter->min_subgroup_size() &&
          v.subgroup_size <= adapter->max_subgroup_size(),
      "The device cannot require subgroup size " +
          std::to_string(v.subgroup_size) + " for " + v.name);
  supported(
      conv2d_pw_coopmat_device_supports(graph, v),
      std::string("The device lacks a SPIR-V capability of ") + v.name);
}

Json build_pw_coopmat(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  check_options(options, {"variant"});
  if (!options.contains("variant") || !options.at("variant").is_string()) {
    throw std::invalid_argument("pw_coopmat requires a variant name");
  }
  const std::string name = options.at("variant");
  const Conv2dPwCoopmatVariant* const variant =
      find_conv2d_pw_coopmat_variant(name);
  supported(variant != nullptr, "Unknown conv2d_pw_coopmat variant: " + name);
  device_offers(graph, *variant);
  const ConvShape shape = pointwise_shape(graph, args);
  supported(
      graph.dtype_of(args[0]) == vkapi::kHalf,
      "pw_coopmat requires FP16 input and output");
  supported(
      graph.float16_buffers_enabled(),
      "pw_coopmat requires 16-bit storage buffers for its packed weights");
  const auto stride = spatial_parameter(graph, args[3], true);
  const auto padding = spatial_parameter(graph, args[4], false);
  const auto dilation = spatial_parameter(graph, args[5], true);
  supported(
      stride[0] == 1 && stride[1] == 1 && padding[0] == 0 && padding[1] == 0 &&
          dilation[0] == 1 && dilation[1] == 1,
      "pw_coopmat requires stride 1, padding 0 and dilation 1");
  supported(
      shape.output[2] * shape.output[3] > 1,
      "pw_coopmat requires more than one output pixel");
  const size_t first = graph.execute_nodes().size();
  add_pointwise(graph, args, {Conv2dPwRequest::Kind::kCoopmat, variant});
  // The checks above cover the eligibility conditions; a different pick here
  // would mean they missed one, so fail rather than report another kernel.
  for (size_t i = first; i < graph.execute_nodes().size(); ++i) {
    const std::string& node = graph.execute_nodes()[i]->name();
    if (node.rfind("conv2d_pw", 0) == 0 && node != name) {
      throw std::runtime_error(
          "pw_coopmat picked " + node + " instead of " + name);
    }
  }
  return selected("pw_coopmat", {{"variant", name}});
}

Json build_pw_tiled_4x8(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  check_options(options);
  pointwise_shape(graph, args);
  add_pointwise(graph, args, {Conv2dPwRequest::Kind::kTiledWide});
  return selected("pw_tiled_4x8");
}

REGISTER_OPERATOR_ROUTES {
  REGISTER_OPERATOR_ROUTE(
      aten.convolution.default, pw_coopmat, build_pw_coopmat);
  REGISTER_OPERATOR_ROUTE(
      aten.convolution.default, pw_tiled_4x8, build_pw_tiled_4x8);
  REGISTER_OPERATOR_ROUTE(
      et_vk.conv_with_clamp.default, pw_coopmat, build_pw_coopmat);
  REGISTER_OPERATOR_ROUTE(
      et_vk.conv_with_clamp.default, pw_tiled_4x8, build_pw_tiled_4x8);
  REGISTER_OPERATOR_ROUTE(
      et_vk.conv2d.with_epilogue, pw_coopmat, build_pw_coopmat);
  REGISTER_OPERATOR_ROUTE(
      et_vk.conv2d.with_epilogue, pw_tiled_4x8, build_pw_tiled_4x8);
}

} // namespace
} // namespace vkcompute::shader_profiling
