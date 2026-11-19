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
  pointwise_shape(graph, args);
  const auto dilation = spatial_parameter(graph, args[5], true);
  supported(
      dilation[0] == 1 && dilation[1] == 1, "pw_coopmat requires dilation 1");
  const Conv2dEpilogue epilogue = conv2d_epilogue_from_args(graph, args);
  const bool has_bias_or_epilogue = graph.val_is_not_none(args[2]) ||
      epilogue.has_activation() || epilogue.has_residual() ||
      epilogue.has_scale();
  // Every eligibility condition of the dispatch node, for the initial sizes and
  // the storage the prepacking will give the weights, so an ineligible case is
  // reported here instead of being built on the tiled fallback.
  if (const auto reason = conv2d_pw_coopmat_ineligible(
          graph,
          *variant,
          args[0],
          args[1],
          conv2d_pw_weight_storage(graph, args[1], args.back()),
          args[3],
          args[4],
          args.back(),
          /*transposed=*/false,
          has_bias_or_epilogue)) {
    throw UnsupportedRoute(name + " cannot run this case: " + *reason);
  }
  const size_t first = graph.execute_nodes().size();
  add_pointwise(graph, args, {Conv2dPwRequest::Kind::kCoopmat, variant});
  // Unreachable unless conv2d_pw_coopmat_ineligible and the dispatch node's
  // pick disagree: fail rather than report another kernel.
  for (size_t i = first; i < graph.execute_nodes().size(); ++i) {
    const std::string& node = graph.execute_nodes()[i]->name();
    if (node.rfind("conv2d_pw", 0) == 0 && node != name) {
      throw std::runtime_error(
          "pw_coopmat picked " + node + " instead of " + name);
    }
  }
  if (variant->has(Conv2dPwCoopmatVariant::kDiagOnce)) {
    return selected("pw_coopmat", {{"variant", name}, {"diagnostic", true}});
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
