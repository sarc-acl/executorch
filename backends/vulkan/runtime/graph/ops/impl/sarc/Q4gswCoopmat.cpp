/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Q4gswCoopmat.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizeDequantize.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <algorithm>
#include <cctype>

namespace vkcompute {

// Defined in QuantizedLinear.cpp (not declared in a header in release 1.5).
void resize_linear_qw_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args);

GlobalWorkGrid quantized_linear_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

namespace sarc {

namespace {

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
  d.max_shared_bytes = adapter->max_compute_shared_memory_size();
  return d;
}

// args layout of the dispatch node below:
// {{output}, {fp_input, packed_weight, packed_weight_scales, packed_bias}}
// resize_args: {group_size, weight_data, bias_data}
ShapeInfo shape_info(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef output = args.at(0).refs.at(0);
  const ValueRef fp_input = args.at(1).refs.at(0);
  const ValueRef packed_weight = args.at(1).refs.at(1);

  const std::vector<int64_t> out_sizes = graph->sizes_of(output);
  ShapeInfo s;
  s.op = Op::kQ4gswLinear;
  s.N = utils::val_at(-1, out_sizes);
  s.M = utils::val_at(-2, out_sizes);
  s.K = utils::val_at(-1, graph->sizes_of(fp_input));
  s.group_size = graph->extract_scalar<int64_t>(resize_args.at(0));
  for (int64_t d = 0; d < graph->dim_of(output) - 2; d++) {
    s.batch *= utils::val_at(d, out_sizes);
  }
  s.gemv = is_gemv(graph, fp_input);
  s.has_bias = !graph->val_is_none(resize_args.at(2));
  s.half = graph->dtype_of(output) == vkapi::kHalf;
  s.input = to_storage(graph->storage_type_of(fp_input));
  s.output = to_storage(graph->storage_type_of(output));
  s.weight = to_storage(graph->storage_type_of(packed_weight));
  s.io_width_packed = graph->packed_dim_of(output) == WHCN::kWidthDim &&
      graph->packed_dim_of(fp_input) == WHCN::kWidthDim;
  return s;
}

vkapi::ShaderInfo pick_sarc_q4gsw_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef output = args.at(0).refs.at(0);
  const ValueRef fp_input = args.at(1).refs.at(0);
  const ValueRef packed_weight = args.at(1).refs.at(1);

  const std::optional<Choice> choice =
      select(device_info(graph), shape_info(graph, args, resize_args));

  std::string kernel_name;
  if (choice.has_value()) {
    kernel_name = choice->kernel_base;
  } else {
    // Shapes the SARC variants do not cover (decode/GEMV, unaligned M, bias,
    // or a forced fallback): the release-1.5 kernels that read this packed
    // weight layout (prepack_quantized_linear_weight, 4-bit).
    kernel_name = is_gemv(graph, fp_input) ? "linear_q4gsw_coop"
                                           : "linear_q4gsw_tiled";
  }
  add_storage_type_suffix(kernel_name, graph->storage_type_of(output));
  add_storage_type_suffix(kernel_name, graph->storage_type_of(packed_weight));
  add_dtype_suffix(kernel_name, graph->dtype_of(output));
  return VK_KERNEL_FROM_STR(kernel_name);
}

GlobalWorkGrid sarc_q4gsw_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const std::optional<TileDims> dims = dims_for_kernel(shader.kernel_name);
  if (!dims.has_value()) {
    return quantized_linear_gwg(graph, shader, args, resize_args);
  }
  const std::vector<int64_t> out_sizes =
      graph->sizes_of(args.at(0).refs.at(0));
  const uint32_t N = utils::val_at(-1, out_sizes);
  const uint32_t M = utils::val_at(-2, out_sizes);
  // One workgroup per WG_TILE_M x WG_TILE_N output tile; the x extent is
  // multiplied by the workgroup size because the framework divides by it.
  const uint32_t wg_size = dims->wg_size();
  return GlobalWorkGrid(
      {utils::div_up(N, dims->n) * wg_size, utils::div_up(M, dims->m), 1u},
      kTiledWorkGrid,
      LocalWorkGroup(wg_size, 1u, 1u));
}

} // namespace

bool try_add_q4gsw_coopmat(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  const ValueRef fp_input = args.at(0);
  const ValueRef weight_data = args.at(1);
  const ValueRef weight_scales_data = args.at(2);
  const ValueRef group_size = args.at(3);
  const ValueRef bias_data = args.at(4);
  const ValueRef output = args.at(5);

  // SARC variants are fp16 only.
  if (graph.dtype_of(fp_input) != vkapi::kHalf) {
    return false;
  }
  if (!device_has_rows(device_info(&graph), Op::kQ4gswLinear)) {
    return false;
  }

  const int64_t group_size_val = graph.extract_scalar<int64_t>(group_size);
  const QuantizationConfig weight_quant_config(
      4, kPerGroup, {group_size_val});

  const ValueRef packed_weight =
      prepack_quantized_linear_weight(graph, weight_quant_config, weight_data);
  const ValueRef packed_weight_scales = prepack_standard(
      graph, weight_scales_data, utils::kBuffer, utils::kWidthPacked);

  TmpTensor dummy_bias(
      &graph, {}, graph.dtype_of(output), utils::kBuffer, utils::kWidthPacked);
  ValueRef packed_bias = dummy_bias.vref;
  uint32_t apply_bias = 0;
  if (graph.val_is_not_none(bias_data)) {
    packed_bias =
        prepack_standard(graph, bias_data, utils::kBuffer, utils::kWidthPacked);
    apply_bias = 1;
  }

  const int32_t K4_per_group =
      utils::div_up(static_cast<int32_t>(group_size_val), int32_t(4));
  const int32_t num_groups =
      graph.size_at<int32_t>(-1, fp_input) / static_cast<int32_t>(group_size_val);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_sarc_q4gsw_shader,
      sarc_q4gsw_gwg,
      quantized_linear_lwg,
      {{output, vkapi::kWrite},
       {{fp_input, packed_weight, packed_weight_scales, packed_bias},
        vkapi::kRead}},
      {graph.sizes_ubo(output), graph.sizes_ubo(fp_input)},
      {},
      // Same spec constants as release 1.5's linear_q4gsw kernels. N is a
      // spec constant because Xclipse miscompiles UBO-derived store offsets.
      {apply_bias,
       K4_per_group,
       num_groups,
       graph.size_at<int32_t>(-1, output)},
      {group_size, weight_data, bias_data},
      resize_linear_qw_node));
  return true;
}

} // namespace sarc
} // namespace vkcompute
