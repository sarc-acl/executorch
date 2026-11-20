/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2dSimt.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/Convolution.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/im2col/Conv2dGemm.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>

namespace vkcompute {

namespace {

constexpr uint32_t kMergeThreads = 64;
constexpr const char* kMergeShader = "conv2d_simt_merge_float";

struct LiveTaps final {
  int64_t ki0;
  int64_t kh;
  int64_t kj0;
  int64_t kw;
};

LiveTaps live_taps(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef out) {
  const std::vector<int64_t> in_sizes = graph.sizes_of(in);
  const std::vector<int64_t> w_sizes = graph.sizes_of(weight_data);
  const std::vector<int64_t> out_sizes = graph.sizes_of(out);
  const auto s = graph.get_int_list(stride);
  const auto p = graph.get_int_list(padding);
  const auto d = graph.get_int_list(dilation);
  std::pair<int64_t, int64_t> live[2];
  for (size_t a = 0; a < 2; ++a) {
    live[a] = conv2d_live_taps(
        in_sizes.at(2 + a),
        out_sizes.at(2 + a),
        w_sizes.at(2 + a),
        s->at(a),
        p->at(a),
        d->at(a));
  }
  return {live[0].first, live[0].second, live[1].first, live[1].second};
}

utils::uvec3 workgroups(
    ComputeGraph& graph,
    const Conv2dSimtVariant& v,
    const ValueRef out) {
  const uint32_t M =
      graph.size_at<uint32_t>(-1, out) * graph.size_at<uint32_t>(-2, out);
  const uint32_t m_tiles = utils::div_up(M, v.tile_m);
  const uint32_t max_y =
      graph.context()->adapter_ptr()->max_compute_workgroup_count()[1];
  const uint32_t y = std::min(m_tiles, max_y);
  return {
      utils::div_up(graph.size_at<uint32_t>(-3, out), v.tile_n) * v.k_slices,
      y,
      utils::div_up(m_tiles, y)};
}

// resize_args = {in, weight_data, stride, padding, dilation, residual}
void resize_conv2d_simt_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const std::vector<int64_t> out_hw = calc_out_sizes_hw(
      *graph,
      graph->sizes_of(resize_args.at(0)),
      resize_args.at(1),
      /*kernel_size_only=*/false,
      {resize_args.at(2),
       resize_args.at(3),
       resize_args.at(4),
       resize_args.at(4)},
      /*transposed=*/false);
  std::vector<int64_t> new_sizes = graph->sizes_of(out);
  const size_t ndim = new_sizes.size();
  new_sizes.at(ndim - 2) = out_hw.at(0);
  new_sizes.at(ndim - 1) = out_hw.at(1);
  graph->virtual_resize(out, new_sizes);
  check_conv2d_residual_on_resize(graph, out, resize_args);
}

} // namespace

const std::vector<Conv2dSimtVariant>& conv2d_simt_variants() {
  // {name, tile_m, tile_n, tile_k, tm, tn4, k_slices, direct}; must match
  // conv2d_simt.yaml.
  // clang-format off
  static const std::vector<Conv2dSimtVariant> variants = {
      {"conv2d_simt_t32x32_m4n1_float", 32, 32, 16, 4, 1, 1, false},
      {"conv2d_simt_t32x32_m4n1_sk2_float", 32, 32, 16, 4, 1, 2, false},
      {"conv2d_simt_t32x32_m4n1_sk4_float", 32, 32, 16, 4, 1, 4, false},
      {"conv2d_simt_t32x32_m4n1_sk8_float", 32, 32, 16, 4, 1, 8, false},
      {"conv2d_simt_t32x32_m4n1_dg_float", 32, 32, 16, 4, 1, 1, true},
      {"conv2d_simt_t32x32_m4n1_dg_sk2_float", 32, 32, 16, 4, 1, 2, true},
      {"conv2d_simt_t32x32_m4n1_dg_sk4_float", 32, 32, 16, 4, 1, 4, true},
      {"conv2d_simt_t32x32_m4n1_dg_sk8_float", 32, 32, 16, 4, 1, 8, true},
      {"conv2d_simt_t64x32_m4n2_float", 64, 32, 16, 4, 2, 1, false},
      {"conv2d_simt_t64x32_m4n2_sk2_float", 64, 32, 16, 4, 2, 2, false},
      {"conv2d_simt_t64x32_m4n2_sk4_float", 64, 32, 16, 4, 2, 4, false},
      {"conv2d_simt_t64x32_m4n2_sk8_float", 64, 32, 16, 4, 2, 8, false},
      {"conv2d_simt_t64x32_m4n2_dg_float", 64, 32, 16, 4, 2, 1, true},
      {"conv2d_simt_t64x32_m4n2_dg_sk2_float", 64, 32, 16, 4, 2, 2, true},
      {"conv2d_simt_t64x32_m4n2_dg_sk4_float", 64, 32, 16, 4, 2, 4, true},
      {"conv2d_simt_t64x32_m4n2_dg_sk8_float", 64, 32, 16, 4, 2, 8, true},
      {"conv2d_simt_t64x64_m4n2_float", 64, 64, 16, 4, 2, 1, false},
      {"conv2d_simt_t64x64_m4n2_sk2_float", 64, 64, 16, 4, 2, 2, false},
      {"conv2d_simt_t64x64_m4n2_sk4_float", 64, 64, 16, 4, 2, 4, false},
      {"conv2d_simt_t64x64_m4n2_sk8_float", 64, 64, 16, 4, 2, 8, false},
      {"conv2d_simt_t64x64_m4n2_dg_float", 64, 64, 16, 4, 2, 1, true},
      {"conv2d_simt_t64x64_m4n2_dg_sk2_float", 64, 64, 16, 4, 2, 2, true},
      {"conv2d_simt_t64x64_m4n2_dg_sk4_float", 64, 64, 16, 4, 2, 4, true},
      {"conv2d_simt_t64x64_m4n2_dg_sk8_float", 64, 64, 16, 4, 2, 8, true},
      {"conv2d_simt_t128x64_m8n2_float", 128, 64, 16, 8, 2, 1, false},
      {"conv2d_simt_t128x64_m8n2_sk2_float", 128, 64, 16, 8, 2, 2, false},
      {"conv2d_simt_t128x64_m8n2_sk4_float", 128, 64, 16, 8, 2, 4, false},
      {"conv2d_simt_t128x64_m8n2_sk8_float", 128, 64, 16, 8, 2, 8, false},
      {"conv2d_simt_t128x64_m8n2_dg_float", 128, 64, 16, 8, 2, 1, true},
      {"conv2d_simt_t128x64_m8n2_dg_sk2_float", 128, 64, 16, 8, 2, 2, true},
      {"conv2d_simt_t128x64_m8n2_dg_sk4_float", 128, 64, 16, 8, 2, 4, true},
      {"conv2d_simt_t128x64_m8n2_dg_sk8_float", 128, 64, 16, 8, 2, 8, true},
  };
  // clang-format on
  return variants;
}

const Conv2dSimtVariant* find_conv2d_simt_variant(const std::string& name) {
  for (const Conv2dSimtVariant& v : conv2d_simt_variants()) {
    if (name == v.name) {
      return &v;
    }
  }
  return nullptr;
}

std::optional<std::string> conv2d_simt_ineligible(
    ComputeGraph& graph,
    const Conv2dSimtVariant& v,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const int64_t groups,
    const bool transposed,
    const ValueRef out) {
  const auto n = [](const uint64_t x) { return std::to_string(x); };
  if (transposed) {
    return "the convolution is transposed";
  }
  if (groups != 1) {
    return "the convolution is grouped";
  }
  if (graph.dtype_of(in) != vkapi::kFloat ||
      graph.dtype_of(out) != vkapi::kFloat) {
    return "input and output are not FP32";
  }
  if (!graph.val_is_tref(weight_data) ||
      graph.dtype_of(weight_data) != vkapi::kFloat) {
    return "the weights are not a constant FP32 tensor";
  }
  if (!graph.val_is_none(bias) &&
      (!graph.val_is_tref(bias) || graph.dtype_of(bias) != vkapi::kFloat)) {
    return "the bias is not a constant FP32 tensor";
  }
  if (graph.storage_type_of(in) != utils::kTexture3D ||
      graph.storage_type_of(out) != utils::kTexture3D ||
      graph.packed_dim_of(in) != WHCN::kChannelsDim ||
      graph.packed_dim_of(out) != WHCN::kChannelsDim ||
      !graph.has_standard_axis_map(in) || !graph.has_standard_axis_map(out)) {
    return "input and output are not channels-packed textures with the "
           "standard axis map";
  }
  const std::vector<int64_t> in_sizes = graph.sizes_of(in);
  const std::vector<int64_t> w_sizes = graph.sizes_of(weight_data);
  if (in_sizes.size() != 4 || in_sizes.at(0) != 1 || w_sizes.size() != 4 ||
      w_sizes.at(1) != in_sizes.at(1)) {
    return "not a batch-1 2D convolution";
  }
  if (!v.direct) {
    const auto s = graph.get_int_list(stride);
    const auto p = graph.get_int_list(padding);
    if (w_sizes.at(2) != 1 || w_sizes.at(3) != 1 || s->at(0) != 1 ||
        s->at(1) != 1 || p->at(0) != 0 || p->at(1) != 0) {
      return "not a 1x1, stride 1, padding 0 convolution (use a _dg variant)";
    }
  }
  const LiveTaps live =
      live_taps(graph, in, weight_data, stride, padding, dilation, out);
  if (live.kh == 0 || live.kw == 0) {
    return "no kernel tap reads the input";
  }
  vkapi::Adapter* const adapter = graph.context()->adapter_ptr();
  if (v.wg_size() > adapter->max_compute_workgroup_invocations() ||
      v.wg_size() > adapter->max_compute_workgroup_size()[0]) {
    return "work group size " + n(v.wg_size()) + " exceeds the device limit";
  }
  if (v.shared_bytes() > adapter->max_compute_shared_memory_size()) {
    return "shared memory " + n(v.shared_bytes()) + " bytes exceeds the limit";
  }
  const uint64_t M = uint64_t(graph.size_at<uint32_t>(-1, out)) *
      graph.size_at<uint32_t>(-2, out);
  const uint64_t N4 = utils::div_up_4(graph.size_at<uint64_t>(-3, out));
  const utils::uvec3 max_count = adapter->max_compute_workgroup_count();
  if (v.k_slices > 1u) {
    const uint64_t K = live.kh * live.kw * utils::align_up_4(w_sizes.at(1));
    const uint64_t steps = utils::div_up(K, uint64_t(v.tile_k));
    if (steps < 2u * v.k_slices) {
      return "fewer than two K steps per slice of " + n(v.k_slices) + " (" +
          n(steps) + " K steps)";
    }
    if (16u * v.k_slices * M * N4 > graph.max_buffer_numel()) {
      return "the split-K scratch exceeds maxStorageBufferRange";
    }
    if (utils::div_up(M * N4, uint64_t(kMergeThreads)) > max_count[0]) {
      return "the merge needs too many work groups in x";
    }
  }
  return conv2d_pw_coopmat_grid_unsupported(
      workgroups(graph, v, out), max_count);
}

void conv2d_simt_impl(
    ComputeGraph& graph,
    const Conv2dSimtVariant& v,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef out,
    const Conv2dEpilogue& epilogue) {
  check_conv2d_epilogue(graph, epilogue, weight_data, out);
  const std::vector<int64_t> w_sizes = graph.sizes_of(weight_data);
  const LiveTaps live =
      live_taps(graph, in, weight_data, stride, padding, dilation, out);
  const bool all_taps = live.ki0 == 0 && live.kj0 == 0 &&
      live.kh == w_sizes.at(2) && live.kw == w_sizes.at(3);
  ValueRef gather_weight = weight_data;
  if (!all_taps) {
    Conv2dIm2colCoopmatPlan plan{};
    plan.ki0 = live.ki0;
    plan.kh = live.kh;
    plan.kj0 = live.kj0;
    plan.kw = live.kw;
    plan.chunks = 1;
    plan.chunk_channels = w_sizes.at(1);
    gather_weight = slice_conv2d_weight(graph, weight_data, plan);
  }
  const ValueRef packed_weight =
      prepack_conv2d_gemm_weight(graph, gather_weight, utils::kBuffer);
  const ValueRef packed_bias = prepack_biases(
      graph,
      bias,
      weight_data,
      /*transposed=*/false,
      utils::kTexture2D,
      utils::kWidthPacked);
  const ValueRef packed_scale = prepack_conv2d_residual_scale(
      graph, epilogue, weight_data, /*transposed=*/false);
  check_conv_args(graph, in, out);
  const Conv2dEpilogueArgs epilogue_args(
      epilogue, in, packed_bias, packed_scale);

  int32_t s[2], p[2], d[2];
  {
    const auto stride_list = graph.get_int_list(stride);
    const auto padding_list = graph.get_int_list(padding);
    const auto dilation_list = graph.get_int_list(dilation);
    for (size_t a = 0; a < 2; ++a) {
      s[a] = utils::safe_downcast<int32_t>(stride_list->at(a));
      p[a] = utils::safe_downcast<int32_t>(padding_list->at(a));
      d[a] = utils::safe_downcast<int32_t>(dilation_list->at(a));
    }
  }
  // The shader's tap 0 is live tap (ki0, kj0).
  const utils::ivec4 geometry{
      s[0],
      s[1],
      p[0] - utils::safe_downcast<int32_t>(live.ki0) * d[0],
      p[1] - utils::safe_downcast<int32_t>(live.kj0) * d[1]};
  const Conv2dEpilogueParams epilogue_params =
      create_conv2d_epilogue_params(epilogue, /*f16_epilogue=*/false);
  const int32_t K4 = utils::safe_downcast<int32_t>(
      live.kh * live.kw * utils::div_up_4(w_sizes.at(1)));
  const int32_t k_slices = utils::safe_downcast<int32_t>(v.k_slices);
  const std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&geometry, sizeof(geometry)),
      PushConstantDataInfo(&epilogue_params, sizeof(epilogue_params)),
      PushConstantDataInfo(&K4, sizeof(K4)),
      PushConstantDataInfo(&k_slices, sizeof(k_slices)),
  };
  vkapi::SpecVarList spec_vars;
  append_conv2d_epilogue_spec_vars(spec_vars, epilogue);
  spec_vars.append(utils::safe_downcast<uint32_t>(w_sizes.at(1)));
  spec_vars.append(utils::safe_downcast<uint32_t>(live.kh));
  spec_vars.append(utils::safe_downcast<uint32_t>(live.kw));
  spec_vars.append(static_cast<uint32_t>(d[0]));
  spec_vars.append(static_cast<uint32_t>(d[1]));
  const std::vector<ValueRef> resize_args = {
      in, weight_data, stride, padding, dilation, epilogue.residual};
  const Conv2dSimtVariant* const vp = &v;
  const auto kernel_gwg = [vp](
                              ComputeGraph* graph,
                              const vkapi::ShaderInfo&,
                              const std::vector<ArgGroup>& args,
                              const std::vector<ValueRef>&) {
    const utils::uvec3 grid = workgroups(*graph, *vp, args.at(0).refs.at(0));
    return GlobalWorkGrid(
        {grid[0] * vp->wg_size(), grid[1], grid[2]}, kExplicitWorkGrid);
  };
  const auto kernel_lwg = [vp](
                              ComputeGraph*,
                              const vkapi::ShaderInfo&,
                              const GlobalWorkGrid&,
                              const std::vector<ArgGroup>&,
                              const std::vector<ValueRef>&) {
    return LocalWorkGroup(vp->wg_size(), 1u, 1u);
  };
  if (v.k_slices <= 1u) {
    graph.execute_nodes().emplace_back(new DynamicDispatchNode(
        graph,
        VK_KERNEL_FROM_STR(v.name),
        kernel_gwg,
        kernel_lwg,
        {{out, vkapi::kWrite},
         {{in,
           packed_weight,
           packed_bias,
           epilogue_args.residual,
           epilogue_args.scale},
          vkapi::kRead}},
        {graph.sizes_ubo(in), graph.sizes_ubo(out)},
        push_constants,
        spec_vars,
        resize_args,
        resize_conv2d_simt_node));
    return;
  }
  // The scratch covers the largest plane the output's storage can hold.
  const VkExtent3D extents = graph.image_of(out).extents();
  TmpTensor scratch(
      &graph,
      {int64_t(v.k_slices) * extents.width * extents.height *
       utils::div_up_4(graph.size_at<int64_t>(-3, out)) * 4},
      vkapi::kFloat,
      utils::kBuffer,
      utils::kWidthPacked);
  const vkapi::ParamsBindList params = {
      graph.sizes_ubo(in), graph.sizes_ubo(out)};
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(v.name),
      kernel_gwg,
      kernel_lwg,
      {{out, vkapi::kWrite},
       {{in,
         packed_weight,
         packed_bias,
         epilogue_args.residual,
         epilogue_args.scale},
        vkapi::kRead},
       {scratch.vref, vkapi::kReadWrite}},
      params,
      push_constants,
      spec_vars,
      resize_args,
      resize_conv2d_simt_node));
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kMergeShader),
      [](ComputeGraph* graph,
         const vkapi::ShaderInfo&,
         const std::vector<ArgGroup>& args,
         const std::vector<ValueRef>&) {
        const ValueRef out = args.at(0).refs.at(0);
        const uint32_t texels = graph->size_at<uint32_t>(-1, out) *
            graph->size_at<uint32_t>(-2, out) *
            utils::div_up_4(graph->size_at<uint32_t>(-3, out));
        return GlobalWorkGrid(
            {utils::div_up(texels, kMergeThreads) * kMergeThreads, 1u, 1u},
            kExplicitWorkGrid);
      },
      [](ComputeGraph*,
         const vkapi::ShaderInfo&,
         const GlobalWorkGrid&,
         const std::vector<ArgGroup>&,
         const std::vector<ValueRef>&) {
        return LocalWorkGroup(kMergeThreads, 1u, 1u);
      },
      {{out, vkapi::kWrite},
       {{in,
         packed_weight,
         packed_bias,
         epilogue_args.residual,
         epilogue_args.scale},
        vkapi::kRead},
       {scratch.vref, vkapi::kReadWrite}},
      params,
      push_constants,
      spec_vars,
      resize_args,
      resize_conv2d_simt_node));
}

} // namespace vkcompute
