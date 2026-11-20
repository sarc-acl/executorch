/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv1d/Conv1dSimt.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2d.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/im2col/Conv2dGemm.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <algorithm>
#include <cstring>

namespace vkcompute {

namespace {

constexpr uint32_t kMergeThreads = 64;

// The shader's push constants, in two blocks of at most 16 bytes.
struct Geometry final {
  int32_t K4;
  int32_t stride;
  int32_t padding;
  int32_t dilation;
};
struct Epilogue final {
  int32_t taps;
  float out_min;
  float out_max;
  int32_t k_slices;
};

int64_t first_int(ComputeGraph& graph, const ValueRef ref) {
  return graph.get_int_list(ref)->at(0);
}

// GEMM rows and column texels at the output's current sizes.
std::pair<uint32_t, uint32_t> gemm_extents(
    ComputeGraph& graph,
    const Conv1dSimtVariant& v,
    const ValueRef out) {
  const uint32_t L = graph.size_at<uint32_t>(-1, out);
  const uint32_t C = graph.size_at<uint32_t>(-2, out);
  const uint32_t N = graph.size_at<uint32_t>(-3, out);
  if (v.pw) {
    return {N * L, utils::div_up_4(C)};
  }
  return {C, L * utils::div_up_4(N)};
}

utils::uvec3 workgroups(
    ComputeGraph& graph,
    const Conv1dSimtVariant& v,
    const ValueRef out) {
  const auto [M, N4] = gemm_extents(graph, v, out);
  const uint32_t m_tiles = utils::div_up(M, v.tile_m);
  const uint32_t max_y =
      graph.context()->adapter_ptr()->max_compute_workgroup_count()[1];
  const uint32_t y = std::min(m_tiles, max_y);
  return {
      utils::div_up(N4, v.tile_n / 4u) * v.k_slices,
      y,
      utils::div_up(m_tiles, y)};
}

std::string
shader_name(ComputeGraph& graph, const std::string& base, const ValueRef out) {
  std::string name = base;
  add_dtype_suffix(name, graph.dtype_of(out));
  return name;
}

// A copy of the constant weight [C_out, C_in, k] rearranged on the CPU:
// pw: [C_out, C_in, 1, 1] for prepack_conv2d_gemm_weight; otherwise
// [K4, C_out, 4] with entry (k4, co, j) = weight (co, ci, t) for K index
// 4 k4 + j = t C_in + ci, zero past C_in k.
ValueRef rearranged_weight(
    ComputeGraph& graph,
    const ValueRef weight_data,
    const bool pw) {
  std::vector<int64_t> new_sizes;
  vkapi::ScalarType dtype;
  size_t nbytes;
  uint8_t* dst;
  {
    const TensorRefPtr w = graph.get_tref(weight_data);
    const std::vector<int64_t> sizes = w->sizes;
    dtype = w->dtype;
    const int64_t C_out = sizes.at(0);
    const int64_t C_in = sizes.at(1);
    const int64_t taps = sizes.at(2);
    const size_t elem = vkapi::element_size(dtype);
    const uint8_t* const src = static_cast<const uint8_t*>(w->data);
    if (pw) {
      new_sizes = {C_out, C_in, 1, 1};
      nbytes = size_t(C_out * C_in) * elem;
      dst = new uint8_t[nbytes];
      std::memcpy(dst, src, nbytes);
    } else {
      const int64_t K4 = utils::div_up_4(C_in * taps);
      new_sizes = {K4, C_out, 4};
      nbytes = size_t(K4 * C_out * 4) * elem;
      dst = new uint8_t[nbytes]();
      for (int64_t co = 0; co < C_out; ++co) {
        for (int64_t ci = 0; ci < C_in; ++ci) {
          for (int64_t t = 0; t < taps; ++t) {
            const int64_t k = t * C_in + ci;
            const int64_t to = ((k / 4) * C_out + co) * 4 + k % 4;
            const int64_t from = (co * C_in + ci) * taps + t;
            std::memcpy(dst + to * elem, src + from * elem, elem);
          }
        }
      }
    }
  }
  return graph.add_tensorref(
      new_sizes,
      dtype,
      executorch::runtime::FreeableBuffer(
          dst, nbytes, [](void*, void* data, size_t) {
            delete[] static_cast<uint8_t*>(data);
          }));
}

void resize_conv1d_simt_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = resize_args.at(0);
  const std::vector<int64_t> w = graph->sizes_of(resize_args.at(1));
  std::vector<int64_t> sizes = graph->sizes_of(out);
  const std::vector<int64_t> in_sizes = graph->sizes_of(in);
  sizes.at(0) = in_sizes.at(0);
  sizes.at(2) = calc_out_size(
      in_sizes.at(2),
      w.at(2),
      first_int(*graph, resize_args.at(2)),
      first_int(*graph, resize_args.at(3)),
      first_int(*graph, resize_args.at(4)),
      false);
  graph->virtual_resize(out, sizes);
}

} // namespace

const std::vector<Conv1dSimtVariant>& conv1d_simt_variants() {
  // {name, tile_m, tile_n, tile_k, tm, tn4, k_slices, pw}; must match
  // conv1d_simt.yaml.
  // clang-format off
  static const std::vector<Conv1dSimtVariant> variants = {
      {"conv1d_simt_t64x16_m4n1", 64, 16, 16, 4, 1, 1, false},
      {"conv1d_simt_t64x16_m4n1_sk4", 64, 16, 16, 4, 1, 4, false},
      {"conv1d_simt_t64x16_m4n1_sk8", 64, 16, 16, 4, 1, 8, false},
      {"conv1d_simt_t64x16_m4n1_sk16", 64, 16, 16, 4, 1, 16, false},
      {"conv1d_simt_t32x32_m4n1", 32, 32, 16, 4, 1, 1, false},
      {"conv1d_simt_t32x32_m4n1_sk4", 32, 32, 16, 4, 1, 4, false},
      {"conv1d_simt_t32x32_m4n1_sk8", 32, 32, 16, 4, 1, 8, false},
      {"conv1d_simt_t32x32_m4n1_sk16", 32, 32, 16, 4, 1, 16, false},
      {"conv1d_simt_t128x16_m8n1", 128, 16, 16, 8, 1, 1, false},
      {"conv1d_simt_t128x16_m8n1_sk4", 128, 16, 16, 8, 1, 4, false},
      {"conv1d_simt_t128x16_m8n1_sk8", 128, 16, 16, 8, 1, 8, false},
      {"conv1d_simt_t128x16_m8n1_sk16", 128, 16, 16, 8, 1, 16, false},
      {"conv1d_simt_t32x32_m4n1_pw", 32, 32, 16, 4, 1, 1, true},
      {"conv1d_simt_t32x32_m4n1_pw_sk4", 32, 32, 16, 4, 1, 4, true},
      {"conv1d_simt_t32x32_m4n1_pw_sk8", 32, 32, 16, 4, 1, 8, true},
      {"conv1d_simt_t32x32_m4n1_pw_sk16", 32, 32, 16, 4, 1, 16, true},
      {"conv1d_simt_t32x64_m4n2_pw", 32, 64, 16, 4, 2, 1, true},
      {"conv1d_simt_t32x64_m4n2_pw_sk4", 32, 64, 16, 4, 2, 4, true},
      {"conv1d_simt_t32x64_m4n2_pw_sk8", 32, 64, 16, 4, 2, 8, true},
      {"conv1d_simt_t32x64_m4n2_pw_sk16", 32, 64, 16, 4, 2, 16, true},
      {"conv1d_simt_t64x64_m4n2_pw", 64, 64, 16, 4, 2, 1, true},
      {"conv1d_simt_t64x64_m4n2_pw_sk4", 64, 64, 16, 4, 2, 4, true},
      {"conv1d_simt_t64x64_m4n2_pw_sk8", 64, 64, 16, 4, 2, 8, true},
      {"conv1d_simt_t64x64_m4n2_pw_sk16", 64, 64, 16, 4, 2, 16, true},
  };
  // clang-format on
  return variants;
}

const Conv1dSimtVariant* find_conv1d_simt_variant(const std::string& name) {
  for (const Conv1dSimtVariant& v : conv1d_simt_variants()) {
    const std::string base = v.name;
    if (name == base || name == base + "_half" || name == base + "_float") {
      return &v;
    }
  }
  return nullptr;
}

std::optional<std::string> conv1d_simt_ineligible(
    ComputeGraph& graph,
    const Conv1dSimtVariant& v,
    const std::vector<ValueRef>& args) {
  const auto n = [](const uint64_t x) { return std::to_string(x); };
  const ValueRef in = args.at(0);
  const ValueRef weight = args.at(1);
  const ValueRef bias = args.at(2);
  const ValueRef out = args.at(9);
  const vkapi::ScalarType dtype = graph.dtype_of(out);
  if ((dtype != vkapi::kFloat && dtype != vkapi::kHalf) ||
      graph.dtype_of(in) != dtype) {
    return "input and output are not both FP32 or both FP16";
  }
  if (!graph.val_is_tref(weight) || graph.dtype_of(weight) != dtype ||
      graph.dim_of(weight) != 3) {
    return "the weights are not a constant [C_out, C_in, k] tensor of the "
           "output's dtype";
  }
  if (!graph.val_is_none(bias) &&
      (!graph.val_is_tref(bias) || graph.dtype_of(bias) != dtype)) {
    return "the bias is not a constant of the output's dtype";
  }
  if (graph.get_int(args.at(6)) != 1) {
    return "the convolution is grouped";
  }
  if (dtype == vkapi::kHalf && !graph.float16_buffers_enabled()) {
    return "16-bit storage buffers are not enabled";
  }
  if (graph.dim_of(in) != 3 || graph.storage_type_of(in) != utils::kTexture3D ||
      graph.storage_type_of(out) != utils::kTexture3D ||
      !graph.has_standard_axis_map(in) || !graph.has_standard_axis_map(out)) {
    return "input and output are not 3D textures with the standard axis map";
  }
  const int32_t packed = v.pw ? WHCN::kHeightDim : WHCN::kChannelsDim;
  if (graph.packed_dim_of(in) != packed || graph.packed_dim_of(out) != packed) {
    return v.pw ? "input and output are not height-packed (conv1d_pw role)"
                : "input and output are not channels-packed (conv1d role)";
  }
  const std::vector<int64_t> w = graph.sizes_of(weight);
  if (w.at(1) != graph.size_at<int64_t>(-2, in)) {
    return "the weight's input channels differ from the input's";
  }
  if (v.pw &&
      (w.at(2) != 1 || first_int(graph, args.at(3)) != 1 ||
       first_int(graph, args.at(4)) != 0)) {
    return "not a kernel 1, stride 1, padding 0 convolution";
  }
  vkapi::Adapter* const adapter = graph.context()->adapter_ptr();
  if (v.wg_size() > adapter->max_compute_workgroup_invocations() ||
      v.wg_size() > adapter->max_compute_workgroup_size()[0]) {
    return "work group size " + n(v.wg_size()) + " exceeds the device limit";
  }
  if (2u * v.tile_k * (v.tile_m + v.tile_n) * 4u >
      adapter->max_compute_shared_memory_size()) {
    return "shared memory exceeds the device limit";
  }
  const auto [M, N4] = gemm_extents(graph, v, out);
  const utils::uvec3 max_count = adapter->max_compute_workgroup_count();
  if (v.k_slices > 1u) {
    const uint64_t K = v.pw ? utils::align_up_4(uint64_t(w.at(1)))
                            : uint64_t(w.at(1) * w.at(2));
    const uint64_t steps = utils::div_up(K, uint64_t(v.tile_k));
    if (steps < 2u * v.k_slices) {
      return "fewer than two K steps per slice of " + n(v.k_slices);
    }
    if (16u * v.k_slices * uint64_t(M) * N4 > graph.max_buffer_numel()) {
      return "the split-K scratch exceeds maxStorageBufferRange";
    }
    if (utils::div_up(uint64_t(M) * N4, uint64_t(kMergeThreads)) >
        max_count[0]) {
      return "the merge needs too many work groups in x";
    }
  }
  return conv2d_pw_coopmat_grid_unsupported(
      workgroups(graph, v, out), max_count);
}

void conv1d_simt_impl(
    ComputeGraph& graph,
    const Conv1dSimtVariant& v,
    const std::vector<ValueRef>& args) {
  const ValueRef in = args.at(0);
  const ValueRef weight_data = args.at(1);
  const ValueRef bias = args.at(2);
  const ValueRef out = args.at(9);
  const std::vector<int64_t> w = graph.sizes_of(weight_data);

  const ValueRef arranged = rearranged_weight(graph, weight_data, v.pw);
  const ValueRef packed_weight = v.pw
      ? prepack_conv2d_gemm_weight(graph, arranged, utils::kBuffer)
      : prepack_standard(graph, arranged, utils::kBuffer, utils::kWidthPacked);
  TmpTensor dummy_bias(
      &graph, {4}, graph.dtype_of(out), utils::kBuffer, utils::kWidthPacked);
  const bool has_bias = !graph.val_is_none(bias);
  const ValueRef packed_bias = has_bias
      ? prepack_standard(graph, bias, utils::kBuffer, utils::kWidthPacked)
      : dummy_bias.vref;

  const bool has_clamp = is_valid(args.at(7));
  const Geometry geometry{
      utils::safe_downcast<int32_t>(
          v.pw ? utils::div_up_4(w.at(1)) : utils::div_up_4(w.at(1) * w.at(2))),
      utils::safe_downcast<int32_t>(first_int(graph, args.at(3))),
      utils::safe_downcast<int32_t>(first_int(graph, args.at(4))),
      utils::safe_downcast<int32_t>(first_int(graph, args.at(5)))};
  const Epilogue epilogue{
      utils::safe_downcast<int32_t>(w.at(2)),
      has_clamp ? graph.extract_scalar<float>(args.at(7)) : 0.0f,
      has_clamp ? graph.extract_scalar<float>(args.at(8)) : 0.0f,
      utils::safe_downcast<int32_t>(v.k_slices)};
  const vkapi::SpecVarList spec_vars = {has_bias, has_clamp};
  const std::vector<ValueRef> resize_args = {
      in, weight_data, args.at(3), args.at(4), args.at(5)};
  const vkapi::ParamsBindList ubos = {
      graph.sizes_ubo(in), graph.sizes_ubo(out)};
  const std::vector<PushConstantDataInfo> push = {
      PushConstantDataInfo(&geometry, sizeof(geometry)),
      PushConstantDataInfo(&epilogue, sizeof(epilogue))};
  const Conv1dSimtVariant* const vp = &v;
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
  const std::string kernel = shader_name(graph, v.name, out);
  if (v.k_slices <= 1u) {
    graph.execute_nodes().emplace_back(new DynamicDispatchNode(
        graph,
        VK_KERNEL_FROM_STR(kernel),
        kernel_gwg,
        kernel_lwg,
        {{out, vkapi::kWrite},
         {{in, packed_weight, packed_bias}, vkapi::kRead}},
        ubos,
        push,
        spec_vars,
        resize_args,
        resize_conv1d_simt_node));
    return;
  }
  const auto [M, N4] = gemm_extents(graph, v, out);
  TmpTensor scratch(
      &graph,
      {int64_t(v.k_slices) * M * N4 * 4},
      vkapi::kFloat,
      utils::kBuffer,
      utils::kWidthPacked);
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel),
      kernel_gwg,
      kernel_lwg,
      {{out, vkapi::kWrite},
       {{in, packed_weight, packed_bias}, vkapi::kRead},
       {scratch.vref, vkapi::kReadWrite}},
      ubos,
      push,
      spec_vars,
      resize_args,
      resize_conv1d_simt_node));
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(shader_name(
          graph, v.pw ? "conv1d_simt_merge_pw" : "conv1d_simt_merge", out)),
      [vp](
          ComputeGraph* graph,
          const vkapi::ShaderInfo&,
          const std::vector<ArgGroup>& args,
          const std::vector<ValueRef>&) {
        const auto [M, N4] = gemm_extents(*graph, *vp, args.at(0).refs.at(0));
        return GlobalWorkGrid(
            {utils::div_up(M * N4, kMergeThreads) * kMergeThreads, 1u, 1u},
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
       {{in, packed_weight, packed_bias}, vkapi::kRead},
       {scratch.vref, vkapi::kReadWrite}},
      ubos,
      push,
      spec_vars,
      resize_args,
      resize_conv1d_simt_node));
}

const Conv1dSimtVariant* conv1d_simt_table_variant(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  if (!conv2d_coopmat_switch_on() || graph.dim_of(args.at(0)) != 3 ||
      !graph.val_is_tref(args.at(1)) || graph.dim_of(args.at(1)) != 3) {
    return nullptr;
  }
  const vkapi::ScalarType dtype = graph.dtype_of(args.at(9));
  if (dtype != vkapi::kFloat && dtype != vkapi::kHalf) {
    return nullptr;
  }
  const std::vector<int64_t> w = graph.sizes_of(args.at(1));
  const char* const name = conv2d_coopmat_table_lookup(
      graph,
      dtype == vkapi::kHalf ? "fp16" : "fp32",
      {w.at(1),
       w.at(0),
       1,
       w.at(2),
       1,
       first_int(graph, args.at(3)),
       0,
       first_int(graph, args.at(4)),
       1,
       first_int(graph, args.at(5)),
       1,
       graph.size_at<int64_t>(-1, args.at(0))});
  const Conv1dSimtVariant* const v =
      name != nullptr ? find_conv1d_simt_variant(name) : nullptr;
  if (v == nullptr || conv1d_simt_ineligible(graph, *v, args)) {
    return nullptr;
  }
  return v;
}

} // namespace vkcompute
