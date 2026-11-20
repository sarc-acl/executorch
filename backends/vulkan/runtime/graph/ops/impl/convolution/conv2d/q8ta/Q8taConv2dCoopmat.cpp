/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/q8ta/Q8taConv2dCoopmat.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2d.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/q8ta/Q8taConv2dCommon.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/q8ta/Q8taConv2dPW.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/qw/QuantizedLinear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>

#include <algorithm>
#include <limits>

namespace vkcompute {

namespace {

// Work group size of q8ta_conv2d_pw_coopmat_merge.
constexpr uint32_t kMergeThreads = 64;
constexpr const char* kMergeShader = "q8ta_conv2d_pw_coopmat_merge";

struct Geometry final {
  int64_t in_channels;
  int64_t out_channels;
  int64_t kernel_h;
  int64_t kernel_w;
  int64_t stride_h;
  int64_t stride_w;
  int64_t padding_h;
  int64_t padding_w;
  int64_t dilation_h;
  int64_t dilation_w;
  int64_t groups;
};

Geometry geometry_of(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const auto pair = [&graph](const ValueRef ref) {
    const auto list = graph.get_int_list(ref);
    return std::pair<int64_t, int64_t>{
        list->at(0), list->size() > 1 ? list->at(1) : list->at(0)};
  };
  const auto [kh, kw] = pair(args.at(9));
  const auto [sh, sw] = pair(args.at(10));
  const auto [ph, pw] = pair(args.at(11));
  const auto [dh, dw] = pair(args.at(12));
  return {
      graph.size_at<int64_t>(-3, args.at(0)),
      graph.size_at<int64_t>(-3, args.at(15)),
      kh,
      kw,
      sh,
      sw,
      ph,
      pw,
      dh,
      dw,
      graph.extract_scalar<int64_t>(args.at(13))};
}

// Rows of the GEMM: output pixels padded to whole quads of 4 widths.
uint32_t gemm_rows(ComputeGraph& graph, const ValueRef out) {
  return 4u * utils::div_up_4(graph.size_at<uint32_t>(-1, out)) *
      graph.size_at<uint32_t>(-2, out) * graph.size_at<uint32_t>(-4, out);
}

utils::uvec3 workgroups(
    ComputeGraph& graph,
    const Q8taConv2dCoopmatVariant& v,
    const ValueRef out) {
  const uint32_t m_tiles = utils::div_up(gemm_rows(graph, out), v.tile_m);
  const uint32_t max_y =
      graph.context()->adapter_ptr()->max_compute_workgroup_count()[1];
  const uint32_t y = std::min(m_tiles, max_y);
  return {
      utils::div_up(graph.size_at<uint32_t>(-3, out), v.tile_n) * v.k_slices,
      y,
      utils::div_up(m_tiles, y)};
}

uint32_t k4_of(const Q8taConv2dCoopmatVariant& v, const Geometry& g) {
  return v.direct
      ? utils::safe_downcast<uint32_t>(
            utils::div_up_4(g.kernel_h * g.kernel_w * g.in_channels))
      : utils::safe_downcast<uint32_t>(utils::div_up_4(g.in_channels));
}

bool texture_weight_fits(
    ComputeGraph& graph,
    const int64_t K,
    const int64_t N) {
  // The texture/buffer choice of the stock prepacks.
  const int64_t max_extent =
      graph.context()->adapter_ptr()->max_texture2d_dim();
  return utils::div_up_4(N) * 4 <= max_extent * 4 &&
      utils::div_up_4(K) <= max_extent;
}

void resize_q8ta_conv2d_coopmat_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const std::vector<int64_t> out_hw = calc_out_sizes_hw(
      *graph,
      graph->sizes_of(resize_args.at(0)),
      resize_args.at(1),
      /*kernel_size_only=*/true,
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
}

} // namespace

uint32_t Q8taConv2dCoopmatVariant::shared_bytes() const {
  // Mirrors the shared pool of q8ta_conv2d_pw_coopmat.glsl, in uints.
  const uint32_t rs = mma_k / 4u;
  const uint32_t slabs = tile_k / mma_k;
  const uint32_t pool_ab = 2u * slabs * (tile_m + tile_n) * rs;
  const uint32_t pool_c = sg_grid_y * mma_m * tile_n;
  return 4u * std::max(pool_ab, pool_c);
}

const std::vector<Q8taConv2dCoopmatVariant>& q8ta_conv2d_coopmat_variants() {
  // {name, mma_m, mma_n, mma_k, tile_m, tile_n, tile_k, sg_grid_x, sg_grid_y,
  //  subgroup_size, k_slices, direct, bytes}; must match
  //  q8ta_conv2d_pw_coopmat.yaml.
  // clang-format off
  static const std::vector<Q8taConv2dCoopmatVariant> variants = {
      {"q8ta_conv2d_pw_coopmat_xe2_t32x32", 8, 16, 32, 32, 32, 32, 2, 2, 16, 1, false, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t32x32_sk2", 8, 16, 32, 32, 32, 32, 2, 2, 16, 2, false, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t32x32_sk4", 8, 16, 32, 32, 32, 32, 2, 2, 16, 4, false, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t32x32_sk8", 8, 16, 32, 32, 32, 32, 2, 2, 16, 8, false, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t32x32_dg", 8, 16, 32, 32, 32, 32, 2, 2, 16, 1, true, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t32x32_dg_sk2", 8, 16, 32, 32, 32, 32, 2, 2, 16, 2, true, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t32x32_dg_sk4", 8, 16, 32, 32, 32, 32, 2, 2, 16, 4, true, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t32x32_dg_sk8", 8, 16, 32, 32, 32, 32, 2, 2, 16, 8, true, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t32x32_dgb", 8, 16, 32, 32, 32, 32, 2, 2, 16, 1, true, true},
      {"q8ta_conv2d_pw_coopmat_xe2_t32x32_dgb_sk2", 8, 16, 32, 32, 32, 32, 2, 2, 16, 2, true, true},
      {"q8ta_conv2d_pw_coopmat_xe2_t32x32_dgb_sk4", 8, 16, 32, 32, 32, 32, 2, 2, 16, 4, true, true},
      {"q8ta_conv2d_pw_coopmat_xe2_t64x64", 8, 16, 32, 64, 64, 32, 2, 2, 16, 1, false, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t64x64_sk2", 8, 16, 32, 64, 64, 32, 2, 2, 16, 2, false, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t64x64_sk4", 8, 16, 32, 64, 64, 32, 2, 2, 16, 4, false, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t64x64_sk8", 8, 16, 32, 64, 64, 32, 2, 2, 16, 8, false, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t64x64_dg", 8, 16, 32, 64, 64, 32, 2, 2, 16, 1, true, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t64x64_dg_sk2", 8, 16, 32, 64, 64, 32, 2, 2, 16, 2, true, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t64x64_dg_sk4", 8, 16, 32, 64, 64, 32, 2, 2, 16, 4, true, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t64x64_dg_sk8", 8, 16, 32, 64, 64, 32, 2, 2, 16, 8, true, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t64x64_dgb", 8, 16, 32, 64, 64, 32, 2, 2, 16, 1, true, true},
      {"q8ta_conv2d_pw_coopmat_xe2_t64x64_dgb_sk2", 8, 16, 32, 64, 64, 32, 2, 2, 16, 2, true, true},
      {"q8ta_conv2d_pw_coopmat_xe2_t64x64_dgb_sk4", 8, 16, 32, 64, 64, 32, 2, 2, 16, 4, true, true},
      {"q8ta_conv2d_pw_coopmat_xe2_t128x64", 8, 16, 32, 128, 64, 32, 2, 4, 16, 1, false, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t128x64_sk2", 8, 16, 32, 128, 64, 32, 2, 4, 16, 2, false, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t128x64_sk4", 8, 16, 32, 128, 64, 32, 2, 4, 16, 4, false, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t128x64_dg", 8, 16, 32, 128, 64, 32, 2, 4, 16, 1, true, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t128x64_dg_sk2", 8, 16, 32, 128, 64, 32, 2, 4, 16, 2, true, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t128x64_dg_sk4", 8, 16, 32, 128, 64, 32, 2, 4, 16, 4, true, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t64x128", 8, 16, 32, 64, 128, 32, 4, 2, 16, 1, false, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t64x128_sk2", 8, 16, 32, 64, 128, 32, 4, 2, 16, 2, false, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t64x128_sk4", 8, 16, 32, 64, 128, 32, 4, 2, 16, 4, false, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t64x128_dg", 8, 16, 32, 64, 128, 32, 4, 2, 16, 1, true, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t64x128_dg_sk2", 8, 16, 32, 64, 128, 32, 4, 2, 16, 2, true, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t64x128_dg_sk4", 8, 16, 32, 64, 128, 32, 4, 2, 16, 4, true, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t128x128", 8, 16, 32, 128, 128, 32, 4, 4, 16, 1, false, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t128x128_sk2", 8, 16, 32, 128, 128, 32, 4, 4, 16, 2, false, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t128x128_sk4", 8, 16, 32, 128, 128, 32, 4, 4, 16, 4, false, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t128x128_dg", 8, 16, 32, 128, 128, 32, 4, 4, 16, 1, true, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t128x128_dg_sk2", 8, 16, 32, 128, 128, 32, 4, 4, 16, 2, true, false},
      {"q8ta_conv2d_pw_coopmat_xe2_t128x128_dg_sk4", 8, 16, 32, 128, 128, 32, 4, 4, 16, 4, true, false},
      {"q8ta_conv2d_pw_coopmat_amd_t32x32", 16, 16, 16, 32, 32, 32, 1, 1, 32, 1, false, false},
      {"q8ta_conv2d_pw_coopmat_amd_t32x32_sk2", 16, 16, 16, 32, 32, 32, 1, 1, 32, 2, false, false},
      {"q8ta_conv2d_pw_coopmat_amd_t32x32_sk4", 16, 16, 16, 32, 32, 32, 1, 1, 32, 4, false, false},
      {"q8ta_conv2d_pw_coopmat_amd_t32x32_sk8", 16, 16, 16, 32, 32, 32, 1, 1, 32, 8, false, false},
      {"q8ta_conv2d_pw_coopmat_amd_t32x32_dg", 16, 16, 16, 32, 32, 32, 1, 1, 32, 1, true, false},
      {"q8ta_conv2d_pw_coopmat_amd_t32x32_dg_sk2", 16, 16, 16, 32, 32, 32, 1, 1, 32, 2, true, false},
      {"q8ta_conv2d_pw_coopmat_amd_t32x32_dg_sk4", 16, 16, 16, 32, 32, 32, 1, 1, 32, 4, true, false},
      {"q8ta_conv2d_pw_coopmat_amd_t32x32_dg_sk8", 16, 16, 16, 32, 32, 32, 1, 1, 32, 8, true, false},
      {"q8ta_conv2d_pw_coopmat_amd_t32x32_dgb", 16, 16, 16, 32, 32, 32, 1, 1, 32, 1, true, true},
      {"q8ta_conv2d_pw_coopmat_amd_t32x32_dgb_sk2", 16, 16, 16, 32, 32, 32, 1, 1, 32, 2, true, true},
      {"q8ta_conv2d_pw_coopmat_amd_t32x32_dgb_sk4", 16, 16, 16, 32, 32, 32, 1, 1, 32, 4, true, true},
      {"q8ta_conv2d_pw_coopmat_amd_t64x64", 16, 16, 16, 64, 64, 32, 2, 2, 32, 1, false, false},
      {"q8ta_conv2d_pw_coopmat_amd_t64x64_sk2", 16, 16, 16, 64, 64, 32, 2, 2, 32, 2, false, false},
      {"q8ta_conv2d_pw_coopmat_amd_t64x64_sk4", 16, 16, 16, 64, 64, 32, 2, 2, 32, 4, false, false},
      {"q8ta_conv2d_pw_coopmat_amd_t64x64_sk8", 16, 16, 16, 64, 64, 32, 2, 2, 32, 8, false, false},
      {"q8ta_conv2d_pw_coopmat_amd_t64x64_dg", 16, 16, 16, 64, 64, 32, 2, 2, 32, 1, true, false},
      {"q8ta_conv2d_pw_coopmat_amd_t64x64_dg_sk2", 16, 16, 16, 64, 64, 32, 2, 2, 32, 2, true, false},
      {"q8ta_conv2d_pw_coopmat_amd_t64x64_dg_sk4", 16, 16, 16, 64, 64, 32, 2, 2, 32, 4, true, false},
      {"q8ta_conv2d_pw_coopmat_amd_t64x64_dg_sk8", 16, 16, 16, 64, 64, 32, 2, 2, 32, 8, true, false},
      {"q8ta_conv2d_pw_coopmat_amd_t64x64_dgb", 16, 16, 16, 64, 64, 32, 2, 2, 32, 1, true, true},
      {"q8ta_conv2d_pw_coopmat_amd_t64x64_dgb_sk2", 16, 16, 16, 64, 64, 32, 2, 2, 32, 2, true, true},
      {"q8ta_conv2d_pw_coopmat_amd_t64x64_dgb_sk4", 16, 16, 16, 64, 64, 32, 2, 2, 32, 4, true, true},
      {"q8ta_conv2d_pw_coopmat_amd_t128x64", 16, 16, 16, 128, 64, 32, 2, 2, 32, 1, false, false},
      {"q8ta_conv2d_pw_coopmat_amd_t128x64_sk2", 16, 16, 16, 128, 64, 32, 2, 2, 32, 2, false, false},
      {"q8ta_conv2d_pw_coopmat_amd_t128x64_sk4", 16, 16, 16, 128, 64, 32, 2, 2, 32, 4, false, false},
      {"q8ta_conv2d_pw_coopmat_amd_t128x64_dg", 16, 16, 16, 128, 64, 32, 2, 2, 32, 1, true, false},
      {"q8ta_conv2d_pw_coopmat_amd_t128x64_dg_sk2", 16, 16, 16, 128, 64, 32, 2, 2, 32, 2, true, false},
      {"q8ta_conv2d_pw_coopmat_amd_t128x64_dg_sk4", 16, 16, 16, 128, 64, 32, 2, 2, 32, 4, true, false},
      {"q8ta_conv2d_pw_coopmat_amd_t128x128", 16, 16, 16, 128, 128, 32, 4, 2, 32, 1, false, false},
      {"q8ta_conv2d_pw_coopmat_amd_t128x128_sk2", 16, 16, 16, 128, 128, 32, 4, 2, 32, 2, false, false},
      {"q8ta_conv2d_pw_coopmat_amd_t128x128_sk4", 16, 16, 16, 128, 128, 32, 4, 2, 32, 4, false, false},
      {"q8ta_conv2d_pw_coopmat_amd_t128x128_dg", 16, 16, 16, 128, 128, 32, 4, 2, 32, 1, true, false},
      {"q8ta_conv2d_pw_coopmat_amd_t128x128_dg_sk2", 16, 16, 16, 128, 128, 32, 4, 2, 32, 2, true, false},
      {"q8ta_conv2d_pw_coopmat_amd_t128x128_dg_sk4", 16, 16, 16, 128, 128, 32, 4, 2, 32, 4, true, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t32x32", 16, 16, 32, 32, 32, 32, 1, 1, 32, 1, false, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t32x32_sk2", 16, 16, 32, 32, 32, 32, 1, 1, 32, 2, false, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t32x32_sk4", 16, 16, 32, 32, 32, 32, 1, 1, 32, 4, false, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t32x32_sk8", 16, 16, 32, 32, 32, 32, 1, 1, 32, 8, false, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t32x32_dg", 16, 16, 32, 32, 32, 32, 1, 1, 32, 1, true, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t32x32_dg_sk2", 16, 16, 32, 32, 32, 32, 1, 1, 32, 2, true, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t32x32_dg_sk4", 16, 16, 32, 32, 32, 32, 1, 1, 32, 4, true, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t32x32_dg_sk8", 16, 16, 32, 32, 32, 32, 1, 1, 32, 8, true, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t32x32_dgb", 16, 16, 32, 32, 32, 32, 1, 1, 32, 1, true, true},
      {"q8ta_conv2d_pw_coopmat_nvidia_t32x32_dgb_sk2", 16, 16, 32, 32, 32, 32, 1, 1, 32, 2, true, true},
      {"q8ta_conv2d_pw_coopmat_nvidia_t32x32_dgb_sk4", 16, 16, 32, 32, 32, 32, 1, 1, 32, 4, true, true},
      {"q8ta_conv2d_pw_coopmat_nvidia_t64x64", 16, 16, 32, 64, 64, 32, 2, 2, 32, 1, false, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t64x64_sk2", 16, 16, 32, 64, 64, 32, 2, 2, 32, 2, false, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t64x64_sk4", 16, 16, 32, 64, 64, 32, 2, 2, 32, 4, false, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t64x64_sk8", 16, 16, 32, 64, 64, 32, 2, 2, 32, 8, false, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t64x64_dg", 16, 16, 32, 64, 64, 32, 2, 2, 32, 1, true, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t64x64_dg_sk2", 16, 16, 32, 64, 64, 32, 2, 2, 32, 2, true, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t64x64_dg_sk4", 16, 16, 32, 64, 64, 32, 2, 2, 32, 4, true, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t64x64_dg_sk8", 16, 16, 32, 64, 64, 32, 2, 2, 32, 8, true, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t64x64_dgb", 16, 16, 32, 64, 64, 32, 2, 2, 32, 1, true, true},
      {"q8ta_conv2d_pw_coopmat_nvidia_t64x64_dgb_sk2", 16, 16, 32, 64, 64, 32, 2, 2, 32, 2, true, true},
      {"q8ta_conv2d_pw_coopmat_nvidia_t64x64_dgb_sk4", 16, 16, 32, 64, 64, 32, 2, 2, 32, 4, true, true},
      {"q8ta_conv2d_pw_coopmat_nvidia_t128x64", 16, 16, 32, 128, 64, 32, 2, 2, 32, 1, false, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t128x64_sk2", 16, 16, 32, 128, 64, 32, 2, 2, 32, 2, false, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t128x64_sk4", 16, 16, 32, 128, 64, 32, 2, 2, 32, 4, false, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t128x64_dg", 16, 16, 32, 128, 64, 32, 2, 2, 32, 1, true, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t128x64_dg_sk2", 16, 16, 32, 128, 64, 32, 2, 2, 32, 2, true, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t128x64_dg_sk4", 16, 16, 32, 128, 64, 32, 2, 2, 32, 4, true, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t128x128", 16, 16, 32, 128, 128, 32, 4, 2, 32, 1, false, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t128x128_sk2", 16, 16, 32, 128, 128, 32, 4, 2, 32, 2, false, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t128x128_sk4", 16, 16, 32, 128, 128, 32, 4, 2, 32, 4, false, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t128x128_dg", 16, 16, 32, 128, 128, 32, 4, 2, 32, 1, true, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t128x128_dg_sk2", 16, 16, 32, 128, 128, 32, 4, 2, 32, 2, true, false},
      {"q8ta_conv2d_pw_coopmat_nvidia_t128x128_dg_sk4", 16, 16, 32, 128, 128, 32, 4, 2, 32, 4, true, false},
  };
  // clang-format on
  return variants;
}

const Q8taConv2dCoopmatVariant* find_q8ta_conv2d_coopmat_variant(
    const std::string& name) {
  for (const Q8taConv2dCoopmatVariant& v : q8ta_conv2d_coopmat_variants()) {
    if (name == v.name) {
      return &v;
    }
  }
  return nullptr;
}

std::optional<std::string> q8ta_conv2d_coopmat_ineligible(
    ComputeGraph& graph,
    const Q8taConv2dCoopmatVariant& v,
    const std::vector<ValueRef>& args) {
  const auto n = [](const uint64_t x) { return std::to_string(x); };
  if (args.size() != 16u) {
    return "expected the 16 arguments of q8ta_conv2d";
  }
  const ValueRef in = args.at(0);
  const ValueRef weight = args.at(3);
  const ValueRef out = args.at(15);
  if (graph.dtype_of(in) != vkapi::kInt8x4 ||
      graph.dtype_of(out) != vkapi::kInt8x4 || !graph.is_buffer_storage(in) ||
      !graph.is_buffer_storage(out)) {
    return "input and output are not int8x4 buffers";
  }
  const api::PackedDimInfo& in_info = graph.packed_dim_info_of(in);
  if (v.direct ? !q8ta_conv2d_check_packed_dim_info(in_info)
               : !q8ta_conv2d_check_4w4c_packed_dim_info(in_info)) {
    return v.direct ? "the input is not channels-packed 4C1W or 4W4C"
                    : "the input is not 4W4C";
  }
  if (!q8ta_conv2d_check_packed_dim_info(graph.packed_dim_info_of(out))) {
    return "the output is not channels-packed 4C1W or 4W4C";
  }
  if (!graph.val_is_tref(weight) || graph.dtype_of(weight) != vkapi::kChar ||
      graph.dim_of(weight) != 2) {
    return "the weights are not a constant int8 [Cout, K] matrix";
  }
  const std::string activation = graph.extract_string(args.at(14));
  if (activation != "none" && activation != "relu") {
    return "activation is neither none nor relu";
  }
  const Geometry g = geometry_of(graph, args);
  if (g.groups != 1) {
    return "the convolution is grouped";
  }
  const int64_t K = graph.size_at<int64_t>(-1, weight);
  if (v.direct) {
    if (!v.bytes && g.in_channels % 4 != 0) {
      return "input channels " + n(g.in_channels) +
          " not a multiple of 4 (use a _dgb variant)";
    }
    if (K != utils::align_up_4(g.kernel_h * g.kernel_w * g.in_channels)) {
      return "the weight rows are not kernel x input channels";
    }
  } else {
    if (g.kernel_h != 1 || g.kernel_w != 1 || g.stride_h != 1 ||
        g.stride_w != 1 || g.padding_h != 0 || g.padding_w != 0) {
      return "not a 1x1, stride 1, padding 0 convolution (use a _dg variant)";
    }
  }
  if (!texture_weight_fits(graph, K, g.out_channels)) {
    return "the packed weights do not fit a 2D texture";
  }
#ifdef VK_KHR_cooperative_matrix
  vkapi::Adapter* const adapter = graph.context()->adapter_ptr();
  if (!adapter->supports_cooperative_matrix()) {
    return "the device does not support VK_KHR_cooperative_matrix";
  }
  if (!adapter->supports_cooperative_matrix_shape(
          v.mma_m,
          v.mma_n,
          v.mma_k,
          VK_COMPONENT_TYPE_SINT8_KHR,
          VK_COMPONENT_TYPE_SINT8_KHR,
          VK_COMPONENT_TYPE_SINT32_KHR,
          VK_COMPONENT_TYPE_SINT32_KHR)) {
    return "the device does not enumerate the cooperative matrix " +
        n(v.mma_m) + "x" + n(v.mma_n) + "x" + n(v.mma_k) +
        " sint8 -> sint32 at subgroup scope";
  }
  if (!adapter->supports_required_subgroup_size_for_compute() ||
      v.subgroup_size < adapter->min_subgroup_size() ||
      v.subgroup_size > adapter->max_subgroup_size()) {
    return "the device cannot require subgroup size " + n(v.subgroup_size);
  }
  if (const auto reason = conv2d_pw_coopmat_limits_unsupported(
          // Same limits as the fp16 variants.
          Conv2dPwCoopmatVariant{
              v.name,
              v.mma_m,
              v.mma_n,
              v.mma_k,
              v.tile_m,
              v.tile_n,
              v.tile_k,
              v.sg_grid_x,
              v.sg_grid_y,
              v.subgroup_size,
              false,
              false},
          {adapter->max_compute_workgroup_invocations(),
           adapter->max_compute_workgroup_size()[0],
           adapter->max_compute_workgroup_subgroups(),
           std::numeric_limits<uint32_t>::max()})) {
    return reason;
  }
  if (v.shared_bytes() > adapter->max_compute_shared_memory_size()) {
    return "shared memory " + n(v.shared_bytes()) +
        " bytes exceeds maxComputeSharedMemorySize " +
        n(adapter->max_compute_shared_memory_size());
  }
  for (const uint32_t capability :
       VK_KERNEL_FROM_STR(v.name).required_capabilities) {
    if (!adapter->supports_spirv_capability(capability)) {
      return "the device lacks SPIR-V capability " + n(capability);
    }
  }
#else
  return "built without VK_KHR_cooperative_matrix";
#endif /* VK_KHR_cooperative_matrix */
  const utils::uvec3 max_count =
      graph.context()->adapter_ptr()->max_compute_workgroup_count();
  if (v.k_slices > 1u) {
    const uint32_t steps = utils::div_up(4u * k4_of(v, g), v.tile_k);
    if (steps < 2u * v.k_slices) {
      return "fewer than two K steps per slice of " + n(v.k_slices) + " (" +
          n(steps) + " K steps)";
    }
    const uint64_t m_pad =
        utils::align_up(uint64_t(gemm_rows(graph, out)), uint64_t(v.tile_m));
    const uint64_t n_pad =
        utils::align_up(uint64_t(g.out_channels), uint64_t(v.tile_n));
    if (4u * v.k_slices * m_pad * n_pad > graph.max_buffer_numel()) {
      return "the split-K scratch exceeds maxStorageBufferRange";
    }
    if (utils::div_up(
            m_pad * utils::div_up_4(uint64_t(g.out_channels)),
            uint64_t(kMergeThreads)) > max_count[0]) {
      return "the merge needs too many work groups in x";
    }
  }
  return conv2d_pw_coopmat_grid_unsupported(
      workgroups(graph, v, out), max_count);
}

void q8ta_conv2d_coopmat_impl(
    ComputeGraph& graph,
    const Q8taConv2dCoopmatVariant& v,
    const std::vector<ValueRef>& args,
    const bool direct_requant) {
  const ValueRef in = args.at(0);
  const ValueRef weight_data = args.at(3);
  const ValueRef bias_data = args.at(8);
  const ValueRef out = args.at(15);
  const Geometry g = geometry_of(graph, args);

  QuantizationConfig weight_quant_config(8, kPerChannel, {});
  const ValueRef packed_weight = v.direct
      ? prepack_quantized_linear_weight(graph, weight_quant_config, weight_data)
      : prepack_quantized_conv2d_pw_weight(
            graph, weight_quant_config, weight_data, in, out, false);
  VK_CHECK_COND(graph.storage_type_of(packed_weight) == utils::kTexture2D);
  const ValueRef packed_weight_sums =
      prepack_standard(graph, args.at(4), utils::kBuffer, utils::kWidthPacked);
  const ValueRef packed_weight_scales =
      prepack_standard(graph, args.at(5), utils::kBuffer, utils::kWidthPacked);
  TmpTensor dummy_bias(
      &graph,
      {},
      graph.dtype_of(args.at(5)),
      utils::kBuffer,
      utils::kWidthPacked);
  ValueRef packed_bias = dummy_bias.vref;
  if (graph.val_is_not_none(bias_data)) {
    packed_bias =
        prepack_standard(graph, bias_data, utils::kBuffer, utils::kWidthPacked);
  }

  const float input_scale = graph.extract_scalar<float>(args.at(1));
  const int32_t input_zp = graph.extract_scalar<int32_t>(args.at(2));
  const float output_inv_scale = 1.0f / graph.extract_scalar<float>(args.at(6));
  const int32_t output_zp = graph.extract_scalar<int32_t>(args.at(7));
  const int32_t K4 = utils::safe_downcast<int32_t>(k4_of(v, g));
  const int32_t M_pad = utils::safe_downcast<int32_t>(
      utils::align_up(gemm_rows(graph, out), v.tile_m));
  const int32_t N_pad = utils::safe_downcast<int32_t>(
      utils::align_up(uint32_t(g.out_channels), v.tile_n));
  const int32_t k_slices = utils::safe_downcast<int32_t>(v.k_slices);
  const std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&input_scale, sizeof(input_scale)),
      PushConstantDataInfo(&input_zp, sizeof(input_zp)),
      PushConstantDataInfo(&output_inv_scale, sizeof(output_inv_scale)),
      PushConstantDataInfo(&output_zp, sizeof(output_zp)),
      PushConstantDataInfo(&K4, sizeof(K4)),
      PushConstantDataInfo(&M_pad, sizeof(M_pad)),
      PushConstantDataInfo(&N_pad, sizeof(N_pad)),
      PushConstantDataInfo(&k_slices, sizeof(k_slices)),
  };
  const auto i32 = [](const int64_t x) {
    return utils::safe_downcast<int32_t>(x);
  };
  const vkapi::SpecVarList spec_vars = {
      graph.val_is_none(bias_data) ? 0u : 1u,
      static_cast<uint32_t>(
          activation_type_from_string(graph.extract_string(args.at(14)))),
      graph.hashed_layout_of(out),
      graph.hashed_layout_of(in),
      i32(g.in_channels / 4),
      i32(g.kernel_h),
      i32(g.kernel_w),
      i32(g.stride_h),
      i32(g.stride_w),
      i32(g.padding_h),
      i32(g.padding_w),
      i32(g.dilation_h),
      i32(g.dilation_w),
      i32(g.in_channels),
      direct_requant ? 1u : 0u,
  };
  const vkapi::ParamsBindList params = {
      graph.buffer_meta_ubo(out), graph.buffer_meta_ubo(in)};
  const std::vector<ValueRef> resize_args = {
      in, args.at(9), args.at(10), args.at(11), args.at(12)};
  const std::string name = v.name;

  const auto kernel_gwg = [&v](
                              ComputeGraph* graph,
                              const vkapi::ShaderInfo&,
                              const std::vector<ArgGroup>& args,
                              const std::vector<ValueRef>&) {
    const utils::uvec3 grid = workgroups(*graph, v, args.at(0).refs.at(0));
    return GlobalWorkGrid(
        {grid[0] * v.wg_size(), grid[1], grid[2]}, kExplicitWorkGrid);
  };
  const auto kernel_lwg = [&v](
                              ComputeGraph*,
                              const vkapi::ShaderInfo&,
                              const GlobalWorkGrid&,
                              const std::vector<ArgGroup>&,
                              const std::vector<ValueRef>&) {
    return LocalWorkGroup(v.wg_size(), 1u, 1u);
  };

  if (v.k_slices <= 1u) {
    graph.execute_nodes().emplace_back(new DynamicDispatchNode(
        graph,
        VK_KERNEL_FROM_STR(name),
        kernel_gwg,
        kernel_lwg,
        {{out, vkapi::kWrite},
         {{in,
           packed_weight,
           packed_weight_sums,
           packed_weight_scales,
           packed_bias},
          vkapi::kRead}},
        params,
        push_constants,
        spec_vars,
        resize_args,
        resize_q8ta_conv2d_coopmat_node));
    return;
  }

  TmpTensor scratch(
      &graph,
      {int64_t(k_slices) * M_pad * N_pad},
      vkapi::kInt,
      utils::kBuffer,
      utils::kWidthPacked);
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(name),
      kernel_gwg,
      kernel_lwg,
      {{out, vkapi::kWrite},
       {{in,
         packed_weight,
         packed_weight_sums,
         packed_weight_scales,
         packed_bias},
        vkapi::kRead},
       {scratch.vref, vkapi::kReadWrite}},
      params,
      push_constants,
      spec_vars,
      resize_args,
      resize_q8ta_conv2d_coopmat_node));
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kMergeShader),
      [](ComputeGraph* graph,
         const vkapi::ShaderInfo&,
         const std::vector<ArgGroup>& args,
         const std::vector<ValueRef>&) {
        const ValueRef out = args.at(0).refs.at(0);
        const uint32_t ints = gemm_rows(*graph, out) *
            utils::div_up_4(graph->size_at<uint32_t>(-3, out));
        return GlobalWorkGrid(
            {utils::div_up(ints, kMergeThreads) * kMergeThreads, 1u, 1u},
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
         packed_weight_sums,
         packed_weight_scales,
         packed_bias},
        vkapi::kRead},
       {scratch.vref, vkapi::kReadWrite}},
      params,
      push_constants,
      spec_vars,
      resize_args,
      resize_q8ta_conv2d_coopmat_node));
}

const Q8taConv2dCoopmatVariant* q8ta_conv2d_coopmat_table_variant(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  if (!conv2d_coopmat_switch_on() || args.size() != 16u ||
      graph.dim_of(args.at(0)) != 4 ||
      graph.size_at<int64_t>(-4, args.at(0)) != 1) {
    return nullptr;
  }
  const Geometry g = geometry_of(graph, args);
  const char* const name = conv2d_coopmat_table_lookup(
      graph,
      "int8",
      {g.in_channels,
       g.out_channels,
       g.kernel_h,
       g.kernel_w,
       g.stride_h,
       g.stride_w,
       g.padding_h,
       g.padding_w,
       g.dilation_h,
       g.dilation_w,
       graph.size_at<int64_t>(-2, args.at(0)),
       graph.size_at<int64_t>(-1, args.at(0))});
  const Q8taConv2dCoopmatVariant* const v =
      name != nullptr ? find_q8ta_conv2d_coopmat_variant(name) : nullptr;
  if (v == nullptr || q8ta_conv2d_coopmat_ineligible(graph, *v, args)) {
    return nullptr;
  }
  return v;
}

} // namespace vkcompute
