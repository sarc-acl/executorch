/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/GemmCoopmat.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/GemmCommon.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

// ── Linear coopmat ──

static vkapi::ShaderInfo pick_linear_coopmat_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  bool has_bias = graph->get_bool(resize_args.at(1));
  std::string kernel_name = has_bias ? "linear_coopmat_bias" : "linear_coopmat";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  return VK_KERNEL_FROM_STR(kernel_name);
}

static utils::uvec3 pick_linear_coopmat_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const auto out_sizes = graph->sizes_of(out);
  uint32_t M = out_sizes.at(out_sizes.size() - 2);
  uint32_t N = out_sizes.at(out_sizes.size() - 1);
  uint32_t num_tiles_n = utils::div_up(N, kCoopmatTileN);
  uint32_t num_tiles_m = utils::div_up(M, kCoopmatTileM);
  // Each workgroup processes one WG_TILE_M x WG_TILE_N output tile via
  // cooperative-matrix MMAs across its 4 subgroups. We want the dispatch
  // to launch exactly num_tiles_n x num_tiles_m workgroups.
  //
  // The framework computes the group count as
  //   group_count = div_up(global_wg_size, local_wg_size)
  // (see Context.cpp + Command.cpp). With local_wg = (kCoopmatInvocations,
  // 1, 1), multiplying num_tiles_n by kCoopmatInvocations cancels the
  // div, yielding group_count.x = num_tiles_n.
  return {num_tiles_n * kCoopmatInvocations, num_tiles_m, 1};
}

static utils::uvec3 pick_linear_coopmat_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)global_workgroup_size;
  (void)args;
  (void)resize_args;
  return {kCoopmatInvocations, 1, 1};
}

void add_linear_coopmat_node(
    ComputeGraph& graph,
    const ValueRef input,
    const ValueRef packed_weight,
    const ValueRef packed_bias,
    bool has_bias,
    const ValueRef out,
    int32_t weight_B) {
  // weight_B must be 1 — the shader is 2D and dispatch z-dim is hardcoded to
  // 1, so batched weights would only compute the first batch. The
  // is_coopmat_eligible() gate rejects batched outputs upstream; this
  // VK_CHECK_COND turns silent miscompute into a hard fail for any direct
  // caller that bypasses the gate.
  VK_CHECK_COND(
      weight_B == 1, "linear_coopmat does not support batched weights");
  VK_CHECK_COND(graph.packed_dim_of(input) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(out) == WHCN::kWidthDim);
  VK_CHECK_COND(
      graph.storage_type_of(out) == utils::kBuffer,
      "linear_coopmat requires buffer storage");

  std::vector<int64_t> out_sizes = graph.sizes_of(out);
  int32_t orig_N = utils::safe_downcast<int32_t>(out_sizes.back());
  ValueRef orig_N_ref = graph.add_scalar(static_cast<int64_t>(orig_N));
  ValueRef has_bias_ref = graph.add_scalar(has_bias);

  // K-chunk trip count and output width N as spec constants — the Xclipse
  // driver crashes on UBO-derived coopmat loop bounds and miscompiles
  // UBO-derived coopMatStore offsets/strides (see coopmat_mm.glsl). Restored
  // 2026-07 after being silently dropped by 5426101bf4 ("Add int4
  // cooperative-matrix dispatch for quantized linear") -- that commit's
  // message never mentions touching this function; the removal was an
  // unintentional side effect (found while porting yanwen/quant-dev-active's
  // SDPA coopmat work, which depends on this same workaround still being
  // present in add_matmul_coopmat_node below).
  const int32_t K = graph.size_at<int32_t>(-1, input);
  VK_CHECK_COND(K % static_cast<int32_t>(kCoopmatTileK) == 0);
  const int32_t num_k_chunks = K / static_cast<int32_t>(kCoopmatTileK);

  std::vector<ValueRef> read_inputs = {input, packed_weight};
  if (has_bias) {
    read_inputs.push_back(packed_bias);
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_linear_coopmat_shader,
      pick_linear_coopmat_global_wg_size,
      pick_linear_coopmat_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {read_inputs, vkapi::kRead}},
      // Shader params buffers
      {graph.sizes_ubo(input), graph.sizes_ubo(out)},
      // Push Constants
      {},
      // Specialization Constants
      {num_k_chunks, orig_N},
      // Resize Args
      {orig_N_ref, has_bias_ref},
      // Resizing Logic
      resize_linear_node));
}

// ── Matmul coopmat (with tile-sweep variants) ──

// Tile geometry per coopmat_mm.yaml matmul variant. wg_size =
// SG_GRID_X * SG_GRID_Y * SUBGROUP_SIZE and MUST equal the launched thread
// count, or the shader's grid-stride staging passes go out of bounds. Index 0
// is the baseline "matmul_coopmat" (no suffix); the rest are the sweep set.
struct MatmulCoopmatVariant {
  const char* suffix; // appended to "matmul_coopmat"
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t wg_size;
};
static constexpr MatmulCoopmatVariant kMatmulCoopmatVariants[] = {
    {"", 64, 64, 32, 256}, // baseline (2x2 grid x sg64)
    {"_t64x64x32", 64, 64, 32, 256},
    {"_t128x64x32", 128, 64, 32, 256},
    {"_t64x128x32", 64, 128, 32, 256},
    {"_t128x128x32", 128, 128, 32, 256},
    {"_t128x64x16", 128, 64, 16, 128}, // 2x2 grid x sg32
};
static constexpr int kNumMatmulCoopmatVariants =
    sizeof(kMatmulCoopmatVariants) / sizeof(kMatmulCoopmatVariants[0]);

// Map a tile_variant token ("" or e.g. "t128x64x32") to its variant index.
static int matmul_coopmat_variant_index(const std::string& tile_variant) {
  if (tile_variant.empty()) {
    return 0;
  }
  const std::string suffix = "_" + tile_variant;
  for (int i = 1; i < kNumMatmulCoopmatVariants; ++i) {
    if (suffix == kMatmulCoopmatVariants[i].suffix) {
      return i;
    }
  }
  VK_THROW("unknown matmul_coopmat tile_variant: ", tile_variant);
}

static vkapi::ShaderInfo pick_matmul_coopmat_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const int variant = graph->extract_scalar<int32_t>(resize_args.at(1));
  std::string kernel_name = "matmul_coopmat";
  kernel_name += kMatmulCoopmatVariants[variant].suffix;
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  return VK_KERNEL_FROM_STR(kernel_name);
}

static utils::uvec3 pick_matmul_coopmat_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  const ValueRef out = args.at(0).refs.at(0);
  const MatmulCoopmatVariant& v =
      kMatmulCoopmatVariants[graph->extract_scalar<int32_t>(resize_args.at(1))];
  const auto out_sizes = graph->sizes_of(out);
  uint32_t M = out_sizes.at(out_sizes.size() - 2);
  uint32_t N = out_sizes.at(out_sizes.size() - 1);
  uint32_t num_tiles_n = utils::div_up(N, v.n);
  uint32_t num_tiles_m = utils::div_up(M, v.m);
  // local_wg = {wg_size, 1, 1}; multiplying num_tiles_n by wg_size cancels the
  // framework's div_up so group_count.x == num_tiles_n (see linear node above).
  return {num_tiles_n * v.wg_size, num_tiles_m, 1};
}

static utils::uvec3 pick_matmul_coopmat_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)global_workgroup_size;
  (void)args;
  return {
      kMatmulCoopmatVariants[graph->extract_scalar<int32_t>(resize_args.at(1))]
          .wg_size,
      1,
      1};
}

void add_matmul_coopmat_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2,
    const ValueRef out,
    const std::string& tile_variant) {
  VK_CHECK_COND(graph.packed_dim_of(mat1) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(mat2) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(out) == WHCN::kWidthDim);
  VK_CHECK_COND(
      graph.storage_type_of(out) == utils::kBuffer,
      "matmul_coopmat requires buffer storage");

  const int variant = matmul_coopmat_variant_index(tile_variant);
  ValueRef has_bias_ref = graph.add_scalar(false);
  ValueRef variant_ref = graph.add_scalar<int64_t>(variant);

  // Same Xclipse spec-constant workarounds as the linear node above; the K-step
  // is the selected variant's WG_TILE_K. Restored 2026-07 alongside the linear
  // node's identical fix -- see the comment there for why.
  const int32_t tile_k =
      static_cast<int32_t>(kMatmulCoopmatVariants[variant].k);
  const int32_t K = graph.size_at<int32_t>(-1, mat1);
  VK_CHECK_COND(K % tile_k == 0);
  const int32_t num_k_chunks = K / tile_k;
  const int32_t out_N = graph.size_at<int32_t>(-1, out);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_matmul_coopmat_shader,
      pick_matmul_coopmat_global_wg_size,
      pick_matmul_coopmat_local_wg_size,
      // Inputs and Outputs — same binding order as matmul_vec
      {{out, vkapi::kWrite}, {{mat1, mat2}, vkapi::kRead}},
      // Shader params buffers — same UBOs as matmul_vec
      {graph.sizes_ubo(mat1), graph.sizes_ubo(mat2)},
      // Push Constants
      {},
      // Specialization Constants
      {num_k_chunks, out_N},
      // Resize Args (resize_args.at(1) = tile-variant index, read by pickers)
      {has_bias_ref, variant_ref},
      // Resizing Logic
      resize_matmul_tiled_node));
}

} // namespace vkcompute
