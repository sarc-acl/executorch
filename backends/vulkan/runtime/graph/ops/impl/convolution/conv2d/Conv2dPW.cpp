/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2d.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/Convolution.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <map>

namespace vkcompute {

//
// Shader dispatch utilities
//

namespace {

// Below this many wide tile invocations the GPU is underfilled, and Half
// convolutions are faster with the narrow tile. The threshold sits between the
// largest swept Half grid that was faster with the narrow tile on most GPUs
// (2560 wide tile invocations) and the smallest faster with the wide tile on
// every GPU (12288); grids in between were not swept.
constexpr uint64_t kMinWideTileInvocations = 4096;

// Row-aligned work groups on Mali: their thread count, and the pixel and
// channel tile counts of the convolutions that take them.
constexpr uint32_t kRowAlignedThreads = 896;
constexpr uint32_t kRowAlignedMinPixelTiles = 8192;
constexpr uint32_t kRowAlignedMinChannelTiles = 8;
constexpr uint32_t kRowAlignedMaxChannelTiles = 64;

// Pixel tile width of every variant; must match conv2d_pw_tiled.yaml.
constexpr uint32_t kTileW = 2;

// Work group size of conv2d_pw_coopmat_sk_merge.
constexpr uint32_t kMergeThreads = 64;

/*
 * The 2x2-pixel x 8-channel tile issues half as many input and weight loads per
 * FMA as the 2x1-pixel x 4-channel tile. Float uses it on every GPU except
 * Mali, where the wide Float tile spills registers in the Mali offline compiler
 * and was 2-31% slower than the narrow tile.
 */
bool use_wide_pw_tile(ComputeGraph* graph, const ValueRef out) {
  if (graph->dtype_of(out) != vkapi::kHalf) {
    return !graph->device_is_mali();
  }
  const uint64_t M = uint64_t(graph->size_at<uint32_t>(-1, out)) *
      graph->size_at<uint32_t>(-2, out);
  const uint64_t N4 = utils::div_up_4(graph->size_at<uint32_t>(-3, out));
  return utils::div_up(M, uint64_t(4)) * utils::div_up(N4, uint64_t(2)) >=
      kMinWideTileInvocations;
}

/*
 * Half wide tile convolutions took 0.829, 0.949 and 0.949 of the time on Adreno
 * 840, 750 and 740 (0.731 to 1.032 per shape) with the weights in a texture and
 * accumulators that start from the bias and sum each output texel's products
 * in a temporary, over 1x1 convolutions of 32 to 512 channels with 189x252 to
 * 1008x756 outputs. Mali-G715 and Xclipse 940 took 1.80 and 1.28 times as long
 * with them.
 */
bool use_folded_accumulation(ComputeGraph* graph, const ValueRef out) {
  return graph->dtype_of(out) == vkapi::kHalf && graph->device_is_adreno() &&
      use_wide_pw_tile(graph, out);
}

/*
 * On Mali-G715, 1x1 convolutions with at least as many 4-channel output slices
 * as input slices, 8, 16, 32 or 64 channel tiles and at least 8192 pixel tiles
 * took 0.771 to 1.008 of the time of the other picks with work groups of up to
 * 896 threads whose y spans the channel tiles and whose x divides the row of
 * pixel tiles. Other shapes took up to 1.11 of the time with such work groups
 * and keep the other picks. Adreno 740 to 840 took 1.03 to 1.05 of the time
 * with the same rule (up to 1.33 per shape), and Xclipse 940 up to 1.99 per
 * shape.
 */
bool use_row_aligned_lwg(
    ComputeGraph* graph,
    const ValueRef in,
    const ValueRef out,
    const GlobalWorkGrid& gwg) {
  const uint32_t n_tiles = gwg[1];
  return graph->device_is_mali() && gwg[0] >= kRowAlignedMinPixelTiles &&
      n_tiles >= kRowAlignedMinChannelTiles &&
      n_tiles <= kRowAlignedMaxChannelTiles &&
      (n_tiles & (n_tiles - 1u)) == 0u &&
      utils::div_up_4(graph->size_at<uint32_t>(-3, out)) >=
      utils::div_up_4(graph->size_at<uint32_t>(-3, in));
}

// The largest x up to max_x that divides row_tiles, or max_x if that is below
// half of it.
uint32_t pick_row_divisor(const uint32_t row_tiles, const uint32_t max_x) {
  uint32_t x = max_x;
  while (row_tiles % x != 0u) {
    --x;
  }
  return x * 2u >= max_x ? x : max_x;
}

bool is_wide_tile_shader(const vkapi::ShaderInfo& shader) {
  return shader.kernel_name.find("_4x8") != std::string::npos;
}

} // namespace

//
// Cooperative matrix variants
//

const std::vector<Conv2dPwCoopmatVariant>& conv2d_pw_coopmat_variants() {
  // {name, mma_m, mma_n, mma_k, tile_m, tile_n, tile_k, sg_grid_x,
  //  sg_grid_y, subgroup_size, fp32_accumulation, frag_layout[, options[,
  //  k_slices]]}; must match conv2d_pw_coopmat.yaml.
  using V = Conv2dPwCoopmatVariant;
  constexpr uint32_t kDiagOnce = V::kDiagOnce;
  constexpr uint32_t kAligned = V::kAligned;
  constexpr uint32_t kAlignedClampN = V::kAlignedClampN;
  constexpr uint32_t kThreadGrid = V::kThreadGrid;
  constexpr uint32_t kWideWeightLoad = V::kWideWeightLoad;
  constexpr uint32_t kImageInput = V::kImageInput;
  constexpr uint32_t kNoEpilogue = V::kNoEpilogue;
  constexpr uint32_t kColumnMajorB = V::kColumnMajorB;
  // clang-format off
  static const std::vector<Conv2dPwCoopmatVariant> variants = {
      // Intel Xe2
      {"conv2d_pw_coopmat_xe2_t16x32_f16_half", 8, 16, 16, 16, 32, 16, 1, 1, 16, false, true},
      {"conv2d_pw_coopmat_xe2_t16x32_f32_half", 8, 16, 16, 16, 32, 16, 1, 1, 16, true, true},
      {"conv2d_pw_coopmat_xe2_t16x64_f16_half", 8, 16, 16, 16, 64, 16, 1, 1, 16, false, true},
      {"conv2d_pw_coopmat_xe2_t16x64_f32_half", 8, 16, 16, 16, 64, 16, 1, 1, 16, true, true},
      {"conv2d_pw_coopmat_xe2_t32x32_f16_half", 8, 16, 16, 32, 32, 16, 1, 1, 16, false, true},
      {"conv2d_pw_coopmat_xe2_t32x32_f32_half", 8, 16, 16, 32, 32, 16, 1, 1, 16, true, true},
      {"conv2d_pw_coopmat_xe2_t32x64_f16_half", 8, 16, 16, 32, 64, 16, 2, 1, 16, false, true},
      {"conv2d_pw_coopmat_xe2_t32x64_f32_half", 8, 16, 16, 32, 64, 16, 2, 1, 16, true, true},
      {"conv2d_pw_coopmat_xe2_t64x32_f16_half", 8, 16, 16, 64, 32, 16, 1, 2, 16, false, true},
      {"conv2d_pw_coopmat_xe2_t64x32_f32_half", 8, 16, 16, 64, 32, 16, 1, 2, 16, true, true},
      {"conv2d_pw_coopmat_xe2_t64x64_f16_half", 8, 16, 16, 64, 64, 16, 2, 2, 16, false, true},
      {"conv2d_pw_coopmat_xe2_t64x64_f32_half", 8, 16, 16, 64, 64, 16, 2, 2, 16, true, true},
      {"conv2d_pw_coopmat_xe2_t128x128_f16_half", 8, 16, 16, 128, 128, 16, 4, 4, 16, false, true},
      {"conv2d_pw_coopmat_xe2_t128x128_f32_half", 8, 16, 16, 128, 128, 16, 4, 4, 16, true, true},
      // AMD
      {"conv2d_pw_coopmat_amd_t16x32_f16_half", 16, 16, 16, 16, 32, 32, 1, 1, 32, false, false},
      {"conv2d_pw_coopmat_amd_t16x32_f32_half", 16, 16, 16, 16, 32, 32, 1, 1, 32, true, false},
      {"conv2d_pw_coopmat_amd_t16x64_f16_half", 16, 16, 16, 16, 64, 32, 1, 1, 32, false, false},
      {"conv2d_pw_coopmat_amd_t16x64_f32_half", 16, 16, 16, 16, 64, 32, 1, 1, 32, true, false},
      {"conv2d_pw_coopmat_amd_t32x32_f16_half", 16, 16, 16, 32, 32, 32, 1, 1, 32, false, false},
      {"conv2d_pw_coopmat_amd_t32x32_f32_half", 16, 16, 16, 32, 32, 32, 1, 1, 32, true, false},
      {"conv2d_pw_coopmat_amd_t32x64_f16_half", 16, 16, 16, 32, 64, 32, 1, 1, 32, false, false},
      {"conv2d_pw_coopmat_amd_t32x64_f32_half", 16, 16, 16, 32, 64, 32, 1, 1, 32, true, false},
      {"conv2d_pw_coopmat_amd_t64x32_f16_half", 16, 16, 16, 64, 32, 32, 1, 1, 32, false, false},
      {"conv2d_pw_coopmat_amd_t64x32_f32_half", 16, 16, 16, 64, 32, 32, 1, 1, 32, true, false},
      {"conv2d_pw_coopmat_amd_t64x64_f16_half", 16, 16, 16, 64, 64, 32, 2, 1, 32, false, false},
      {"conv2d_pw_coopmat_amd_t64x64_f32_half", 16, 16, 16, 64, 64, 32, 2, 1, 32, true, false},
      {"conv2d_pw_coopmat_amd_t128x128_f16_half", 16, 16, 16, 128, 128, 32, 4, 2, 32, false, false},
      {"conv2d_pw_coopmat_amd_t128x128_f32_half", 16, 16, 16, 128, 128, 32, 4, 2, 32, true, false},
      // NVIDIA
      {"conv2d_pw_coopmat_nvidia_t16x32_f16_half", 16, 16, 16, 16, 32, 16, 1, 1, 32, false, false},
      {"conv2d_pw_coopmat_nvidia_t16x32_f32_half", 16, 16, 16, 16, 32, 16, 1, 1, 32, true, false},
      {"conv2d_pw_coopmat_nvidia_t16x64_f16_half", 16, 16, 16, 16, 64, 16, 1, 1, 32, false, false},
      {"conv2d_pw_coopmat_nvidia_t16x64_f32_half", 16, 16, 16, 16, 64, 16, 1, 1, 32, true, false},
      {"conv2d_pw_coopmat_nvidia_t32x32_f16_half", 16, 16, 16, 32, 32, 16, 1, 1, 32, false, false},
      {"conv2d_pw_coopmat_nvidia_t32x32_f32_half", 16, 16, 16, 32, 32, 16, 1, 1, 32, true, false},
      {"conv2d_pw_coopmat_nvidia_t32x64_f16_half", 16, 16, 16, 32, 64, 16, 1, 1, 32, false, false},
      {"conv2d_pw_coopmat_nvidia_t32x64_f32_half", 16, 16, 16, 32, 64, 16, 1, 1, 32, true, false},
      {"conv2d_pw_coopmat_nvidia_t64x32_f16_half", 16, 16, 16, 64, 32, 16, 1, 1, 32, false, false},
      {"conv2d_pw_coopmat_nvidia_t64x32_f32_half", 16, 16, 16, 64, 32, 16, 1, 1, 32, true, false},
      {"conv2d_pw_coopmat_nvidia_t64x64_f16_half", 16, 16, 16, 64, 64, 16, 2, 1, 32, false, false},
      {"conv2d_pw_coopmat_nvidia_t64x64_f32_half", 16, 16, 16, 64, 64, 16, 2, 1, 32, true, false},
      {"conv2d_pw_coopmat_nvidia_t128x128_f16_half", 16, 16, 16, 128, 128, 16, 4, 2, 32, false, false},
      {"conv2d_pw_coopmat_nvidia_t128x128_f32_half", 16, 16, 16, 128, 128, 16, 4, 2, 32, true, false},
      // Diagnostics of the device parents (staged-once roof)
      {"conv2d_pw_coopmat_xe2_t128x128_f16_diag_once_half", 8, 16, 16, 128, 128, 16, 4, 4, 16, false, true, kDiagOnce},
      {"conv2d_pw_coopmat_xe2_t128x128_f32_diag_once_half", 8, 16, 16, 128, 128, 16, 4, 4, 16, true, true, kDiagOnce},
      {"conv2d_pw_coopmat_amd_t16x32_f16_diag_once_half", 16, 16, 16, 16, 32, 32, 1, 1, 32, false, false, kDiagOnce},
      {"conv2d_pw_coopmat_amd_t16x32_f32_diag_once_half", 16, 16, 16, 16, 32, 32, 1, 1, 32, true, false, kDiagOnce},
      // Aligned-only (_al)
      {"conv2d_pw_coopmat_xe2_t16x32_f16_al_half", 8, 16, 16, 16, 32, 16, 1, 1, 16, false, true, kAligned},
      {"conv2d_pw_coopmat_xe2_t16x32_f32_al_half", 8, 16, 16, 16, 32, 16, 1, 1, 16, true, true, kAligned},
      {"conv2d_pw_coopmat_xe2_t64x64_f16_al_half", 8, 16, 16, 64, 64, 16, 2, 2, 16, false, true, kAligned},
      {"conv2d_pw_coopmat_xe2_t64x64_f32_al_half", 8, 16, 16, 64, 64, 16, 2, 2, 16, true, true, kAligned},
      {"conv2d_pw_coopmat_amd_t16x32_f16_al_half", 16, 16, 16, 16, 32, 32, 1, 1, 32, false, false, kAligned},
      {"conv2d_pw_coopmat_amd_t16x32_f32_al_half", 16, 16, 16, 16, 32, 32, 1, 1, 32, true, false, kAligned},
      {"conv2d_pw_coopmat_amd_t16x64_f16_al_half", 16, 16, 16, 16, 64, 32, 1, 1, 32, false, false, kAligned},
      {"conv2d_pw_coopmat_amd_t16x64_f32_al_half", 16, 16, 16, 16, 64, 32, 1, 1, 32, true, false, kAligned},
      {"conv2d_pw_coopmat_nvidia_t32x32_f16_al_half", 16, 16, 16, 32, 32, 16, 1, 1, 32, false, false, kAligned},
      {"conv2d_pw_coopmat_nvidia_t32x32_f32_al_half", 16, 16, 16, 32, 32, 16, 1, 1, 32, true, false, kAligned},
      {"conv2d_pw_coopmat_nvidia_t16x64_f16_al_half", 16, 16, 16, 16, 64, 16, 1, 1, 32, false, false, kAligned},
      {"conv2d_pw_coopmat_nvidia_t16x64_f32_al_half", 16, 16, 16, 16, 64, 16, 1, 1, 32, true, false, kAligned},
      // Thread grid (_g4x2) on Xe2 and the NVIDIA 16x16 tile
      {"conv2d_pw_coopmat_xe2_t64x64_f16_g4x2_half", 8, 16, 16, 64, 64, 16, 4, 2, 16, false, true, kThreadGrid},
      {"conv2d_pw_coopmat_xe2_t64x64_f32_g4x2_half", 8, 16, 16, 64, 64, 16, 4, 2, 16, true, true, kThreadGrid},
      {"conv2d_pw_coopmat_xe2_t32x64_f16_g4x2_half", 8, 16, 16, 32, 64, 16, 4, 2, 16, false, true, kThreadGrid},
      {"conv2d_pw_coopmat_xe2_t32x64_f32_g4x2_half", 8, 16, 16, 32, 64, 16, 4, 2, 16, true, true, kThreadGrid},
      {"conv2d_pw_coopmat_nvidia_t16x16_f16_half", 16, 16, 16, 16, 16, 16, 1, 1, 32, false, false},
      {"conv2d_pw_coopmat_nvidia_t16x16_f32_half", 16, 16, 16, 16, 16, 16, 1, 1, 32, true, false},
      // Wide weight load (_w16) on the N-tile-32 aligned-only tiles
      {"conv2d_pw_coopmat_xe2_t16x32_f16_al_w16_half", 8, 16, 16, 16, 32, 16, 1, 1, 16, false, true, kAligned | kWideWeightLoad},
      {"conv2d_pw_coopmat_xe2_t16x32_f32_al_w16_half", 8, 16, 16, 16, 32, 16, 1, 1, 16, true, true, kAligned | kWideWeightLoad},
      {"conv2d_pw_coopmat_amd_t16x32_f16_al_w16_half", 16, 16, 16, 16, 32, 32, 1, 1, 32, false, false, kAligned | kWideWeightLoad},
      {"conv2d_pw_coopmat_amd_t16x32_f32_al_w16_half", 16, 16, 16, 16, 32, 32, 1, 1, 32, true, false, kAligned | kWideWeightLoad},
      {"conv2d_pw_coopmat_nvidia_t32x32_f16_al_w16_half", 16, 16, 16, 32, 32, 16, 1, 1, 32, false, false, kAligned | kWideWeightLoad},
      {"conv2d_pw_coopmat_nvidia_t32x32_f32_al_w16_half", 16, 16, 16, 32, 32, 16, 1, 1, 32, true, false, kAligned | kWideWeightLoad},
      // Storage-image input (_img) on the Xe2 N-tile-32 aligned-only tile
      {"conv2d_pw_coopmat_xe2_t16x32_f16_al_img_half", 8, 16, 16, 16, 32, 16, 1, 1, 16, false, true, kAligned | kImageInput},
      {"conv2d_pw_coopmat_xe2_t16x32_f32_al_img_half", 8, 16, 16, 16, 32, 16, 1, 1, 16, true, true, kAligned | kImageInput},
      // Epilogue skip (_epi0) on the N-tile-32 aligned-only tiles
      {"conv2d_pw_coopmat_xe2_t16x32_f16_al_epi0_half", 8, 16, 16, 16, 32, 16, 1, 1, 16, false, true, kAligned | kNoEpilogue},
      {"conv2d_pw_coopmat_xe2_t16x32_f32_al_epi0_half", 8, 16, 16, 16, 32, 16, 1, 1, 16, true, true, kAligned | kNoEpilogue},
      {"conv2d_pw_coopmat_amd_t16x32_f16_al_epi0_half", 16, 16, 16, 16, 32, 32, 1, 1, 32, false, false, kAligned | kNoEpilogue},
      {"conv2d_pw_coopmat_amd_t16x32_f32_al_epi0_half", 16, 16, 16, 16, 32, 32, 1, 1, 32, true, false, kAligned | kNoEpilogue},
      {"conv2d_pw_coopmat_nvidia_t32x32_f16_al_epi0_half", 16, 16, 16, 32, 32, 16, 1, 1, 32, false, false, kAligned | kNoEpilogue},
      {"conv2d_pw_coopmat_nvidia_t32x32_f32_al_epi0_half", 16, 16, 16, 32, 32, 16, 1, 1, 32, true, false, kAligned | kNoEpilogue},
      // Column-major B staging (_bcm) on the AMD 16x32 tile
      {"conv2d_pw_coopmat_amd_t16x32_f16_bcm_half", 16, 16, 16, 16, 32, 32, 1, 1, 32, false, false, kColumnMajorB},
      {"conv2d_pw_coopmat_amd_t16x32_f32_bcm_half", 16, 16, 16, 16, 32, 32, 1, 1, 32, true, false, kColumnMajorB},
      // Split-K across work groups (_sk2, _sk4)
      {"conv2d_pw_coopmat_xe2_t128x128_f16_sk2_half", 8, 16, 16, 128, 128, 16, 4, 4, 16, false, true, 0u, 2u},
      {"conv2d_pw_coopmat_xe2_t128x128_f16_sk4_half", 8, 16, 16, 128, 128, 16, 4, 4, 16, false, true, 0u, 4u},
      {"conv2d_pw_coopmat_xe2_t128x128_f32_sk2_half", 8, 16, 16, 128, 128, 16, 4, 4, 16, true, true, 0u, 2u},
      {"conv2d_pw_coopmat_xe2_t128x128_f32_sk4_half", 8, 16, 16, 128, 128, 16, 4, 4, 16, true, true, 0u, 4u},
      {"conv2d_pw_coopmat_nvidia_t128x128_f16_sk2_half", 16, 16, 16, 128, 128, 16, 4, 2, 32, false, false, 0u, 2u},
      {"conv2d_pw_coopmat_nvidia_t128x128_f16_sk4_half", 16, 16, 16, 128, 128, 16, 4, 2, 32, false, false, 0u, 4u},
      {"conv2d_pw_coopmat_nvidia_t128x128_f32_sk2_half", 16, 16, 16, 128, 128, 16, 4, 2, 32, true, false, 0u, 2u},
      {"conv2d_pw_coopmat_nvidia_t128x128_f32_sk4_half", 16, 16, 16, 128, 128, 16, 4, 2, 32, true, false, 0u, 4u},
      // Split-K for the AMD family (four slices)
      {"conv2d_pw_coopmat_amd_t16x32_f16_sk4_half", 16, 16, 16, 16, 32, 32, 1, 1, 32, false, false, 0u, 4u},
      {"conv2d_pw_coopmat_amd_t16x32_f32_sk4_half", 16, 16, 16, 16, 32, 32, 1, 1, 32, true, false, 0u, 4u},
      // Clamped-read aligned-only (_alc) on the Xe2 128x128 tile, with split-K,
      // and on the thread-rich tiles
      {"conv2d_pw_coopmat_xe2_t128x128_f16_alc_half", 8, 16, 16, 128, 128, 16, 4, 4, 16, false, true, kAlignedClampN},
      {"conv2d_pw_coopmat_xe2_t128x128_f32_alc_half", 8, 16, 16, 128, 128, 16, 4, 4, 16, true, true, kAlignedClampN},
      {"conv2d_pw_coopmat_xe2_t128x128_f16_alc_sk2_half", 8, 16, 16, 128, 128, 16, 4, 4, 16, false, true, kAlignedClampN, 2u},
      {"conv2d_pw_coopmat_xe2_t128x128_f16_alc_sk4_half", 8, 16, 16, 128, 128, 16, 4, 4, 16, false, true, kAlignedClampN, 4u},
      {"conv2d_pw_coopmat_xe2_t64x64_f16_g4x2_alc_half", 8, 16, 16, 64, 64, 16, 4, 2, 16, false, true, kThreadGrid | kAlignedClampN},
      {"conv2d_pw_coopmat_xe2_t32x64_f16_g4x2_alc_half", 8, 16, 16, 32, 64, 16, 4, 2, 16, false, true, kThreadGrid | kAlignedClampN},
      {"conv2d_pw_coopmat_xe2_t32x64_f16_g4x2_alc_sk2_half", 8, 16, 16, 32, 64, 16, 4, 2, 16, false, true, kThreadGrid | kAlignedClampN, 2u},
      {"conv2d_pw_coopmat_xe2_t32x64_f16_g4x2_alc_sk4_half", 8, 16, 16, 32, 64, 16, 4, 2, 16, false, true, kThreadGrid | kAlignedClampN, 4u},
  };
  // clang-format on
  return variants;
}

const Conv2dPwCoopmatVariant* find_conv2d_pw_coopmat_variant(
    const std::string& name) {
  for (const Conv2dPwCoopmatVariant& variant : conv2d_pw_coopmat_variants()) {
    if (name == variant.name) {
      return &variant;
    }
  }
  return nullptr;
}

namespace {

// Option suffixes in name order; kSlices stands for "_sk<n>".
constexpr uint32_t kSlices = 0u;
constexpr uint32_t kSuffixOrder[] = {
    Conv2dPwCoopmatVariant::kThreadGrid,
    Conv2dPwCoopmatVariant::kAligned,
    Conv2dPwCoopmatVariant::kAlignedClampN,
    Conv2dPwCoopmatVariant::kWideWeightLoad,
    Conv2dPwCoopmatVariant::kImageInput,
    Conv2dPwCoopmatVariant::kNoEpilogue,
    Conv2dPwCoopmatVariant::kColumnMajorB,
    kSlices,
    Conv2dPwCoopmatVariant::kDiagOnce,
};

bool has_suffix_item(const Conv2dPwCoopmatVariant& v, const uint32_t item) {
  return item == kSlices ? v.k_slices > 1u : (v.options & item) != 0u;
}

std::string suffix_item(const Conv2dPwCoopmatVariant& v, const uint32_t item) {
  switch (item) {
    case kSlices:
      return "_sk" + std::to_string(v.k_slices);
    case Conv2dPwCoopmatVariant::kThreadGrid:
      return "_g" + std::to_string(v.sg_grid_x) + "x" +
          std::to_string(v.sg_grid_y);
    case Conv2dPwCoopmatVariant::kAligned:
      return "_al";
    case Conv2dPwCoopmatVariant::kAlignedClampN:
      return "_alc";
    case Conv2dPwCoopmatVariant::kWideWeightLoad:
      return "_w16";
    case Conv2dPwCoopmatVariant::kImageInput:
      return "_img";
    case Conv2dPwCoopmatVariant::kNoEpilogue:
      return "_epi0";
    case Conv2dPwCoopmatVariant::kColumnMajorB:
      return "_bcm";
    case Conv2dPwCoopmatVariant::kDiagOnce:
      return "_diag_once";
    default:
      return "_unknown" + std::to_string(item);
  }
}

} // namespace

std::string conv2d_pw_coopmat_option_suffix(
    const Conv2dPwCoopmatVariant& variant) {
  std::string suffix;
  for (const uint32_t item : kSuffixOrder) {
    if (has_suffix_item(variant, item)) {
      suffix += suffix_item(variant, item);
    }
  }
  return suffix;
}

std::vector<std::string> conv2d_pw_coopmat_registry_problems(
    const std::vector<Conv2dPwCoopmatVariant>& variants) {
  std::vector<std::string> problems;
  std::map<std::string, const Conv2dPwCoopmatVariant*> by_name;
  for (const Conv2dPwCoopmatVariant& v : variants) {
    if (!by_name.emplace(v.name, &v).second) {
      problems.push_back(std::string("duplicate variant name ") + v.name);
    }
  }
  uint32_t known = 0u;
  for (const uint32_t item : kSuffixOrder) {
    known |= item;
  }
  for (const Conv2dPwCoopmatVariant& v : variants) {
    const std::string name = v.name;
    const std::string suffix = conv2d_pw_coopmat_option_suffix(v);
    const std::string base = std::string("_t") + std::to_string(v.tile_m) +
        "x" + std::to_string(v.tile_n) +
        (v.fp32_accumulation ? "_f32" : "_f16");
    bool named = false;
    for (const char* family : {"xe2", "amd", "nvidia"}) {
      named = named ||
          name ==
              std::string("conv2d_pw_coopmat_") + family + base + suffix +
                  "_half";
    }
    if (!named) {
      problems.push_back(
          name + ": not conv2d_pw_coopmat_<family>" + base + suffix + "_half");
    }
    if ((v.options & ~known) != 0u || v.k_slices == 0u) {
      problems.push_back(name + ": unknown option or zero K slices");
    }
    if (v.has(Conv2dPwCoopmatVariant::kWideWeightLoad) &&
        !v.has(Conv2dPwCoopmatVariant::kAligned)) {
      problems.push_back(name + ": _w16 needs _al");
    }
    if (v.has(Conv2dPwCoopmatVariant::kAlignedClampN) &&
        (v.has(Conv2dPwCoopmatVariant::kAligned) ||
         v.has(Conv2dPwCoopmatVariant::kWideWeightLoad))) {
      problems.push_back(name + ": _alc excludes _al and _w16");
    }
    if (v.has(Conv2dPwCoopmatVariant::kColumnMajorB) && v.frag_layout) {
      problems.push_back(name + ": _bcm needs the padded row-major layout");
    }
    if (suffix.empty()) {
      continue;
    }
    // The parent is the variant without the last option.
    Conv2dPwCoopmatVariant parent = v;
    uint32_t last = kSlices;
    for (const uint32_t item : kSuffixOrder) {
      if (has_suffix_item(v, item)) {
        last = item;
      }
    }
    if (last == kSlices) {
      parent.k_slices = 1u;
    } else {
      parent.options &= ~last;
    }
    const std::string parent_name =
        name.substr(
            0, name.size() - suffix.size() - std::string("_half").size()) +
        conv2d_pw_coopmat_option_suffix(parent) + "_half";
    const auto found = by_name.find(parent_name);
    if (found == by_name.end()) {
      problems.push_back(name + ": parent " + parent_name + " is missing");
      continue;
    }
    const Conv2dPwCoopmatVariant& p = *found->second;
    const bool grid_free = last == Conv2dPwCoopmatVariant::kThreadGrid;
    if (p.mma_m != v.mma_m || p.mma_n != v.mma_n || p.mma_k != v.mma_k ||
        p.tile_m != v.tile_m || p.tile_n != v.tile_n || p.tile_k != v.tile_k ||
        (!grid_free &&
         (p.sg_grid_x != v.sg_grid_x || p.sg_grid_y != v.sg_grid_y)) ||
        p.subgroup_size != v.subgroup_size ||
        p.fp32_accumulation != v.fp32_accumulation ||
        p.frag_layout != v.frag_layout || p.options != parent.options ||
        p.k_slices != parent.k_slices) {
      problems.push_back(
          name + ": differs from its parent " + parent_name + " in more than " +
          suffix_item(v, last));
    }
  }
  return problems;
}

std::optional<std::string> conv2d_pw_coopmat_limits_unsupported(
    const Conv2dPwCoopmatVariant& variant,
    const Conv2dPwCoopmatLimits& limits) {
  const auto n = [](const uint64_t v) { return std::to_string(v); };
  // Every local invocation is on x (the local work group is {wg_size, 1, 1}).
  if (variant.wg_size() > limits.max_invocations) {
    return "work group size " + n(variant.wg_size()) +
        " exceeds maxComputeWorkGroupInvocations " + n(limits.max_invocations);
  }
  if (variant.wg_size() > limits.max_size_x) {
    return "work group size " + n(variant.wg_size()) +
        " exceeds maxComputeWorkGroupSize.x " + n(limits.max_size_x);
  }
  // VUID-VkPipelineShaderStageCreateInfo-pNext-02756.
  if (variant.wg_size() >
      uint64_t(variant.subgroup_size) * limits.max_subgroups) {
    return "work group size " + n(variant.wg_size()) +
        " exceeds subgroup size " + n(variant.subgroup_size) +
        " x maxComputeWorkgroupSubgroups " + n(limits.max_subgroups);
  }
  if (variant.shared_bytes() > limits.max_shared_bytes) {
    return "shared memory " + n(variant.shared_bytes()) +
        " bytes exceeds maxComputeSharedMemorySize " +
        n(limits.max_shared_bytes);
  }
  return std::nullopt;
}

uint32_t Conv2dPwCoopmatVariant::shared_bytes() const {
  // Mirrors the shared pool of conv2d_pw_coopmat_body.glslh and the drain
  // pass split of conv2d_pw_coopmat.glsl (32 KiB budget for the staged rows).
  constexpr uint32_t kVec4Bytes = 16u;
  constexpr uint32_t kDrainBudgetBytes = 32768u;
  const uint32_t a_slice =
      frag_layout ? tile_m * (tile_k / 8u) : tile_m * ((tile_k + 8u) / 8u);
  const uint32_t b_slice = frag_layout ? tile_k * (tile_n / 8u)
      : has(kColumnMajorB)             ? tile_n * ((tile_k + 8u) / 8u)
                                       : tile_k * ((tile_n + 8u) / 8u);
  const uint32_t pool_ab = 2u * a_slice + 2u * b_slice;
  const uint32_t acc_bytes = fp32_accumulation ? 4u : 2u;
  const uint32_t mmas_m = tile_m / sg_grid_y / mma_m;
  uint32_t drain_g = 1u;
  for (uint32_t g = 2u; g <= mmas_m; g *= 2u) {
    if (mmas_m % g == 0u &&
        sg_grid_y * mma_m * g * tile_n * acc_bytes <= kDrainBudgetBytes) {
      drain_g = g;
    }
  }
  const uint32_t pool_c =
      sg_grid_y * mma_m * drain_g * tile_n * acc_bytes / kVec4Bytes;
  return (pool_ab > pool_c ? pool_ab : pool_c) * kVec4Bytes;
}

utils::uvec3 conv2d_pw_coopmat_workgroups(
    ComputeGraph& graph,
    const Conv2dPwCoopmatVariant& variant,
    const uint32_t N,
    const uint32_t M) {
  // Pixel tiles fill y up to the device limit and continue in z; the shader
  // reads the tile as gl_WorkGroupID.y + gl_WorkGroupID.z * gl_NumWorkGroups.y.
  const uint32_t m_tiles = utils::div_up(M, variant.tile_m);
  const uint32_t max_y =
      graph.context()->adapter_ptr()->max_compute_workgroup_count()[1];
  const uint32_t y = m_tiles < max_y ? m_tiles : max_y;
  // Split-K: x is channel tile + K slice x channel tiles.
  return {
      utils::div_up(N, variant.tile_n) * variant.k_slices,
      y,
      utils::div_up(m_tiles, y)};
}

std::vector<std::pair<uint32_t, uint32_t>> conv2d_pw_coopmat_k_slices(
    const Conv2dPwCoopmatVariant& variant,
    const uint32_t K) {
  const uint32_t steps = utils::div_up(K, variant.tile_k);
  const uint32_t q = utils::div_up(steps, variant.k_slices);
  std::vector<std::pair<uint32_t, uint32_t>> slices;
  for (uint32_t j = 0; j < variant.k_slices; ++j) {
    const uint32_t begin = std::min(j * q, steps);
    slices.emplace_back(begin, std::min(begin + q, steps));
  }
  return slices;
}

std::string conv2d_pw_coopmat_merge_shader(
    const Conv2dPwCoopmatVariant& variant) {
  return "conv2d_pw_coopmat_sk" + std::to_string(variant.k_slices) + "_merge_" +
      (variant.fp32_accumulation ? "f32" : "f16") + "_half";
}

uint32_t conv2d_pw_coopmat_scratch_capacity(
    ComputeGraph& graph,
    const ValueRef out) {
  // Width x height of the output texture: the largest plane a resize can give
  // the output (virtual resizes stay within the allocated extents).
  const VkExtent3D extents = graph.image_of(out).extents();
  return extents.width * extents.height;
}

uint32_t conv2d_pw_coopmat_scratch_rows(
    const Conv2dPwCoopmatVariant& variant,
    const uint32_t M) {
  // Texels of all slices, one f16vec4 entry (fp16) or two (fp32) each, in
  // rows of N4 * 4 entries of the packed weight tensor.
  if (variant.k_slices <= 1u) {
    return 0u;
  }
  const uint32_t entries_per_pixel =
      variant.k_slices * (variant.fp32_accumulation ? 2u : 1u);
  return utils::div_up(M * entries_per_pixel, 4u);
}

std::optional<std::string> conv2d_pw_coopmat_device_unsupported(
    ComputeGraph& graph,
    const Conv2dPwCoopmatVariant& variant) {
  const auto n = [](const uint64_t v) { return std::to_string(v); };
#ifdef VK_KHR_cooperative_matrix
  vkapi::Adapter* const adapter = graph.context()->adapter_ptr();
  if (!adapter->supports_cooperative_matrix()) {
    return "the device does not support VK_KHR_cooperative_matrix";
  }
  const VkComponentTypeKHR acc = variant.fp32_accumulation
      ? VK_COMPONENT_TYPE_FLOAT32_KHR
      : VK_COMPONENT_TYPE_FLOAT16_KHR;
  if (!adapter->supports_cooperative_matrix_shape(
          variant.mma_m,
          variant.mma_n,
          variant.mma_k,
          VK_COMPONENT_TYPE_FLOAT16_KHR,
          VK_COMPONENT_TYPE_FLOAT16_KHR,
          acc,
          acc)) {
    return "the device does not enumerate the cooperative matrix " +
        n(variant.mma_m) + "x" + n(variant.mma_n) + "x" + n(variant.mma_k) +
        " fp16 -> " + (variant.fp32_accumulation ? "fp32" : "fp16") +
        " at subgroup scope";
  }
  if (!adapter->supports_required_subgroup_size_for_compute() ||
      variant.subgroup_size < adapter->min_subgroup_size() ||
      variant.subgroup_size > adapter->max_subgroup_size()) {
    return "the device cannot require subgroup size " +
        n(variant.subgroup_size) + " for compute (range " +
        n(adapter->min_subgroup_size()) + "-" +
        n(adapter->max_subgroup_size()) + ")";
  }
  if (const auto reason = conv2d_pw_coopmat_limits_unsupported(
          variant,
          {adapter->max_compute_workgroup_invocations(),
           adapter->max_compute_workgroup_size()[0],
           adapter->max_compute_workgroup_subgroups(),
           adapter->max_compute_shared_memory_size()})) {
    return reason;
  }
  for (const uint32_t capability :
       VK_KERNEL_FROM_STR(variant.name).required_capabilities) {
    if (!adapter->supports_spirv_capability(capability)) {
      return "the device lacks SPIR-V capability " + n(capability);
    }
  }
  return std::nullopt;
#else
  (void)graph;
  (void)variant;
  (void)n;
  return "built without VK_KHR_cooperative_matrix";
#endif /* VK_KHR_cooperative_matrix */
}

bool conv2d_pw_coopmat_device_supports(
    ComputeGraph& graph,
    const Conv2dPwCoopmatVariant& variant) {
  return !conv2d_pw_coopmat_device_unsupported(graph, variant).has_value();
}

std::optional<std::string> conv2d_pw_coopmat_grid_unsupported(
    const utils::uvec3& grid,
    const utils::uvec3& max_count) {
  static const char* const kAxis[] = {"x", "y", "z"};
  for (uint32_t i = 0; i < 3u; ++i) {
    if (grid[i] > max_count[i]) {
      return std::string("dispatch needs ") + std::to_string(grid[i]) +
          " work groups in " + kAxis[i] +
          ", more than maxComputeWorkGroupCount." + kAxis[i] + " " +
          std::to_string(max_count[i]);
    }
  }
  return std::nullopt;
}

std::optional<std::string> conv2d_pw_coopmat_ineligible(
    ComputeGraph& graph,
    const Conv2dPwCoopmatVariant& variant,
    const ValueRef in,
    const ValueRef weight_data,
    const utils::StorageType weight_storage,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef out,
    const bool transposed,
    const bool has_bias_or_epilogue) {
  if (transposed) {
    return "the convolution is transposed";
  }
  if (graph.dtype_of(in) != vkapi::kHalf ||
      graph.dtype_of(out) != vkapi::kHalf) {
    return "input and output are not FP16";
  }
  if (!graph.float16_buffers_enabled()) {
    return "16-bit storage buffers are not enabled";
  }
  if (weight_storage != utils::kBuffer) {
    return "the packed weights are not stored in a buffer";
  }
  if (!graph.is_texture_storage(in) || !graph.is_texture_storage(out) ||
      graph.packed_dim_of(in) != WHCN::kChannelsDim ||
      graph.packed_dim_of(out) != WHCN::kChannelsDim) {
    return "input and output are not channels-packed textures";
  }
  const std::vector<int64_t> in_sizes = graph.sizes_of(in);
  const std::vector<int64_t> weight_sizes = graph.sizes_of(weight_data);
  if (in_sizes.size() == 4 && in_sizes.at(0) != 1) {
    return "the batch is not 1";
  }
  if (weight_sizes.size() != 4 ||
      weight_sizes.at(1) != graph.size_at<int64_t>(-3, in) ||
      weight_sizes.at(2) != 1 || weight_sizes.at(3) != 1) {
    return "the convolution is not an ungrouped 1x1 convolution";
  }
  const auto stride_list = graph.get_int_list(stride);
  const auto padding_list = graph.get_int_list(padding);
  if (stride_list->at(0) != 1 || stride_list->at(1) != 1 ||
      padding_list->at(0) != 0 || padding_list->at(1) != 0) {
    return "stride is not 1 or padding is not 0";
  }
  const uint32_t M =
      graph.size_at<uint32_t>(-1, out) * graph.size_at<uint32_t>(-2, out);
  if (M <= 1u) {
    return "the output has a single pixel";
  }
  if (const auto reason =
          conv2d_pw_coopmat_device_unsupported(graph, variant)) {
    return reason;
  }
  if (variant.has(Conv2dPwCoopmatVariant::kImageInput)) {
    // The shader reads the input with imageLoad: the image needs storage
    // usage and a format the device can read as a storage image; the node
    // binds it in VK_IMAGE_LAYOUT_GENERAL (add_conv2d_pw_tiled_node).
    const vkapi::VulkanImage& image = graph.image_of(in);
    if ((image.usage() & VK_IMAGE_USAGE_STORAGE_BIT) == 0u) {
      return "the input image has no storage usage";
    }
    if (!graph.context()->adapter_ptr()->supports_storage_image_format(
            image.format(), image.tiling())) {
      return "the device cannot use the input format " +
          std::to_string(image.format()) + " as a storage image";
    }
  }
  if (variant.has(Conv2dPwCoopmatVariant::kAligned)) {
    const uint32_t K = graph.size_at<uint32_t>(-3, in);
    const uint32_t N = graph.size_at<uint32_t>(-3, out);
    if (K % variant.tile_k != 0u || N % variant.tile_n != 0u) {
      return "input channels " + std::to_string(K) + " or output channels " +
          std::to_string(N) + " not a multiple of the variant's K step " +
          std::to_string(variant.tile_k) + " and N tile " +
          std::to_string(variant.tile_n);
    }
  }
  if (variant.has(Conv2dPwCoopmatVariant::kAlignedClampN)) {
    const uint32_t K = graph.size_at<uint32_t>(-3, in);
    if (K % variant.tile_k != 0u) {
      return "input channels " + std::to_string(K) +
          " not a multiple of the variant's K step " +
          std::to_string(variant.tile_k);
    }
  }
  if (variant.has(Conv2dPwCoopmatVariant::kNoEpilogue) &&
      has_bias_or_epilogue) {
    return "the convolution has a bias, activation, residual or scale, which "
           "the variant does not apply";
  }
  const utils::uvec3 max_count =
      graph.context()->adapter_ptr()->max_compute_workgroup_count();
  if (variant.k_slices > 1u) {
    // Design D6: slice j covers K steps [j q, (j + 1) q) with
    // q = ceil(steps / slices); every slice needs at least two.
    const uint32_t K = graph.size_at<uint32_t>(-3, in);
    for (const auto& [begin, end] : conv2d_pw_coopmat_k_slices(variant, K)) {
      if (end < begin + 2u) {
        return "input channels " + std::to_string(K) + " give " +
            std::to_string(utils::div_up(K, variant.tile_k)) +
            " K steps: fewer than two K steps in a slice of " +
            std::to_string(variant.k_slices);
      }
    }
    // The scratch is sized for every plane the output's storage can hold
    // (conv2d_pw_coopmat_scratch_capacity), and shares the packed weight
    // buffer, whose whole range is bound.
    const uint64_t N4 = utils::div_up_4(graph.size_at<uint32_t>(-3, out));
    const uint64_t rows =
        utils::div_up_4(uint64_t(K)) +
        conv2d_pw_coopmat_scratch_rows(
            variant, conv2d_pw_coopmat_scratch_capacity(graph, out));
    const uint64_t bytes = rows * N4 * 16u * 2u;
    const uint64_t max_range =
        // maxStorageBufferRange (bytes).
        graph.context()->adapter_ptr()->max_buffer_numel();
    if (bytes > max_range) {
      return "the packed weights and split-K scratch take " +
          std::to_string(bytes) + " bytes, more than maxStorageBufferRange " +
          std::to_string(max_range);
    }
    const uint64_t merge_groups = utils::div_up(
        uint64_t(M) * utils::div_up_4(graph.size_at<uint32_t>(-3, out)),
        uint64_t(kMergeThreads));
    if (merge_groups > max_count[0]) {
      return "the merge needs " + std::to_string(merge_groups) +
          " work groups in x, more than maxComputeWorkGroupCount.x " +
          std::to_string(max_count[0]);
    }
  }
  return conv2d_pw_coopmat_grid_unsupported(
      conv2d_pw_coopmat_workgroups(
          graph, variant, graph.size_at<uint32_t>(-3, out), M),
      max_count);
}

bool conv2d_pw_coopmat_eligible(
    ComputeGraph& graph,
    const Conv2dPwCoopmatVariant& variant,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef packed_weight,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef out,
    const bool transposed,
    const bool has_bias_or_epilogue) {
  if (variant.k_slices > 1u) {
    // The scratch rows the packed weight tensor carries must hold the current
    // plane.
    const uint32_t M =
        graph.size_at<uint32_t>(-1, out) * graph.size_at<uint32_t>(-2, out);
    const uint32_t K4 = utils::div_up_4(graph.size_at<uint32_t>(-3, in));
    const uint32_t rows = graph.size_at<uint32_t>(-2, packed_weight);
    if (rows < K4 || rows - K4 < conv2d_pw_coopmat_scratch_rows(variant, M)) {
      return false;
    }
  }
  return !conv2d_pw_coopmat_ineligible(
              graph,
              variant,
              in,
              weight_data,
              graph.storage_type_of(packed_weight),
              stride,
              padding,
              out,
              transposed,
              has_bias_or_epilogue)
              .has_value();
}

void resize_conv2d_pw_tiled_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef self = args.at(1).refs.at(0);

  std::vector<int64_t> self_sizes = graph->sizes_of(self);
  TensorRefPtr weight_ref = graph->get_tref(extra_args.at(0));
  const auto& weight_sizes = weight_ref->sizes;

  const auto stride_list = graph->get_int_list(extra_args.at(1));
  const auto padding_list = graph->get_int_list(extra_args.at(2));

  const int64_t stride_h = stride_list->at(0);
  const int64_t stride_w = stride_list->at(1);
  const int64_t padding_h = padding_list->at(0);
  const int64_t padding_w = padding_list->at(1);

  const int64_t in_h = self_sizes.at(self_sizes.size() - 2);
  const int64_t in_w = self_sizes.at(self_sizes.size() - 1);

  // For 1x1 kernel with dilation=1: out = (in + 2*padding - 1) / stride + 1
  const int64_t out_h = (in_h + 2 * padding_h - 1) / stride_h + 1;
  const int64_t out_w = (in_w + 2 * padding_w - 1) / stride_w + 1;

  std::vector<int64_t> new_out_sizes = self_sizes;
  new_out_sizes.at(self_sizes.size() - 3) = weight_sizes.at(0);
  new_out_sizes.at(self_sizes.size() - 2) = out_h;
  new_out_sizes.at(self_sizes.size() - 1) = out_w;

  graph->virtual_resize(out, new_out_sizes);
  check_conv2d_residual_on_resize(graph, out, extra_args);
}

vkapi::ShaderInfo pick_conv2d_pw_tiled_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args,
    const Conv2dPwRequest& request,
    const bool transposed,
    const bool has_bias_or_epilogue) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  const ValueRef packed_weight = args.at(2).refs.at(0);

  if (request.kind == Conv2dPwRequest::Kind::kCoopmat &&
      conv2d_pw_coopmat_eligible(
          *graph,
          *request.coopmat_variant,
          in,
          resize_args.at(0),
          packed_weight,
          resize_args.at(1),
          resize_args.at(2),
          out,
          transposed,
          has_bias_or_epilogue)) {
    return VK_KERNEL_FROM_STR(request.coopmat_variant->name);
  }

  std::string kernel_name = "conv2d_pw_tiled";
  kernel_name.reserve(kShaderNameReserve);
  if (request.kind == Conv2dPwRequest::Kind::kTiledWide ||
      use_wide_pw_tile(graph, out)) {
    kernel_name += "_4x8";
  }
  if (graph->is_buffer_storage(packed_weight)) {
    kernel_name += "_bufw";
  }
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  return VK_KERNEL_FROM_STR(kernel_name);
}

GlobalWorkGrid pick_conv2d_pw_tiled_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  uint32_t W = graph->size_at<uint32_t>(-1, out);
  uint32_t H = graph->size_at<uint32_t>(-2, out);
  uint32_t C_out = graph->size_at<uint32_t>(-3, out);
  // One work group per output tile, channel tiles on x.
  if (const Conv2dPwCoopmatVariant* const variant =
          find_conv2d_pw_coopmat_variant(shader.kernel_name)) {
    const utils::uvec3 grid =
        conv2d_pw_coopmat_workgroups(*graph, *variant, C_out, W * H);
    return GlobalWorkGrid(
        {grid[0] * variant->wg_size(), grid[1], grid[2]}, kExplicitWorkGrid);
  }
  uint32_t N4 = utils::div_up_4(C_out);
  // Tile sizes must match conv2d_pw_tiled.yaml.
  const bool wide = is_wide_tile_shader(shader);
  const uint32_t tile_n4 = wide ? 2u : 1u;
  const uint32_t tile_h = wide ? 2u : 1u;
  const uint32_t n_tiles = utils::div_up(N4, tile_n4);
  const uint32_t m_tiles = utils::div_up(W, kTileW) * utils::div_up(H, tile_h);
  // The wide tile variant puts pixel tiles on x.
  if (wide) {
    return GlobalWorkGrid({m_tiles, n_tiles, 1u}, kTiledWorkGrid);
  }
  return GlobalWorkGrid({n_tiles, m_tiles, 1u}, kTiledWorkGrid);
}

/*
 * Wide tile work groups span the channel tiles of a row of pixel tiles, so the
 * input they share is read through one core's cache: y covers the channel tiles
 * (rounded up to a power of two) while x keeps at least min_x pixel tiles, and
 * x takes the remaining threads. Row-aligned work groups instead take more
 * threads, cover every channel tile in y, and pick an x that divides the row of
 * pixel tiles, which can leave fewer than min_x pixel tiles. Pixel tiles are
 * dispatched on x alone, so x also grows where the thread count allows to keep
 * the work group count on x within the device limit. Row-aligned work groups
 * are only taken where their x can do so.
 *
 * The thread count and min_x per GPU family come from sweeping 12 local sizes
 * (32 to 512 threads with min_x 4 to 32, and an 8x8 square) over 57 Float and
 * Half 1x1 convolutions with 7x7 to 1008x756 outputs on Mali, Adreno and
 * Xclipse phones. On the large convolutions, Mali was 5-8% faster with 512
 * threads than with 128, Adreno was 1-10% slower with 512 than with 128, and
 * Xclipse was 30% slower with 128 than with 64.
 */
LocalWorkGroup pick_conv2d_pw_tiled_lwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const GlobalWorkGrid& gwg,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  if (const Conv2dPwCoopmatVariant* const variant =
          find_conv2d_pw_coopmat_variant(shader.kernel_name)) {
    return LocalWorkGroup(variant->wg_size(), 1u, 1u);
  }
  if (is_wide_tile_shader(shader)) {
    const vkapi::Adapter* const adapter = graph->context()->adapter_ptr();
    const auto fit_threads = [adapter](uint32_t nthreads) {
      while (nthreads > adapter->max_compute_workgroup_invocations() ||
             nthreads > adapter->max_compute_workgroup_size()[0]) {
        nthreads /= 2u;
      }
      return nthreads;
    };
    const uint32_t min_count_x =
        utils::div_up(gwg[0], adapter->max_compute_workgroup_count()[0]);
    if (use_row_aligned_lwg(graph, args.at(1).refs.at(0), out, gwg)) {
      const uint32_t max_x = fit_threads(kRowAlignedThreads) / gwg[1];
      if (max_x >= min_count_x) {
        const uint32_t row_tiles =
            utils::div_up(graph->size_at<uint32_t>(-1, out), kTileW);
        const uint32_t x = pick_row_divisor(row_tiles, max_x);
        return LocalWorkGroup(x >= min_count_x ? x : max_x, gwg[1], 1u);
      }
    }
    const bool mali = graph->device_is_mali();
    const bool adreno = graph->device_is_adreno();
    const uint32_t nthreads = fit_threads(mali ? 512u : adreno ? 128u : 64u);
    uint32_t min_x = mali || adreno ? 16u : 4u;
    while (min_x < min_count_x && min_x < nthreads) {
      min_x *= 2u;
    }
    // y stays at or below nthreads / min_x (32 at most), within the 128 that
    // Vulkan guarantees for the second work group dimension.
    uint32_t y = 1u;
    while (y < gwg[1] && y * min_x < nthreads) {
      y *= 2u;
    }
    return LocalWorkGroup(nthreads / y, y, 1u);
  }
  // On Mali, a work group of pixel tiles that share one channel tile lets each
  // warp read a single weight stream instead of one per channel tile.
  if (graph->device_is_mali() && gwg[1] >= 64u) {
    return LocalWorkGroup(1u, 64u, 1u);
  }
  return pick_xy_square_lwg(graph, shader, gwg, args, resize_args);
}

//
// Prepack nodes
//

struct PackParams {
  int32_t N;
  int32_t K;
  int32_t B;
  int32_t is_transposed;
};

utils::StorageType conv2d_pw_weight_storage(
    ComputeGraph& graph,
    const ValueRef weight_data,
    const ValueRef out) {
  const std::vector<int64_t> weight_sizes = graph.sizes_of(weight_data);
  const int64_t output_height = utils::div_up(weight_sizes.at(1), int64_t(4));
  const int64_t output_width =
      utils::div_up(weight_sizes.at(0), int64_t(4)) * 4 * 4;
  // Buffer weights are faster than texture weights on every GPU measured. The
  // folded accumulation reads texture weights, the only storage it was measured
  // with. Half buffers need 16-bit storage support.
  const uint32_t max_extent =
      graph.context()->adapter_ptr()->max_texture2d_dim();
  const bool buffer_weights = !use_folded_accumulation(&graph, out) &&
      (graph.dtype_of(weight_data) != vkapi::kHalf ||
       graph.float16_buffers_enabled());
  if (buffer_weights || output_width / 4 > max_extent ||
      static_cast<uint32_t>(output_height) > max_extent) {
    return utils::kBuffer;
  }
  return utils::kTexture2D;
}

// `scratch_rows` extra rows after the K4 weight rows hold the partial sums of a
// split-K variant (conv2d_pw_coopmat_scratch_rows); the prepack writes only the
// weights.
ValueRef prepack_conv2d_pw_weight(
    ComputeGraph& graph,
    const ValueRef weight_data,
    const ValueRef out,
    const int64_t scratch_rows = 0) {
  const std::vector<int64_t> weight_sizes = graph.sizes_of(weight_data);
  const int64_t N = weight_sizes.at(0); // C_out
  const int64_t K = weight_sizes.at(1); // C_in
  const int64_t N4 = utils::div_up(N, int64_t(4));
  const int64_t K4 = utils::div_up(K, int64_t(4));

  const int64_t output_height = K4 + scratch_rows;
  const int64_t output_width = N4 * 4 * 4;

  const utils::StorageType weight_storage =
      conv2d_pw_weight_storage(graph, weight_data, out);

  ValueRef packed_weight = graph.add_tensor(
      {output_height, output_width},
      graph.dtype_of(weight_data),
      weight_storage,
      utils::kWidthPacked);

  const GlobalWorkGrid gwg(
      {utils::safe_downcast<uint32_t>(N4),
       utils::safe_downcast<uint32_t>(K4),
       1u},
      kTiledWorkGrid);

  PackParams pack_params{
      utils::safe_downcast<int32_t>(N), utils::safe_downcast<int32_t>(K), 1, 1};

  std::string pack_kernel_name = "pack_fp_linear_weight";
  add_storage_type_suffix(pack_kernel_name, weight_storage);
  add_dtype_suffix(pack_kernel_name, graph.dtype_of(packed_weight));
  add_dtype_suffix(pack_kernel_name, graph.get_staging_dtype_for(weight_data));

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR(pack_kernel_name),
      gwg,
      graph.create_lwg(gwg),
      weight_data,
      packed_weight,
      {},
      {},
      {PushConstantDataInfo(&pack_params, sizeof(PackParams))}));

  return packed_weight;
}

//
// Dispatch nodes
//

void add_conv2d_pw_tiled_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef packed_weight,
    const ValueRef packed_bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef out,
    const ValueRef weight_data,
    const Conv2dEpilogue& epilogue,
    const Conv2dEpilogueArgs& epilogue_args,
    const Conv2dPwRequest& request,
    const bool transposed,
    const bool has_bias) {
  int32_t stride_h, stride_w, padding_h, padding_w;
  {
    const auto stride_list = graph.get_int_list(stride);
    const auto padding_list = graph.get_int_list(padding);
    stride_h = utils::safe_downcast<int32_t>(stride_list->at(0));
    stride_w = utils::safe_downcast<int32_t>(stride_list->at(1));
    padding_h = utils::safe_downcast<int32_t>(padding_list->at(0));
    padding_w = utils::safe_downcast<int32_t>(padding_list->at(1));
  }

  bool s1p0 =
      stride_h == 1 && stride_w == 1 && padding_h == 0 && padding_w == 0;

  utils::ivec4 stride_padding{stride_h, stride_w, padding_h, padding_w};

  const Conv2dEpilogueParams epilogue_params = create_conv2d_epilogue_params(
      epilogue, graph.dtype_of(out) == vkapi::kHalf);
  // The conv2d_pw_coopmat variants that accumulate in fp32 also compute the
  // epilogue in fp32; they read this unrounded copy, pushed after the params of
  // the shaders that compute it in f16 (tiled and fp16-accumulation variants),
  // so whichever shader a resize picks finds its params.
  const Conv2dEpilogueParams epilogue_params_f32 =
      create_conv2d_epilogue_params(epilogue, /*f16_epilogue=*/false);

  vkapi::SpecVarList spec_vars = {s1p0 ? 1 : 0};
  append_conv2d_epilogue_spec_vars(spec_vars, epilogue);
  // The weight storage and folded_accumulation are fixed here, but the tile is
  // picked again on resize. Only the wide tile texture weight Half variant
  // declares folded_accumulation; the other variants ignore it.
  spec_vars.append(vkapi::SpecVar(use_folded_accumulation(&graph, out)));

  const vkapi::MemoryAccessFlags in_access =
      request.kind == Conv2dPwRequest::Kind::kCoopmat &&
          request.coopmat_variant->has(Conv2dPwCoopmatVariant::kImageInput)
      ? vkapi::kReadWrite
      : vkapi::kRead;

  const bool has_bias_or_epilogue = has_bias || epilogue.has_activation() ||
      epilogue.has_residual() || epilogue.has_scale();

  if (request.kind == Conv2dPwRequest::Kind::kCoopmat &&
      request.coopmat_variant->k_slices > 1u) {
    // Split-K (design D6): a kernel node that writes the partial sums to the
    // scratch after the packed weights, and a node that merges them, both with
    // the tiled kernel's bindings. While the convolution is not eligible for
    // the variant, the kernel node dispatches nothing and the merge node runs
    // the tiled kernel instead.
    const Conv2dPwCoopmatVariant& variant = *request.coopmat_variant;
    const auto eligible = [request, transposed, has_bias_or_epilogue](
                              ComputeGraph* graph,
                              const std::vector<ArgGroup>& args,
                              const std::vector<ValueRef>& resize_args) {
      return conv2d_pw_coopmat_eligible(
          *graph,
          *request.coopmat_variant,
          args.at(1).refs.at(0),
          resize_args.at(0),
          args.at(2).refs.at(0),
          resize_args.at(1),
          resize_args.at(2),
          args.at(0).refs.at(0),
          transposed,
          has_bias_or_epilogue);
    };
    graph.execute_nodes().emplace_back(new DynamicDispatchNode(
        graph,
        [name = std::string(variant.name)](
            ComputeGraph*,
            const std::vector<ArgGroup>&,
            const std::vector<ValueRef>&) { return VK_KERNEL_FROM_STR(name); },
        [eligible](
            ComputeGraph* graph,
            const vkapi::ShaderInfo& shader,
            const std::vector<ArgGroup>& args,
            const std::vector<ValueRef>& resize_args) {
          if (!eligible(graph, args, resize_args)) {
            return GlobalWorkGrid({0u, 0u, 0u}, kExplicitWorkGrid);
          }
          return pick_conv2d_pw_tiled_gwg(graph, shader, args, resize_args);
        },
        pick_conv2d_pw_tiled_lwg,
        {{out, vkapi::kWrite},
         {in, in_access},
         {packed_weight, vkapi::kReadWrite},
         {packed_bias, vkapi::kRead},
         {epilogue_args.residual,
          epilogue_args.residual == in ? in_access : vkapi::kRead},
         {epilogue_args.scale, vkapi::kRead}},
        {graph.sizes_ubo(in), graph.sizes_ubo(out)},
        {PushConstantDataInfo(&stride_padding, sizeof(stride_padding)),
         PushConstantDataInfo(&epilogue_params, sizeof(epilogue_params)),
         PushConstantDataInfo(
             &epilogue_params_f32, sizeof(epilogue_params_f32))},
        spec_vars,
        {weight_data, stride, padding, epilogue.residual},
        resize_conv2d_pw_tiled_node));
    const std::string merge = conv2d_pw_coopmat_merge_shader(variant);
    graph.execute_nodes().emplace_back(new DynamicDispatchNode(
        graph,
        [eligible, merge, transposed, has_bias_or_epilogue](
            ComputeGraph* graph,
            const std::vector<ArgGroup>& args,
            const std::vector<ValueRef>& resize_args) {
          if (eligible(graph, args, resize_args)) {
            return VK_KERNEL_FROM_STR(merge);
          }
          return pick_conv2d_pw_tiled_shader(
              graph, args, resize_args, {}, transposed, has_bias_or_epilogue);
        },
        [merge](
            ComputeGraph* graph,
            const vkapi::ShaderInfo& shader,
            const std::vector<ArgGroup>& args,
            const std::vector<ValueRef>& resize_args) {
          if (shader.kernel_name != merge) {
            return pick_conv2d_pw_tiled_gwg(graph, shader, args, resize_args);
          }
          // One invocation per output texel.
          const ValueRef out = args.at(0).refs.at(0);
          const uint32_t texels = graph->size_at<uint32_t>(-1, out) *
              graph->size_at<uint32_t>(-2, out) *
              utils::div_up_4(graph->size_at<uint32_t>(-3, out));
          return GlobalWorkGrid(
              {utils::div_up(texels, kMergeThreads) * kMergeThreads, 1u, 1u},
              kExplicitWorkGrid);
        },
        [merge](
            ComputeGraph* graph,
            const vkapi::ShaderInfo& shader,
            const GlobalWorkGrid& gwg,
            const std::vector<ArgGroup>& args,
            const std::vector<ValueRef>& resize_args) {
          if (shader.kernel_name != merge) {
            return pick_conv2d_pw_tiled_lwg(
                graph, shader, gwg, args, resize_args);
          }
          return LocalWorkGroup(kMergeThreads, 1u, 1u);
        },
        {{out, vkapi::kWrite},
         {in, in_access},
         {{packed_weight, packed_bias}, vkapi::kRead},
         {epilogue_args.residual,
          epilogue_args.residual == in ? in_access : vkapi::kRead},
         {epilogue_args.scale, vkapi::kRead}},
        {graph.sizes_ubo(in), graph.sizes_ubo(out)},
        {PushConstantDataInfo(&stride_padding, sizeof(stride_padding)),
         PushConstantDataInfo(&epilogue_params, sizeof(epilogue_params)),
         PushConstantDataInfo(
             &epilogue_params_f32, sizeof(epilogue_params_f32))},
        spec_vars,
        {weight_data, stride, padding, epilogue.residual},
        resize_conv2d_pw_tiled_node));
    return;
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      [request, transposed, has_bias_or_epilogue](
          ComputeGraph* graph,
          const std::vector<ArgGroup>& args,
          const std::vector<ValueRef>& resize_args) {
        return pick_conv2d_pw_tiled_shader(
            graph,
            args,
            resize_args,
            request,
            transposed,
            has_bias_or_epilogue);
      },
      pick_conv2d_pw_tiled_gwg,
      pick_conv2d_pw_tiled_lwg,
      // Inputs and Outputs. A storage-image input (_img) must be in
      // VK_IMAGE_LAYOUT_GENERAL, which read-write access gives it; the tiled
      // fallback samples it in that layout as well. Without a residual the
      // input is also bound as the residual placeholder, with the same access
      // so that its layout stays GENERAL.
      {{out, vkapi::kWrite},
       {in, in_access},
       {{packed_weight, packed_bias}, vkapi::kRead},
       {epilogue_args.residual,
        epilogue_args.residual == in ? in_access : vkapi::kRead},
       {epilogue_args.scale, vkapi::kRead}},
      // Shader params buffers
      {graph.sizes_ubo(in), graph.sizes_ubo(out)},
      // Push Constants
      {PushConstantDataInfo(&stride_padding, sizeof(stride_padding)),
       PushConstantDataInfo(&epilogue_params, sizeof(epilogue_params)),
       PushConstantDataInfo(&epilogue_params_f32, sizeof(epilogue_params_f32))},
      // Specialization Constants
      spec_vars,
      // Resize Args
      {weight_data, stride, padding, epilogue.residual},
      // Resizing Logic
      resize_conv2d_pw_tiled_node));
}

//
// High level operator impl
//

void conv2d_pw_impl(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef out,
    const bool transposed_val,
    const Conv2dEpilogue& epilogue,
    const Conv2dPwRequest& request) {
  VK_CHECK_COND(
      (request.kind == Conv2dPwRequest::Kind::kCoopmat) ==
          (request.coopmat_variant != nullptr),
      "conv2d_pw: a coopmat request names exactly one variant");
  if (request.kind == Conv2dPwRequest::Kind::kCoopmat) {
    // An explicit request for a convolution the variant cannot run is
    // unsupported; only a later resize falls back to the tiled kernel.
    const bool has_bias_or_epilogue = !graph.val_is_none(bias) ||
        epilogue.has_activation() || epilogue.has_residual() ||
        epilogue.has_scale();
    if (const auto reason = conv2d_pw_coopmat_ineligible(
            graph,
            *request.coopmat_variant,
            in,
            weight_data,
            conv2d_pw_weight_storage(graph, weight_data, out),
            stride,
            padding,
            out,
            transposed_val,
            has_bias_or_epilogue)) {
      VK_THROW(
          "conv2d_pw: ",
          request.coopmat_variant->name,
          " cannot run this convolution: ",
          *reason);
    }
  }
  const bool split_k = request.kind == Conv2dPwRequest::Kind::kCoopmat &&
      request.coopmat_variant->k_slices > 1u;
  ValueRef packed_weight = prepack_conv2d_pw_weight(
      graph,
      weight_data,
      out,
      split_k ? conv2d_pw_coopmat_scratch_rows(
                    *request.coopmat_variant,
                    conv2d_pw_coopmat_scratch_capacity(graph, out))
              : 0);

  ValueRef packed_bias = prepack_biases(
      graph,
      bias,
      weight_data,
      transposed_val,
      utils::kTexture2D,
      utils::kWidthPacked);
  const ValueRef packed_scale = prepack_conv2d_residual_scale(
      graph, epilogue, weight_data, transposed_val);

  check_conv_args(graph, in, out);

  const std::vector<int64_t> in_sizes = graph.sizes_of(in);
  if (in_sizes.at(0) > 1) {
    VK_THROW("conv2d: input batch size > 1 is not supported yet!");
  }

  add_conv2d_pw_tiled_node(
      graph,
      in,
      packed_weight,
      packed_bias,
      stride,
      padding,
      out,
      weight_data,
      epilogue,
      Conv2dEpilogueArgs(epilogue, in, packed_bias, packed_scale),
      request,
      transposed_val,
      /*has_bias=*/!graph.val_is_none(bias));
}

} // namespace vkcompute
