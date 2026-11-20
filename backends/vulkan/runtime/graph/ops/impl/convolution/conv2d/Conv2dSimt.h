/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2d.h>

#include <optional>
#include <string>
#include <vector>

namespace vkcompute {

// A conv2d_simt shader variant (conv2d_simt.yaml): the FP32 register-tiled
// GEMM of change conv2d-all-used-kernels. direct: the input is gathered per K
// step (any kernel, stride, padding and dilation); otherwise the convolution
// must be 1x1, stride 1, padding 0.
struct Conv2dSimtVariant final {
  const char* name;
  uint32_t tile_m;
  uint32_t tile_n;
  uint32_t tile_k;
  uint32_t tm;
  uint32_t tn4;
  uint32_t k_slices;
  bool direct;

  uint32_t wg_size() const {
    return (tile_m / tm) * (tile_n / (4u * tn4));
  }
  uint32_t shared_bytes() const {
    return 2u * tile_k * (tile_m + tile_n) * 4u;
  }
};

const std::vector<Conv2dSimtVariant>& conv2d_simt_variants();

const Conv2dSimtVariant* find_conv2d_simt_variant(const std::string& name);

// Why the variant cannot run the convolution, or nullopt. `bias` may be none.
std::optional<std::string> conv2d_simt_ineligible(
    ComputeGraph& graph,
    const Conv2dSimtVariant& variant,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const int64_t groups,
    const bool transposed,
    const ValueRef out);

void conv2d_simt_impl(
    ComputeGraph& graph,
    const Conv2dSimtVariant& variant,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef out,
    const Conv2dEpilogue& epilogue);

} // namespace vkcompute
