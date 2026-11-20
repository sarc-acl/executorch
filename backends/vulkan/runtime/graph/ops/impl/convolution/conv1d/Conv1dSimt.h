/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <optional>
#include <string>
#include <vector>

namespace vkcompute {

// A conv1d_simt shader variant (conv1d_simt.yaml, change
// conv2d-all-used-kernels): pw for the height-packed conv1d_pw role (kernel
// 1, stride 1, padding 0), otherwise the channels-packed conv1d role. The
// shader name is `name` plus the dtype suffix.
struct Conv1dSimtVariant final {
  const char* name;
  uint32_t tile_m;
  uint32_t tile_n;
  uint32_t tile_k;
  uint32_t tm;
  uint32_t tn4;
  uint32_t k_slices;
  bool pw;

  uint32_t wg_size() const {
    return (tile_m / tm) * (tile_n / (4u * tn4));
  }
};

const std::vector<Conv1dSimtVariant>& conv1d_simt_variants();

// The variant of a shader name with or without its dtype suffix, or nullptr.
const Conv1dSimtVariant* find_conv1d_simt_variant(const std::string& name);

// conv1d arguments: in, weight, bias, stride, padding, dilation, groups,
// out_min (or kDummyValueRef), out_max (or kDummyValueRef), out.
std::optional<std::string> conv1d_simt_ineligible(
    ComputeGraph& graph,
    const Conv1dSimtVariant& variant,
    const std::vector<ValueRef>& args);

void conv1d_simt_impl(
    ComputeGraph& graph,
    const Conv1dSimtVariant& variant,
    const std::vector<ValueRef>& args);

// With ETVK_CONV2D_COOPMAT=1: the variant of this device's table row
// (precision fp16 or fp32 by dtype, configuration 1 x k on a 1 x L input),
// if any and eligible.
const Conv1dSimtVariant* conv1d_simt_table_variant(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args);

} // namespace vkcompute
