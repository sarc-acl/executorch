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

// A q8ta_conv2d_pw_coopmat shader variant (q8ta_conv2d_pw_coopmat.yaml).
// direct: the convolution input is gathered per K step (any kernel, stride,
// padding and dilation, groups 1, input channels a multiple of 4, or any
// count with bytes, which gathers each K byte alone); otherwise
// the input is read as the A matrix of an ungrouped 1x1, stride 1, padding 0
// convolution.
struct Q8taConv2dCoopmatVariant final {
  const char* name;
  uint32_t mma_m;
  uint32_t mma_n;
  uint32_t mma_k;
  uint32_t tile_m;
  uint32_t tile_n;
  uint32_t tile_k;
  uint32_t sg_grid_x;
  uint32_t sg_grid_y;
  uint32_t subgroup_size;
  uint32_t k_slices;
  bool direct;
  bool bytes;

  uint32_t wg_size() const {
    return sg_grid_x * sg_grid_y * subgroup_size;
  }
  uint32_t shared_bytes() const;
};

const std::vector<Q8taConv2dCoopmatVariant>& q8ta_conv2d_coopmat_variants();

const Q8taConv2dCoopmatVariant* find_q8ta_conv2d_coopmat_variant(
    const std::string& name);

// Why the variant cannot run the q8ta_conv2d / q8ta_conv2d_pw call with these
// arguments (the operators' 16 arguments), or nullopt.
std::optional<std::string> q8ta_conv2d_coopmat_ineligible(
    ComputeGraph& graph,
    const Q8taConv2dCoopmatVariant& variant,
    const std::vector<ValueRef>& args);

// Adds the prepacks and dispatches of the variant; the call must be eligible.
// direct_requant: requantize as the direct q8ta_conv2d shader does (the
// route stock takes for the call) instead of as q8ta_conv2d_pw does, so the
// output equals stock bit for bit either way.
void q8ta_conv2d_coopmat_impl(
    ComputeGraph& graph,
    const Q8taConv2dCoopmatVariant& variant,
    const std::vector<ValueRef>& args,
    const bool direct_requant);

// With ETVK_CONV2D_COOPMAT=1: the variant of the table row (precision int8)
// of this device and configuration, if any and eligible.
const Q8taConv2dCoopmatVariant* q8ta_conv2d_coopmat_table_variant(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args);

} // namespace vkcompute
