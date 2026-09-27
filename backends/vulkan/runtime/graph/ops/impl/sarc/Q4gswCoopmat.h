/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace vkcompute {
namespace sarc {

// Hook for et_vk.linear_q4gsw / et_vk.q4gsw_linear (Q4gswLinear.cpp). If the
// device has an active SARC row for 4w linear, builds the op on the SARC path
// and returns true; otherwise returns false and the caller keeps the upstream
// path unchanged. `args` is the op's argument list:
// {input, weight, weight_scales, group_size, bias, output}.
bool try_add_q4gsw_coopmat(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args);

} // namespace sarc
} // namespace vkcompute
