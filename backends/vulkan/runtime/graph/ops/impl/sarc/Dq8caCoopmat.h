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

// Hook for et_vk.linear_dq8ca_q4gsw (QuantizedLinear.cpp). If a SARC row
// applies to the op's build-time shape, builds the whole op (activation
// quantize/pack + int8 coopmat linear) on the SARC path and returns true;
// otherwise returns false and the caller keeps the upstream path. `args`:
// {input, input_scale, input_zp, weight, weight_sums, weight_scales,
//  group_size, bias, output}.
bool try_add_dq8ca_coopmat(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args);

} // namespace sarc
} // namespace vkcompute
