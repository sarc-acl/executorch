/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Glue between ComputeGraph and the Vulkan-free selection layer (Select.h).

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

namespace vkcompute {
namespace sarc {

Storage to_storage(utils::StorageType s);

DeviceInfo device_info(ComputeGraph* graph);

// The storage prepack_quantized_linear_weight() (QuantizedLinear.cpp) picks
// for a 4-bit weight: texture2d unless the packed extents exceed the limit.
utils::StorageType predicted_q4_weight_storage(
    ComputeGraph& graph,
    ValueRef weight_data);

} // namespace sarc
} // namespace vkcompute
