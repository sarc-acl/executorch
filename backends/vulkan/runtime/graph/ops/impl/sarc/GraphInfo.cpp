/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/GraphInfo.h>

#include <algorithm>
#include <cctype>

namespace vkcompute {
namespace sarc {

Storage to_storage(const utils::StorageType s) {
  switch (s) {
    case utils::kBuffer:
      return Storage::kBuffer;
    case utils::kTexture2D:
      return Storage::kTexture2D;
    case utils::kTexture3D:
      return Storage::kTexture3D;
    default:
      return Storage::kOther;
  }
}

DeviceInfo device_info(ComputeGraph* graph) {
  const auto* adapter = graph->context()->adapter_ptr();
  DeviceInfo d;
  d.name = graph->device_name();
  std::transform(d.name.begin(), d.name.end(), d.name.begin(), [](char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  });
  d.is_amd = graph->device_is_amd();
  d.subgroup_size = adapter->subgroup_size();
  d.min_subgroup_size = adapter->min_subgroup_size();
  d.max_subgroup_size = adapter->max_subgroup_size();
  d.subgroup_size_control = adapter->supports_subgroup_size_control();
  d.coopmat = adapter->supports_cooperative_matrix();
  d.int8_coopmat = adapter->supports_int8_cooperative_matrix();
  d.max_shared_bytes = adapter->max_compute_shared_memory_size();
  return d;
}

utils::StorageType predicted_q4_weight_storage(
    ComputeGraph& graph,
    const ValueRef weight_data) {
  const std::vector<int64_t> sizes = graph.sizes_of(weight_data);
  const int64_t K = utils::val_at(-1, sizes) * 2;
  const int64_t N = utils::val_at(-2, sizes);
  const int64_t height = utils::div_up(N, int64_t(8));
  const int64_t width = utils::div_up(K, int64_t(4)) * 4;
  const uint32_t max_extent =
      graph.context()->adapter_ptr()->max_texture2d_dim();
  return (width > int64_t(max_extent) * 4 || height > int64_t(max_extent))
      ? utils::kBuffer
      : utils::kTexture2D;
}

} // namespace sarc
} // namespace vkcompute
