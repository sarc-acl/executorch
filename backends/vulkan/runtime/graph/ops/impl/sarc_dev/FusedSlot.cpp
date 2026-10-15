/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SARC development zone: see FusedSlot.h. GPU-free (test_sarc_select links
// it). Not part of a release.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc_dev/FusedSlot.h>

#include <cstdlib>
#include <iostream>

namespace vkcompute {
namespace sarc {

FusedSlots& fused_slots() {
  // Function-local: usable from any file's static initializer.
  static FusedSlots slots;
  return slots;
}

void register_fused_node(FusedNode node, FusedAdd add, FusedServes serves) {
  fused_slots().put(node, add, serves);
}

const char* fused_node_name(int node) {
  static const char* const names[kFusedNodes] = {
      "780m (ET_VK_SARC_780M_PROFILE / ET_VK_SARC_780M_SDPA_FUSED)",
      "4070ti (ET_VK_SARC_DEV_PROFILE=4070ti-fused*)",
      "b580/b70 (ET_VK_SARC_DEV_PROFILE=b580-fused* / b70-fused*)",
      "orin (ET_VK_SARC_DEV_PROFILE=orin-fused* / ET_VK_SARC_ORIN_SDPA_FUSED)",
      "m51 (ET_VK_SARC_M51_PROFILE / ET_VK_SARC_M51_SDPA_FUSED)",
  };
  return node >= 0 && node < kFusedNodes ? names[node] : "none";
}

int resolve_fused_node(const char* const (&lists)[kFusedNodes]) {
  int found = -1;
  int count = 0;
  for (int i = 0; i < kFusedNodes; ++i) {
    if (lists[i] != nullptr && *lists[i] != 0) {
      found = i;
      ++count;
    }
  }
  if (count > 1) {
    std::cerr << "[sarc_dev] more than one fused attention node requested:";
    for (int i = 0; i < kFusedNodes; ++i) {
      if (lists[i] != nullptr && *lists[i] != 0) {
        std::cerr << "\n  " << fused_node_name(i) << ": " << lists[i];
      }
    }
    std::cerr << "\n[sarc_dev] set the variables of one device only"
              << std::endl;
    std::abort();
  }
  return found;
}

void fused_dispatch_add(ComputeGraph& graph, const std::vector<int32_t>& refs) {
  const FusedEntry e = fused_slots().active();
  if (e.add != nullptr) {
    e.add(graph, refs);
  }
}

bool fused_dispatch_serves(
    ComputeGraph* graph,
    const std::vector<int32_t>& resize_args) {
  const FusedEntry e = fused_slots().active();
  return e.serves != nullptr && e.serves(graph, resize_args);
}

} // namespace sarc
} // namespace vkcompute
