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
#include <cstring>
#include <iostream>
#include <string>

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

namespace {

// Retired fused kernels (owner decision 2026-10-15, M5 review): the 780M's
// fused2 and fused3 read shared-memory slots written by other lanes of the
// subgroup, ordered only by memoryBarrierShared() and no subgroupBarrier()
// (LESSONS.md L8). fused3sb is fused3 with the barriers; the RX 7600 and
// RX 7900 XTX use the 780M node, so the retirement covers them as well.
struct Retired {
  FusedNode node;
  const char* prefix;
  const char* use_instead;
};
const Retired kRetired[] = {
    {FusedNode::k780m, "fused2_", "fused3sb_ (ET_VK_SARC_DEV_PROFILE=780m-final, 780M profile c11)"},
    {FusedNode::k780m, "fused3_", "the fused3sb_ variant of the same tile (780M profile c11)"},
};

} // namespace

const char* retired_fused_variant(int node, const char* list) {
  if (list == nullptr) {
    return nullptr;
  }
  for (const Retired& r : kRetired) {
    if (static_cast<int>(r.node) != node) {
      continue;
    }
    // A variant starts the list or follows a comma.
    for (const char* p = list; p != nullptr && *p != 0;) {
      if (std::strncmp(p, r.prefix, std::strlen(r.prefix)) == 0) {
        return r.prefix;
      }
      p = std::strchr(p, ',');
      p = p != nullptr ? p + 1 : nullptr;
    }
  }
  return nullptr;
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
  const char* prefix =
      found >= 0 ? retired_fused_variant(found, lists[found]) : nullptr;
  if (prefix != nullptr) {
    const char* instead = "";
    for (const Retired& r : kRetired) {
      if (static_cast<int>(r.node) == found &&
          std::strcmp(r.prefix, prefix) == 0) {
        instead = r.use_instead;
      }
    }
    std::cerr << "[sarc_dev] fused attention: the " << fused_node_name(found)
              << " variants " << lists[found] << " name the retired family "
              << std::string(prefix, std::strlen(prefix) - 1)
              << " (shared-memory reads without subgroupBarrier(), LESSONS.md "
                 "L8; retired 2026-10-15). Use "
              << instead << "." << std::endl;
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
