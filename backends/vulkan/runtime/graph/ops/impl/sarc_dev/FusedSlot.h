/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SARC development zone: the one fused attention node a dev build runs
// (Override::sdpa_fused_*, the release-zone entry of the owner decision of
// 2026-10-05). Not part of a release.
//
// Each fused node file (impl/sarc_dev/<device>/Sdpa<Device>Fused.cpp) stores
// its entry points here from its static initializer, and Overrides.cpp
// records which node the profile variables request from its own. The two
// writes go to separate places, and the functions installed into the Override
// look the node up when they are called, after every static initializer has
// run. So the order of the initializers of different files, which C++ leaves
// undefined, cannot change which node runs (M2 review, 2026-10-15). Two
// requested nodes abort at start-up.

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

#include <cstdint>
#include <vector>

// test_sarc_select declares these weak, so that it also links without the
// development zone.
#ifndef SARC_DEV_DECL
#define SARC_DEV_DECL
#endif

namespace vkcompute {
namespace sarc {

using FusedAdd = void (*)(ComputeGraph&, const std::vector<int32_t>&);
using FusedServes = bool (*)(ComputeGraph*, const std::vector<int32_t>&);

// The fused attention nodes of the development zone. The Arc B580 and the Arc
// Pro B70 share one node (b580/SdpaB580Fused.cpp); the RX 7600 and RX 7900 XTX
// use the 780M's.
enum class FusedNode : int { k780m, k4070ti, kXe2, kOrin, kM51, kCount };
constexpr int kFusedNodes = static_cast<int>(FusedNode::kCount);

struct FusedEntry {
  FusedAdd add = nullptr;
  FusedServes serves = nullptr;
};

struct FusedSlots {
  FusedEntry entries[kFusedNodes] = {};
  int requested = -1; // a FusedNode, or -1: none

  void put(FusedNode node, FusedAdd add, FusedServes serves) {
    entries[static_cast<int>(node)] = {add, serves};
  }
  void request(int node) {
    requested = node;
  }
  // The requested node's entry points; empty while none is requested or the
  // requested node's file is not linked.
  FusedEntry active() const {
    return requested >= 0 && requested < kFusedNodes ? entries[requested]
                                                     : FusedEntry{};
  }
};

// The process's slots.
SARC_DEV_DECL FusedSlots& fused_slots();
// For the fused node files: store the node's entry points (no other effect).
SARC_DEV_DECL void
register_fused_node(FusedNode node, FusedAdd add, FusedServes serves);
// The node whose variant list (one per FusedNode; null or "" = not requested)
// is non-empty, or -1. More than one: prints them and aborts.
SARC_DEV_DECL int resolve_fused_node(const char* const (&lists)[kFusedNodes]);
SARC_DEV_DECL const char* fused_node_name(int node);
// Installed into the Override by Overrides.cpp when a node is requested:
// forward to fused_slots().active() at call time.
SARC_DEV_DECL void fused_dispatch_add(
    ComputeGraph& graph,
    const std::vector<int32_t>& refs);
SARC_DEV_DECL bool fused_dispatch_serves(
    ComputeGraph* graph,
    const std::vector<int32_t>& resize_args);

} // namespace sarc
} // namespace vkcompute
