/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// SARC cooperative-matrix SDPA (LLM mode, prefill): QK^T and attn*V coopmat
// kernels plus a causally truncated softmax. Entry points are called from
// release 1.5's SDPA.cpp (sarc/HOOKS); each returns nothing (std::nullopt /
// the upstream value) on devices without an active SDPA row, so upstream
// behaviour is unchanged there.

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/ExecuteNode.h>

#include <optional>

namespace vkcompute {
namespace sarc {

// Pickers: args/resize_args as in pick_sdpa_{qk,av}_shader (LLM mode).
std::optional<vkapi::ShaderInfo> pick_sdpa_qk(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);
std::optional<vkapi::ShaderInfo> pick_sdpa_av(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

// Launch geometry for a SARC SDPA kernel (by name), else std::nullopt.
std::optional<GlobalWorkGrid> sdpa_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ValueRef>& resize_args);

// Spec constants for the QK / AV nodes. `upstream` is release 1.5's list; it
// is returned unchanged unless the device has an SDPA row, in which case the
// SARC kernels' constants are appended after it.
vkapi::SpecVarList sdpa_qk_spec_vars(
    ComputeGraph& graph,
    ValueRef q,
    ValueRef input_pos_symint,
    const vkapi::SpecVarList& upstream);
vkapi::SpecVarList sdpa_av_spec_vars(
    ComputeGraph& graph,
    ValueRef q,
    ValueRef v,
    const vkapi::SpecVarList& upstream);

// The softmax shader name for LLM mode: the SARC truncated copy on devices
// with an SDPA row, else `upstream_name`.
std::string sdpa_softmax_shader_name(
    ComputeGraph& graph,
    const std::string& upstream_name);

} // namespace sarc
} // namespace vkcompute
