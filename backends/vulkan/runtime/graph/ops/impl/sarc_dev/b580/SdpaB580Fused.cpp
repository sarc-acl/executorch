/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// b580-fused (openspec/changes/sarc-1.5-b580-fused-port): the node of the fused
// prefill SDPA kernel glsl/sarc_dev/sarc_dev_b580_sdpa_fused and of the copy
// pass sarc_dev_b580_sdpa_kvt that writes its tile-packed K and V. The kernels
// come from the profile alone: ET_VK_SARC_DEV_PROFILE=b580-fused1 (or a
// b580-fused-<variant> screening profile) names one variant per head_dim
// (Overrides.cpp, b580-fused block); there is no second variable. For the
// calls a variant fits (below) the three SDPA kernels dispatch nothing and
// this node writes the output; every other call (decode, unaligned prompt, a
// head_dim without a variant) is untouched.
// Entry point: Override::sdpa_fused_{serves,add} (impl/sarc/Select.h), handed
// to Overrides.cpp at the end of this file. In a subdirectory because
// impl/sarc_dev/*.cpp must stay GPU-free (sarc/tools/check.sh compiles those
// into test_sarc_select). Adapted from the 780M campaign's
// impl/sarc_dev/780m/Sdpa780mFused.cpp.

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/SDPA.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/GraphInfo.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>

namespace vkcompute {
namespace sarc {

// Overrides.cpp, b580-fused block.
const char* sdpa_fused_variants_b580();
void register_sdpa_fused_b580(
    void (*add)(ComputeGraph&, const std::vector<ValueRef>&),
    bool (*serves)(ComputeGraph*, const std::vector<ValueRef>&));

namespace {

// One workgroup is one subgroup of `sg` invocations and owns `m` query rows;
// a block is `n` context columns.
struct Variant {
  std::string name;
  uint32_t head_dim, m, n, sg;
};

const std::vector<Variant>& variants() {
  static const std::vector<Variant> parsed = [] {
    std::vector<Variant> out;
    std::string list = sdpa_fused_variants_b580();
    while (!list.empty()) {
      const size_t comma = list.find(',');
      const std::string token = list.substr(0, comma);
      list = comma == std::string::npos ? "" : list.substr(comma + 1);
      unsigned d, m, n, sg;
      VK_CHECK_COND(
          std::sscanf(token.c_str(), "d%u_t%ux%us%u", &d, &m, &n, &sg) == 4,
          "b580-fused: expected d<D>_t<M>x<N>s<S>m8...");
      out.push_back({"sarc_dev_b580_sdpa_fused_" + token, d, m, n, sg});
    }
    return out;
  }();
  return parsed;
}

const Variant* variant_for(ComputeGraph* graph, const ValueRef q) {
  if (variants().empty() || std::getenv("ET_VK_DISABLE_COOPMAT") != nullptr ||
      graph->storage_type_of(q) != utils::kBuffer ||
      graph->dtype_of(q) != vkapi::kHalf ||
      !device_has_active_rows(device_info(graph), Op::kSdpaQk)) {
    return nullptr;
  }
  const uint32_t head_dim = graph->size_at<uint32_t>(-1, q);
  for (const Variant& v : variants()) {
    if (v.head_dim == head_dim) {
      return &v;
    }
  }
  return nullptr;
}

bool fused_active(
    ComputeGraph* graph,
    const ValueRef q,
    const ValueRef input_pos_symint) {
  const Variant* v = variant_for(graph, q);
  if (v == nullptr || !is_valid(input_pos_symint)) {
    return false;
  }
  const uint32_t S = graph->size_at<uint32_t>(-3, q);
  const uint32_t input_pos =
      static_cast<uint32_t>(graph->read_symint(input_pos_symint));
  return S >= v->m && S % v->m == 0 && S % v->n == 0 && input_pos % v->n == 0;
}

vkapi::ShaderInfo pick_fused_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)args;
  return VK_KERNEL_FROM_STR(
      variant_for(graph, resize_args.at(0))->name + "_buffer_buffer_half");
}

// The local size is the required subgroup size of the variant, so a workgroup
// is one subgroup; the kernel checks it and writes NaN otherwise.
GlobalWorkGrid pick_fused_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)args;
  const ValueRef q = resize_args.at(0);
  const Variant* v = variant_for(graph, q);
  if (!fused_active(graph, q, resize_args.at(1))) {
    return GlobalWorkGrid(
        {0u, 0u, 0u}, kTiledWorkGrid, LocalWorkGroup(v->sg, 1u, 1u));
  }
  return GlobalWorkGrid(
      {v->sg,
       graph->size_at<uint32_t>(-3, q) / v->m,
       graph->size_at<uint32_t>(-2, q)},
      kTiledWorkGrid,
      LocalWorkGroup(v->sg, 1u, 1u));
}

GlobalWorkGrid pick_kvt_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  const ValueRef q = resize_args.at(0);
  const ValueRef input_pos_symint = resize_args.at(1);
  // In 4 x 4 blocks: x along context, y along head_dim.
  const LocalWorkGroup lwg(8u, 8u, 1u);
  if (!fused_active(graph, q, input_pos_symint)) {
    return GlobalWorkGrid({0u, 0u, 0u}, kTiledWorkGrid, lwg);
  }
  const ValueRef v = args.at(1).refs.back();
  const uint32_t context_len = graph->size_at<uint32_t>(-3, q) +
      static_cast<uint32_t>(graph->read_symint(input_pos_symint));
  return GlobalWorkGrid(
      {context_len / 4u,
       graph->size_at<uint32_t>(-1, v) / 4u,
       graph->size_at<uint32_t>(-2, v)},
      kTiledWorkGrid,
      lwg);
}

void resize_fused_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  graph->virtual_resize(
      args.at(0).refs.at(0), graph->sizes_of(resize_args.at(0)));
}

// resize_args are those of the three SDPA nodes:
// [q, k, input_pos_symint_or_dummy, mode].
bool sdpa_fused_serves_b580(
    ComputeGraph* graph,
    const std::vector<ValueRef>& resize_args) {
  return static_cast<SDPAMode>(resize_args.at(3)) == SDPAMode::LLM &&
      fused_active(graph, resize_args.at(0), resize_args.at(2));
}

void sdpa_fused_add_b580(
    ComputeGraph& graph,
    const std::vector<ValueRef>& refs) {
  const ValueRef q = refs.at(0);
  const ValueRef k = refs.at(1);
  const ValueRef v = refs.at(2);
  const ValueRef input_pos_symint = refs.at(3);
  const ValueRef out = refs.at(4);
  if (variant_for(&graph, q) == nullptr || !is_valid(input_pos_symint)) {
    return;
  }
  // fused_active() sees only q: the other tensors must share its storage.
  for (const ValueRef t : {k, v, out}) {
    VK_CHECK_COND(
        graph.storage_type_of(t) == utils::kBuffer &&
        graph.dtype_of(t) == vkapi::kHalf);
  }
  const int32_t head_dim = graph.size_at<int32_t>(-1, q);
  const int32_t num_q_heads = graph.size_at<int32_t>(-2, q);
  const int32_t num_kv_heads = graph.size_at<int32_t>(-2, v);
  const int32_t max_context = graph.size_at<int32_t>(-3, v);
  VK_CHECK_COND(max_context % 16 == 0 && head_dim % 16 == 0);

  // Tile-packed copies of the K and V caches (layout in the kvt shader).
  const std::vector<int64_t> scratch_sizes = {
      1, num_kv_heads, head_dim, max_context};
  TmpTensor kt(
      &graph, scratch_sizes, vkapi::kHalf, utils::kBuffer, utils::kWidthPacked);
  TmpTensor vt(
      &graph, scratch_sizes, vkapi::kHalf, utils::kBuffer, utils::kWidthPacked);
  const ValueRef k_packed = kt;
  const ValueRef v_packed = vt;
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR("sarc_dev_b580_sdpa_kvt_buffer_half"),
      pick_kvt_gwg,
      pick_required_lwg,
      // Inputs and Outputs
      {{{k_packed, v_packed}, vkapi::kWrite}, {{k, v}, vkapi::kRead}},
      // Shader param buffers
      {graph.sizes_ubo(q),
       graph.sizes_ubo(v),
       graph.get_or_create_int_param_buffer(input_pos_symint)},
      // Push Constants
      {},
      // Specialization Constants
      {max_context},
      // Resize Args
      {q, input_pos_symint},
      // Resizing Logic
      nullptr));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_fused_shader,
      pick_fused_gwg,
      pick_required_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{q, k_packed, v_packed}, vkapi::kRead}},
      // Shader param buffers
      {graph.sizes_ubo(q),
       graph.sizes_ubo(k),
       graph.get_or_create_int_param_buffer(input_pos_symint)},
      // Push Constants
      {},
      // Specialization Constants: inv_scale, row stride of q / out, context
      // capacity of the packed copies.
      {1.0f / std::sqrt(static_cast<float>(head_dim)),
       num_q_heads * head_dim,
       max_context},
      // Resize Args
      {q, input_pos_symint},
      // Resizing Logic
      resize_fused_node));
}

struct RegistrarB580Fused {
  RegistrarB580Fused() {
    register_sdpa_fused_b580(sdpa_fused_add_b580, sdpa_fused_serves_b580);
  }
} registrar_b580_fused;

} // namespace

} // namespace sarc
} // namespace vkcompute
