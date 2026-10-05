/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// 780m (openspec/changes/sarc-1.5-780m-prefill-refine): the node of the fused
// prefill SDPA kernels, glsl/sarc_dev/sarc_dev_780m_sdpa_fused{,2}.
//   ET_VK_SARC_780M_SDPA_FUSED=<variant>[,<variant>]
//   e.g. fused2_d64_t128x32g18s32,fused2_d128_t64x32g14s32
// one variant (the shader name after sarc_dev_780m_sdpa_) per head_dim. The
// fused3 family reads V from a transposed scratch copy that a second node
// (sarc_dev_780m_sdpa_vt) writes first; its packed variants (tile token
// ending in k, or km1) read tile-packed copies of both K and V
// (sarc_dev_780m_sdpa_kvt). For the calls it fits (below) the three SDPA
// kernels dispatch nothing and this node writes the output; every other call
// (decode, unaligned prompt, a head_dim without a variant) is untouched.
// Compiled only when the release zone has the fused-SDPA hook (that change's
// hooks/sdpa-fused-hook.patch, not applied on this branch).

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

#ifdef SARC_HAS_SDPA_FUSED_HOOK

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/GraphInfo.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>

namespace vkcompute {
namespace sarc {
namespace {

struct Variant {
  std::string name;
  uint32_t head_dim, m, n, wg_size;
  bool needs_vt;
  bool packed;
};

const std::vector<Variant>& variants() {
  static const std::vector<Variant> parsed = [] {
    std::vector<Variant> out;
    const char* e = std::getenv("ET_VK_SARC_780M_SDPA_FUSED");
    std::string list = e != nullptr ? e : "";
    while (!list.empty()) {
      const size_t comma = list.find(',');
      const std::string token = list.substr(0, comma);
      list = comma == std::string::npos ? "" : list.substr(comma + 1);
      unsigned d, m, n, gx, gy, sg;
      const size_t dims = token.find("_d");
      VK_CHECK_COND(
          dims != std::string::npos &&
              std::sscanf(
                  token.c_str() + dims,
                  "_d%u_t%ux%ug%1u%1us%u",
                  &d, &m, &n, &gx, &gy, &sg) == 6,
          "ET_VK_SARC_780M_SDPA_FUSED: expected <family>_d<D>_t<M>x<N>g<X><Y>s<S>");
      out.push_back(
          {"sarc_dev_780m_sdpa_" + token,
           d,
           m,
           n,
           gx * gy * sg,
           token.rfind("fused3", 0) == 0,
           token.rfind("fused3", 0) == 0 &&
               token.find("s32rk") != std::string::npos});
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

} // namespace

// Registered by the 780m block of Overrides.cpp.
bool sdpa_fused_active_780m(
    ComputeGraph* graph,
    const int32_t q,
    const int32_t input_pos_symint) {
  const Variant* v = variant_for(graph, q);
  if (v == nullptr || !is_valid(input_pos_symint)) {
    return false;
  }
  const uint32_t S = graph->size_at<uint32_t>(-3, q);
  const uint32_t input_pos =
      static_cast<uint32_t>(graph->read_symint(input_pos_symint));
  return S >= v->m && S % v->m == 0 && S % v->n == 0 && input_pos % v->n == 0;
}

namespace {

vkapi::ShaderInfo pick_fused_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)args;
  return VK_KERNEL_FROM_STR(
      variant_for(graph, resize_args.at(0))->name + "_buffer_buffer_half");
}

GlobalWorkGrid pick_fused_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)args;
  const ValueRef q = resize_args.at(0);
  const Variant* v = variant_for(graph, q);
  if (!sdpa_fused_active_780m(graph, q, resize_args.at(1))) {
    return GlobalWorkGrid(
        {0u, 0u, 0u}, kTiledWorkGrid, LocalWorkGroup(v->wg_size, 1u, 1u));
  }
  return GlobalWorkGrid(
      {v->wg_size,
       graph->size_at<uint32_t>(-3, q) / v->m,
       graph->size_at<uint32_t>(-2, q)},
      kTiledWorkGrid,
      LocalWorkGroup(v->wg_size, 1u, 1u));
}

GlobalWorkGrid pick_vt_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  const ValueRef q = resize_args.at(0);
  const ValueRef input_pos_symint = resize_args.at(1);
  const LocalWorkGroup lwg(8u, 8u, 1u);
  if (!sdpa_fused_active_780m(graph, q, input_pos_symint)) {
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

} // namespace

void sdpa_fused_add_780m(
    ComputeGraph& graph,
    const int32_t q,
    const int32_t k,
    const int32_t v,
    const int32_t input_pos_symint,
    const int32_t out) {
  if (variant_for(&graph, q) == nullptr || !is_valid(input_pos_symint)) {
    return;
  }
  // sdpa_fused_active_780m() sees only q: the other tensors must share its storage.
  for (const ValueRef t : {k, v, out}) {
    VK_CHECK_COND(
        graph.storage_type_of(t) == utils::kBuffer &&
        graph.dtype_of(t) == vkapi::kHalf);
  }
  const int32_t head_dim = graph.size_at<int32_t>(-1, q);
  const int32_t num_q_heads = graph.size_at<int32_t>(-2, q);
  const int32_t num_kv_heads = graph.size_at<int32_t>(-2, v);
  const int32_t max_context = graph.size_at<int32_t>(-3, v);

  // The transposed V [kv_h][d][c] of the fused3 family; other families read
  // the cache itself.
  ValueRef k_arg = k;
  ValueRef v_arg = v;
  std::unique_ptr<TmpTensor> kt;
  std::unique_ptr<TmpTensor> vt;
  const Variant* variant = variant_for(&graph, q);
  if (variant->needs_vt) {
    VK_CHECK_COND(max_context % 16 == 0);
    const std::vector<int64_t> scratch_sizes = {
        1, num_kv_heads, head_dim, max_context};
    vt = std::make_unique<TmpTensor>(
        &graph, scratch_sizes, vkapi::kHalf, utils::kBuffer, utils::kWidthPacked);
    v_arg = *vt;
    std::vector<ValueRef> written = {v_arg};
    std::vector<ValueRef> read = {v};
    if (variant->packed) {
      kt = std::make_unique<TmpTensor>(
          &graph, scratch_sizes, vkapi::kHalf, utils::kBuffer, utils::kWidthPacked);
      k_arg = *kt;
      written = {k_arg, v_arg};
      read = {k, v};
    }
    graph.execute_nodes().emplace_back(new DynamicDispatchNode(
        graph,
        VK_KERNEL_FROM_STR(
            variant->packed ? "sarc_dev_780m_sdpa_kvt_buffer_half"
                            : "sarc_dev_780m_sdpa_vt_buffer_half"),
        pick_vt_gwg,
        pick_required_lwg,
        // Inputs and Outputs
        {{written, vkapi::kWrite}, {read, vkapi::kRead}},
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
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_fused_shader,
      pick_fused_gwg,
      pick_required_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{q, k_arg, v_arg}, vkapi::kRead}},
      // Shader param buffers
      {graph.sizes_ubo(q),
       graph.sizes_ubo(k),
       graph.get_or_create_int_param_buffer(input_pos_symint)},
      // Push Constants
      {},
      // Specialization Constants: {inv_scale, row strides of q / out, of the
      // caches and of the transposed V}; the first two families declare two.
      {1.0f / std::sqrt(static_cast<float>(head_dim)),
       num_q_heads * head_dim,
       num_kv_heads * head_dim,
       max_context},
      // Resize Args
      {q, input_pos_symint},
      // Resizing Logic
      resize_fused_node));
}

} // namespace sarc
} // namespace vkcompute

#endif // SARC_HAS_SDPA_FUSED_HOOK
