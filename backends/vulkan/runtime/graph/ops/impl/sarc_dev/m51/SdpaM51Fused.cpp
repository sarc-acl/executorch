/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// M51 (openspec/changes/sarc-1.5-m51-prefill-refine): the node of the fused
// prefill SDPA kernels glsl/sarc_dev/sarc_dev_m51_sdpa_fused3 (the 780M's
// fused3 with subgroup control barriers), the 780M's 780m/Sdpa780mFused.cpp
// restricted to the packed form:
//   ET_VK_SARC_M51_SDPA_FUSED=<variant>[,<variant>]
//   e.g. fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko
// one variant (the shader name after sarc_dev_m51_sdpa_) per head_dim; without
// the variable, the variants of ET_VK_SARC_M51_PROFILE (Overrides.cpp, m51
// block). A copy pass (the 780M's sarc_dev_780m_sdpa_kvt, unchanged) first
// writes tile-packed copies of K and V. For the calls the kernel fits the
// three SDPA kernels dispatch nothing and this node writes the output; every
// other call (decode, unaligned prompt, a head_dim without a variant) is
// untouched. Entry point: Override::sdpa_fused_{serves,add}, installed by the
// m51 block of Overrides.cpp.

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/SDPA.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/GraphInfo.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace vkcompute {
namespace sarc {

// Overrides.cpp, m51 block.
const char* sdpa_fused_variants_m51();
void register_sdpa_fused_m51(
    void (*add)(ComputeGraph&, const std::vector<ValueRef>&),
    bool (*serves)(ComputeGraph*, const std::vector<ValueRef>&));

namespace {

struct Variant {
  std::string name;
  uint32_t head_dim, m, n, wg_size;
};

const std::vector<Variant>& variants() {
  static const std::vector<Variant> parsed = [] {
    std::vector<Variant> out;
    const char* e = std::getenv("ET_VK_SARC_M51_SDPA_FUSED");
    std::string list = e != nullptr ? e : sdpa_fused_variants_m51();
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
          "ET_VK_SARC_M51_SDPA_FUSED: expected fused3_d<D>_t<M>x<N>g<X><Y>s<S>rko");
      out.push_back({"sarc_dev_m51_sdpa_" + token, d, m, n, gx * gy * sg});
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

bool sdpa_fused_active_m51(
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

GlobalWorkGrid pick_fused_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)args;
  const ValueRef q = resize_args.at(0);
  const Variant* v = variant_for(graph, q);
  if (!sdpa_fused_active_m51(graph, q, resize_args.at(1))) {
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
  // Workgroup shape of the copy pass (x along context, y along head_dim, in
  // 4 x 4 blocks), as on the 780M.
  const LocalWorkGroup lwg(8u, 8u, 1u);
  if (!sdpa_fused_active_m51(graph, q, input_pos_symint)) {
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
bool sdpa_fused_serves_m51(
    ComputeGraph* graph,
    const std::vector<ValueRef>& resize_args) {
  return static_cast<SDPAMode>(resize_args.at(3)) == SDPAMode::LLM &&
      sdpa_fused_active_m51(graph, resize_args.at(0), resize_args.at(2));
}

void sdpa_fused_add_m51(
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
  // sdpa_fused_active_m51() sees only q: the other tensors must share its storage.
  for (const ValueRef t : {k, v, out}) {
    VK_CHECK_COND(
        graph.storage_type_of(t) == utils::kBuffer &&
        graph.dtype_of(t) == vkapi::kHalf);
  }
  const int32_t head_dim = graph.size_at<int32_t>(-1, q);
  const int32_t num_q_heads = graph.size_at<int32_t>(-2, q);
  const int32_t num_kv_heads = graph.size_at<int32_t>(-2, v);
  const int32_t max_context = graph.size_at<int32_t>(-3, v);

  // Tile-packed copies of K and V (sarc_dev_780m_sdpa_kvt), with room for the
  // whole cache.
  VK_CHECK_COND(max_context % 16 == 0);
  const std::vector<int64_t> scratch_sizes = {
      1, num_kv_heads, head_dim, max_context};
  TmpTensor kt(
      &graph, scratch_sizes, vkapi::kHalf, utils::kBuffer, utils::kWidthPacked);
  TmpTensor vt(
      &graph, scratch_sizes, vkapi::kHalf, utils::kBuffer, utils::kWidthPacked);
  const ValueRef k_arg = kt;
  const ValueRef v_arg = vt;
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR("sarc_dev_780m_sdpa_kvt_buffer_half"),
      pick_vt_gwg,
      pick_required_lwg,
      // Inputs and Outputs
      {{{k_arg, v_arg}, vkapi::kWrite}, {{k, v}, vkapi::kRead}},
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
      {{out, vkapi::kWrite}, {{q, k_arg, v_arg}, vkapi::kRead}},
      // Shader param buffers
      {graph.sizes_ubo(q),
       graph.sizes_ubo(k),
       graph.get_or_create_int_param_buffer(input_pos_symint)},
      // Push Constants
      {},
      // Specialization Constants: {inv_scale, row strides of q / out, of the
      // caches, and the context capacity of the copies}.
      {1.0f / std::sqrt(static_cast<float>(head_dim)),
       num_q_heads * head_dim,
       num_kv_heads * head_dim,
       max_context},
      // Resize Args
      {q, input_pos_symint},
      // Resizing Logic
      resize_fused_node));
}

struct RegistrarM51Fused {
  RegistrarM51Fused() {
    register_sdpa_fused_m51(sdpa_fused_add_m51, sdpa_fused_serves_m51);
  }
} registrar_m51_fused;

} // namespace

} // namespace sarc
} // namespace vkcompute
