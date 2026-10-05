/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/GraphInfo.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {
namespace sarc {

namespace {

// LLM-mode SDPA dims (mirrors compute_sdpa_dims in SDPA.cpp, file-local
// there). Q: [1, S, H, D]; context_len = S + input_pos.
struct Dims {
  int64_t S;
  int64_t D;
  int64_t context_len;
};

Dims llm_dims(ComputeGraph& graph, ValueRef q, ValueRef input_pos_symint) {
  Dims d;
  d.D = graph.size_at<int64_t>(-1, q);
  d.S = graph.size_at<int64_t>(-3, q);
  const int32_t input_pos =
      is_valid(input_pos_symint) ? graph.read_symint(input_pos_symint) : 0;
  d.context_len = d.S + input_pos;
  return d;
}

bool buf_half(ComputeGraph& graph, ValueRef t) {
  return graph.storage_type_of(t) == utils::kBuffer &&
      graph.dtype_of(t) == vkapi::kHalf;
}

// The SDPA rows are matched with the generic fit check: all-buffer storage
// (kBufBuf), fp16, not single-token, tile-aligned M/N/K (group_size = K so
// the group constraint reduces to K % tile_k).
ShapeInfo sdpa_shape(Op op, int64_t M, int64_t N, int64_t K, bool gemv) {
  ShapeInfo s;
  s.op = op;
  s.M = M;
  s.N = N;
  s.K = K;
  s.group_size = K;
  s.gemv = gemv;
  s.half = true;
  s.input = Storage::kBuffer;
  s.output = Storage::kBuffer;
  s.weight = Storage::kBuffer;
  return s;
}

std::optional<Choice> choose(ComputeGraph& graph, const ShapeInfo& s) {
  return select(device_info(&graph), s);
}

std::string kernel_name(
    ComputeGraph& graph,
    const Choice& c,
    ValueRef out,
    ValueRef second) {
  std::string name = c.kernel_base;
  add_storage_type_suffix(name, graph.storage_type_of(out));
  add_storage_type_suffix(name, graph.storage_type_of(second));
  add_dtype_suffix(name, graph.dtype_of(out));
  return name;
}

} // namespace

std::optional<vkapi::ShaderInfo> pick_sdpa_qk(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef attn_weights = args.at(0).refs.at(0);
  const ValueRef q = args.at(1).refs.at(0);
  const ValueRef k_cache = args.at(1).refs.at(1);
  if (!buf_half(*graph, q) || !buf_half(*graph, k_cache) ||
      !buf_half(*graph, attn_weights)) {
    return std::nullopt;
  }
  const Dims d = llm_dims(*graph, q, resize_args.at(2));
  const auto c = choose(
      *graph,
      sdpa_shape(Op::kSdpaQk, d.S, d.context_len, d.D, d.S == 1));
  if (!c.has_value()) {
    return std::nullopt;
  }
  return VK_KERNEL_FROM_STR(kernel_name(*graph, *c, q, k_cache));
}

std::optional<vkapi::ShaderInfo> pick_sdpa_av(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef attn_weights_softmax = args.at(1).refs.at(0);
  const ValueRef v_cache = args.at(1).refs.at(1);
  const ValueRef q = resize_args.at(0);
  if (!buf_half(*graph, out) || !buf_half(*graph, attn_weights_softmax) ||
      !buf_half(*graph, v_cache)) {
    return std::nullopt;
  }
  const Dims d = llm_dims(*graph, q, resize_args.at(2));
  const auto c = choose(
      *graph,
      sdpa_shape(Op::kSdpaAv, d.S, d.D, d.context_len, d.S == 1));
  if (!c.has_value()) {
    return std::nullopt;
  }
  return VK_KERNEL_FROM_STR(kernel_name(*graph, *c, out, v_cache));
}

std::optional<GlobalWorkGrid> sdpa_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ValueRef>& resize_args) {
  if (auto skip = sdpa_fused_skip(graph, resize_args)) {
    return skip;
  }
  const std::optional<TileDims> dims = dims_for_kernel(shader.kernel_name);
  if (!dims.has_value()) {
    return std::nullopt;
  }
  const ValueRef q = resize_args.at(0);
  const Dims d = llm_dims(*graph, q, resize_args.at(2));
  const uint32_t H = graph->size_at<uint32_t>(-2, q);
  // QK^T tiles (S x context_len); attn*V tiles (S x head_dim). One workgroup
  // per tile, heads in z.
  const bool qk = shader.kernel_name.rfind("sarc_sdpa_qk_", 0) == 0;
  const uint32_t n_extent =
      static_cast<uint32_t>(qk ? d.context_len : d.D);
  const uint32_t wg = dims->wg_size();
  return GlobalWorkGrid(
      {utils::div_up(n_extent, dims->n) * wg,
       utils::div_up(static_cast<uint32_t>(d.S), dims->m),
       H},
      kTiledWorkGrid,
      LocalWorkGroup(wg, 1u, 1u));
}

std::optional<GlobalWorkGrid> sdpa_fused_skip(
    ComputeGraph* graph,
    const std::vector<ValueRef>& resize_args) {
  const Override& o = get_override();
  if (o.sdpa_fused_serves == nullptr ||
      !o.sdpa_fused_serves(graph, resize_args)) {
    return std::nullopt;
  }
  return GlobalWorkGrid(
      {0u, 0u, 0u}, kTiledWorkGrid, LocalWorkGroup(64u, 1u, 1u));
}

void add_sdpa_fused(ComputeGraph& graph, const std::vector<ValueRef>& refs) {
  if (get_override().sdpa_fused_add != nullptr) {
    get_override().sdpa_fused_add(graph, refs);
  }
}

vkapi::SpecVarList sdpa_qk_spec_vars(
    ComputeGraph& graph,
    const ValueRef q,
    const ValueRef input_pos_symint,
    const vkapi::SpecVarList& upstream) {
  const DeviceInfo device = device_info(&graph);
  if (!device_has_active_rows(device, Op::kSdpaQk)) {
    return upstream;
  }
  // The QK coopmat kernel: {inv_scale (upstream), num_k_chunks = D / tile_k,
  // aw_row_width = align4(S + input_pos) at construction}.
  const Dims d = llm_dims(graph, q, input_pos_symint);
  const auto c = select_table(
      device, sdpa_shape(Op::kSdpaQk, d.S, d.context_len, d.D, false));
  const uint32_t tile_k = c.has_value() ? c->dims.k : 32u;
  vkapi::SpecVarList spec = upstream;
  spec.append(static_cast<int32_t>(d.D / tile_k));
  spec.append(static_cast<int32_t>(utils::align_up_4(d.context_len)));
  return spec;
}

vkapi::SpecVarList sdpa_av_spec_vars(
    ComputeGraph& graph,
    const ValueRef q,
    const ValueRef v,
    const vkapi::SpecVarList& upstream) {
  const DeviceInfo device = device_info(&graph);
  if (!device_has_active_rows(device, Op::kSdpaAv)) {
    return upstream;
  }
  // Slots 0-1 belong to release 1.5's GQA decode shader ({1.0, group_size},
  // or nothing); the SARC AV coopmat kernel declares placeholders there and
  // reads {num_k_chunks, out_row_stride, head_dim} from slot 2 on.
  const int32_t head_dim = graph.size_at<int32_t>(-1, q);
  const int32_t num_q_heads = graph.size_at<int32_t>(-2, q);
  const int32_t max_context = graph.size_at<int32_t>(-3, v);
  const Dims d = llm_dims(graph, q, kDummyValueRef);
  const auto c = select_table(
      device, sdpa_shape(Op::kSdpaAv, d.S, d.D, max_context, false));
  const int32_t tile_k = c.has_value() ? static_cast<int32_t>(c->dims.k) : 32;
  vkapi::SpecVarList spec = upstream;
  if (spec.size() == 0) {
    spec.append(1.0f);
  }
  if (spec.size() == 1) {
    spec.append(int32_t(1));
  }
  spec.append((max_context + tile_k - 1) / tile_k);
  spec.append(num_q_heads * head_dim);
  spec.append(head_dim);
  return spec;
}

std::string sdpa_softmax_shader_name(
    ComputeGraph& graph,
    const std::string& upstream_name) {
  if (!device_has_active_rows(device_info(&graph), Op::kSdpaQk)) {
    return upstream_name;
  }
  const char* variant = get_override().softmax_variant;
  if (variant == nullptr) {
    return "sarc_" + upstream_name;
  }
  return "sarc_" + upstream_name + "_" + variant;
}

} // namespace sarc
} // namespace vkcompute
