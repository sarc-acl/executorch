/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/MatMul.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/RepeatInterleave.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Slice.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Softmax.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <string>

namespace vkcompute {

namespace {

//
// SDPA mode: distinguishes the two dispatch families sharing this file.
//   LLM   — Llama-style KV-cache SDPA. Q layout [B=1, S, H,    D] (DHSB).
//           Separate k_cache/v_cache inputs + input_pos_symint for dynamic
//           context_len. attn_weights are padded to multiples of 4 in the
//           S/context_len dims and carry the input dtype. A coop (GEMV)
//           shader variant is selected for single-token decode.
//   FUSED — General SDPA fused op. Q layout [B, H, S, D] (DSHB). No cache,
//           optional additive attn_mask, optional scale arg. attn_weights
//           are unpadded and always fp32. Tiled shader variant only.
//
enum class SDPAMode { LLM, FUSED };

//
// Common dimension helper: folds the axis-swap for LLM vs fused Q layouts.
// `input_pos_symint` is used only for LLM (context_len = S + input_pos);
// pass kDummyValueRef for FUSED.
//
struct SDPADims {
  int64_t B = 1;
  int64_t H = 0;
  int64_t S = 0;
  int64_t D = 0;
  int64_t context_len = 0; // LLM: S + input_pos_val; FUSED: size_at(-2, k)
  int64_t max_context_len = 0; // LLM: size_at(-3, k); FUSED: size_at(-2, k)
};

SDPADims compute_sdpa_dims(
    ComputeGraph& graph,
    const ValueRef q,
    const ValueRef k,
    const ValueRef input_pos_symint,
    const SDPAMode mode) {
  SDPADims d;
  d.D = graph.size_at<int64_t>(-1, q);
  if (mode == SDPAMode::LLM) {
    // Q: [B=1, S, H, D] (DHSB), K: [B=1, C_max, H_kv, D]
    // `k` may be kDummyValueRef in dispatch pickers that don't need it;
    // max_context_len is only read when k is valid.
    d.B = 1;
    d.H = graph.size_at<int64_t>(-2, q);
    d.S = graph.size_at<int64_t>(-3, q);
    d.max_context_len = is_valid(k) ? graph.size_at<int64_t>(-3, k) : 0;
    const int32_t input_pos_val =
        is_valid(input_pos_symint) ? graph.read_symint(input_pos_symint) : 0;
    d.context_len = d.S + input_pos_val;
  } else {
    // Q: [B, H, S, D] (DSHB), K: [B, H_kv, L, D]
    d.B = graph.size_at<int64_t>(-4, q);
    d.H = graph.size_at<int64_t>(-3, q);
    d.S = graph.size_at<int64_t>(-2, q);
    d.context_len = graph.size_at<int64_t>(-2, k);
    d.max_context_len = d.context_len;
  }
  return d;
}

} // namespace

bool is_single_token(ComputeGraph* graph, const ValueRef& q_projected) {
  return graph->size_at<uint32_t>(-3, q_projected) == 1;
}

//
// Resize functions
//

// Unified attn_weights resize. In LLM mode the shape is padded to multiples of
// 4 in the S/context_len dims (to match the tiled shader's iteration space);
// in fused mode it's the unpadded [B, H, S, L].
// resize_args layout: [q, k, input_pos_symint_or_dummy, mode_as_int]
void resize_sdpa_attn_weights_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef attn_weights = args.at(0).refs.at(0);
  const ValueRef q = resize_args.at(0);
  const ValueRef k = resize_args.at(1);
  const ValueRef input_pos_symint = resize_args.at(2);
  const SDPAMode mode = static_cast<SDPAMode>(resize_args.at(3));

  std::vector<int64_t> out_sizes;
  if (mode == SDPAMode::LLM) {
    const int64_t num_q_heads = graph->size_at<int64_t>(-2, q);
    const int64_t seq_len = graph->size_at<int64_t>(-3, q);
    const int32_t input_pos_val = graph->read_symint(input_pos_symint);
    const int64_t context_len = seq_len + input_pos_val;
    out_sizes = {
        1,
        num_q_heads,
        static_cast<int64_t>(utils::align_up_4(seq_len)),
        static_cast<int64_t>(utils::align_up_4(context_len))};
  } else {
    const int64_t B = graph->size_at<int64_t>(-4, q);
    const int64_t H = graph->size_at<int64_t>(-3, q);
    const int64_t S = graph->size_at<int64_t>(-2, q);
    const int64_t L = graph->size_at<int64_t>(-2, k);
    out_sizes = {B, H, S, L};
  }
  graph->virtual_resize(attn_weights, out_sizes);
}

// Softmax preserves attn_weights shape exactly; identical across modes.
void resize_sdpa_attn_weights_softmax_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef attn_weights_softmax = args.at(0).refs.at(0);
  const ValueRef attn_weights = args.at(1).refs.at(0);

  graph->virtual_resize(attn_weights_softmax, graph->sizes_of(attn_weights));
}

// Out matches Q's shape in both modes. resize_args[0] = q.
void resize_sdpa_out_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef q = resize_args.at(0);

  graph->virtual_resize(out, graph->sizes_of(q));
}

//
// Shader dispatch pick functions
//

utils::uvec3 kv_cache_update_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef projected = args.at(1).refs.at(0);

  const uint32_t head_dim_size = graph->size_at<uint32_t>(-1, projected);
  const uint32_t num_heads = graph->size_at<uint32_t>(-2, projected);
  const uint32_t seq_len = graph->size_at<uint32_t>(-3, projected);

  return {utils::div_up_4(head_dim_size), seq_len, num_heads};
}

// resize_args layout for SDPA dispatch pickers mirrors the node creation
// helper: [q, k, input_pos_symint_or_dummy, mode_as_int].
static inline SDPAMode mode_of(const std::vector<ValueRef>& resize_args) {
  return static_cast<SDPAMode>(resize_args.at(3));
}

//
// Cooperative-matrix (WMMA) SDPA prefill path.
//
// The QK^T and attn*V coopmat shaders use a 64x64x32 WG tile with 4 subgroups
// of 64 lanes (256 invocations), matching coopmat_mm.glsl. Selected only for
// LLM prefill (S > 1) on a coopmat-capable discrete RDNA GPU with buffer/fp16
// tensors and tile-aligned shapes; decode (S == 1) stays on the _coop GEMV
// path and ineligible shapes fall back to _tiled. Enabled by default on
// capability-eligible devices; ET_VK_DISABLE_COOPMAT remains the kill switch
// (shared with the q4gsw linear coopmat path).
//
// SDPA coopmat tile-sweep variants (mirrors QuantizedLinear.cpp's
// ET_VK_Q4GSW_COOPMAT_VARIANT / ET_VK_DQ8CA_COOPMAT_VARIANT mechanism).
// sdpa_compute_attn_weights_coopmat (QK^T) and sdpa_compute_out_coopmat
// (attn*V) are independent shaders with independent tile geometries, so each
// gets its own env var. Token format
// "t<WG_TILE_M>x<WG_TILE_N>k<WG_TILE_K>g<SG_GRID_X><SG_GRID_Y>s<SUBGROUP_SIZE>"
// -- no dbuf-namespace prefix needed since SDPA has no loop-structure
// variants (unlike q4gsw/dq8ca's dbuf1-4), so this drops the "tsweep_"
// prefix QuantizedLinear.cpp's kTsweepPrefixes uses for that
// disambiguation.
struct SdpaCoopmatTileDims {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t sgx;
  uint32_t sgy;
  uint32_t sub;
};
inline uint32_t sdpa_wg_size(const SdpaCoopmatTileDims& d) {
  return d.sgx * d.sgy * d.sub;
}

// Current shipped tiles (sdpa_compute_attn_weights_coopmat.yaml /
// sdpa_compute_out_coopmat.yaml parameter_names_with_default_values),
// expressed as tokens so the default path and an explicit env-var override
// run through the identical parse+dispatch code (QuantizedLinear.cpp's
// q4gsw crash -- see 3ceeefc269 -- was exactly a bare/no-token default
// skipping this). QK^T's 128-tall M-tile is the 128x64 tile-sweep optimum
// that still fits the masked shader's shared memory (a naive reuse of the
// generic-matmul sweep's 128x128 winner needs ~50KB LDS once the
// causal-mask Csh scratch is added -- overflows M51); attn*V keeps 64x64
// (no Csh, smaller budget). Both use the same WG_SIZE (256 = 2x2 subgroups
// x 64), so only the M-tile differs between them.
constexpr SdpaCoopmatTileDims kSdpaAttnDefaultDims = {128, 64, 32, 2, 2, 64};
constexpr SdpaCoopmatTileDims kSdpaOutDefaultDims = {64, 64, 32, 2, 2, 64};

static bool is_recognized_sdpa_coopmat_variant_token(const std::string& v) {
  if (v.size() < 2 || v[0] != 't' || !std::isdigit((unsigned char)v[1])) {
    return false;
  }
  return v.find('x') != std::string::npos && v.find('k') != std::string::npos &&
      v.find('g') != std::string::npos && v.find('s') != std::string::npos;
}

// Parses "t<M>x<N>k<K>g<SGX><SGY>s<sub>" -> dims. Returns fallback unchanged
// for an unrecognized token (mirrors QuantizedLinear.cpp's
// parse_tsweep_tile).
static SdpaCoopmatTileDims parse_sdpa_tsweep_tile(
    const std::string& variant,
    const SdpaCoopmatTileDims& fallback) {
  if (!is_recognized_sdpa_coopmat_variant_token(variant)) {
    return fallback;
  }
  const size_t t_pos = 1; // token always starts with 't'
  const size_t x_pos = variant.find('x', t_pos);
  const size_t k_pos = variant.find('k', x_pos);
  const size_t g_pos = variant.find('g', k_pos);
  const size_t s_pos = variant.find('s', g_pos);
  const uint32_t m = std::stoul(variant.substr(t_pos, x_pos - t_pos));
  const uint32_t n = std::stoul(variant.substr(x_pos + 1, k_pos - x_pos - 1));
  const uint32_t k = std::stoul(variant.substr(k_pos + 1, g_pos - k_pos - 1));
  const std::string grid = variant.substr(g_pos + 1, s_pos - g_pos - 1);
  const uint32_t sgx = grid[0] - '0';
  const uint32_t sgy = grid[1] - '0';
  const uint32_t sub = std::stoul(variant.substr(s_pos + 1));
  return {m, n, k, sgx, sgy, sub};
}

static const std::string& sdpa_attn_weights_coopmat_variant() {
  static const std::string variant = [] {
    const char* env = std::getenv("ET_VK_SDPA_ATTN_COOPMAT_VARIANT");
    if (!env) {
      return std::string("t128x64k32g22s64");
    }
    const std::string v(env);
    if (is_recognized_sdpa_coopmat_variant_token(v)) {
      return v;
    }
    return std::string("t128x64k32g22s64");
  }();
  return variant;
}

static const std::string& sdpa_out_coopmat_variant() {
  static const std::string variant = [] {
    const char* env = std::getenv("ET_VK_SDPA_OUT_COOPMAT_VARIANT");
    if (!env) {
      return std::string("t64x64k32g22s64");
    }
    const std::string v(env);
    if (is_recognized_sdpa_coopmat_variant_token(v)) {
      return v;
    }
    return std::string("t64x64k32g22s64");
  }();
  return variant;
}

static SdpaCoopmatTileDims sdpa_attn_tile_dims() {
  return parse_sdpa_tsweep_tile(
      sdpa_attn_weights_coopmat_variant(), kSdpaAttnDefaultDims);
}
static SdpaCoopmatTileDims sdpa_out_tile_dims() {
  return parse_sdpa_tsweep_tile(
      sdpa_out_coopmat_variant(), kSdpaOutDefaultDims);
}

static bool sdpa_coopmat_not_disabled() {
  return std::getenv("ET_VK_DISABLE_COOPMAT") == nullptr;
}

static bool sdpa_coopmat_device_ok(ComputeGraph* graph) {
  if (!sdpa_coopmat_not_disabled()) {
    return false;
  }
  const auto* adapter = graph->context()->adapter_ptr();
  // NOTE: intentionally NO !is_integrated_gpu() check. The target M5 EVT1
  // (Xclipse 970) is a unified-memory "integrated" GPU but has fast fp16 WMMA;
  // the q4gsw linear coopmat gate likewise omits this check and runs there. The
  // generic matmul gate keeps it (to avoid coopmat on iGPUs without WMMA); SDPA
  // coopmat is enabled by default on capability-eligible devices, so the
  // subgroup/cooperative-matrix checks below are the only gate.
  return adapter->supports_cooperative_matrix() &&
      adapter->subgroup_size() == 64;
}

static bool sdpa_buf_half(ComputeGraph* graph, const ValueRef t) {
  return graph->storage_type_of(t) == utils::kBuffer &&
      graph->dtype_of(t) == vkapi::kHalf;
}

static inline bool sdpa_cm_aligned(
    int64_t m,
    int64_t n,
    int64_t k,
    const SdpaCoopmatTileDims& dims) {
  return m % static_cast<int64_t>(dims.m) == 0 &&
      n % static_cast<int64_t>(dims.n) == 0 &&
      k % static_cast<int64_t>(dims.k) == 0;
}

static inline bool is_sdpa_coopmat(const vkapi::ShaderInfo& shader) {
  return shader.kernel_name.find("_coopmat") != std::string::npos;
}

vkapi::ShaderInfo pick_sdpa_qk_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const SDPAMode mode = mode_of(resize_args);
  if (mode == SDPAMode::LLM) {
    const ValueRef attn_weights = args.at(0).refs.at(0);
    const ValueRef q_projected = args.at(1).refs.at(0);
    const ValueRef k_cache = args.at(1).refs.at(1);
    const bool is_gemv = is_single_token(graph, q_projected);

    // Prefill WMMA path: Q @ K^T with K = head_dim, N = context_len, M = S.
    if (!is_gemv && sdpa_coopmat_device_ok(graph) &&
        sdpa_buf_half(graph, q_projected) && sdpa_buf_half(graph, k_cache) &&
        sdpa_buf_half(graph, attn_weights)) {
      const SDPADims d = compute_sdpa_dims(
          *graph, q_projected, k_cache, resize_args.at(2), SDPAMode::LLM);
      const SdpaCoopmatTileDims attn_dims = sdpa_attn_tile_dims();
      if (d.S % static_cast<int64_t>(attn_dims.m) == 0 &&
          d.context_len % static_cast<int64_t>(attn_dims.n) == 0 &&
          d.D % static_cast<int64_t>(attn_dims.k) == 0) {
        std::string shader_name = "sdpa_compute_attn_weights_coopmat_" +
            sdpa_attn_weights_coopmat_variant();
        add_storage_type_suffix(
            shader_name, graph->storage_type_of(q_projected));
        add_storage_type_suffix(shader_name, graph->storage_type_of(k_cache));
        add_dtype_suffix(shader_name, graph->dtype_of(q_projected));
        return VK_KERNEL_FROM_STR(shader_name);
      }
    }

    std::string shader_name = "sdpa_compute_attn_weights";
    shader_name += is_gemv ? "_coop" : "_tiled";
    add_storage_type_suffix(shader_name, graph->storage_type_of(q_projected));
    add_storage_type_suffix(shader_name, graph->storage_type_of(k_cache));
    add_dtype_suffix(shader_name, graph->dtype_of(q_projected));
    return VK_KERNEL_FROM_STR(shader_name);
  } else {
    const ValueRef q = args.at(1).refs.at(0);
    const ValueRef k = args.at(1).refs.at(1);
    // Fused path uses bias variant iff attn_mask was provided (signalled via
    // 3 inputs in the read group: q, k, attn_mask).
    const bool has_bias = args.at(1).refs.size() >= 3;
    std::string shader_name =
        has_bias ? "fused_sdpa_qk_tiled_bias" : "fused_sdpa_qk_tiled";
    add_storage_type_suffix(shader_name, graph->storage_type_of(q));
    add_storage_type_suffix(shader_name, graph->storage_type_of(k));
    add_dtype_suffix(shader_name, graph->dtype_of(q));
    return VK_KERNEL_FROM_STR(shader_name);
  }
}

utils::uvec3 pick_sdpa_qk_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)args;
  const SDPAMode mode = mode_of(resize_args);
  const ValueRef q = resize_args.at(0);
  const ValueRef k = resize_args.at(1);
  const ValueRef input_pos_symint = resize_args.at(2);
  const SDPADims d = compute_sdpa_dims(*graph, q, k, input_pos_symint, mode);

  if (is_sdpa_coopmat(shader)) {
    // One workgroup per output tile (N = context_len, M = S); *wg_size
    // cancels the framework div_up against local x. z carries the head
    // index. Tile dims come from the active ET_VK_SDPA_ATTN_COOPMAT_VARIANT
    // (default: unchanged from the old kSdpaCmQkTileM/kSdpaCmTileN
    // constants).
    const SdpaCoopmatTileDims dims = sdpa_attn_tile_dims();
    const uint32_t num_tiles_n =
        utils::div_up(static_cast<uint32_t>(d.context_len), dims.n);
    const uint32_t num_tiles_m =
        utils::div_up(static_cast<uint32_t>(d.S), dims.m);
    return {
        num_tiles_n * sdpa_wg_size(dims),
        num_tiles_m,
        static_cast<uint32_t>(d.H * d.B)};
  }

  // Dispatch grid: (context_len tiles, S tiles, H * B).
  const uint32_t N4 = utils::div_up_4(static_cast<uint32_t>(d.context_len));
  const uint32_t M4 = utils::div_up_4(static_cast<uint32_t>(d.S));
  return {N4, M4, static_cast<uint32_t>(d.H * d.B)};
}

utils::uvec3 pick_sdpa_qk_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const SDPAMode mode = mode_of(resize_args);
  if (mode == SDPAMode::LLM) {
    // _coopmat must be checked before _coop (the former contains the latter as
    // a substring); the coopmat shaders use a flat workgroup sized by the
    // active ET_VK_SDPA_ATTN_COOPMAT_VARIANT (default: 256 = 2x2 subgroups x
    // 64, unchanged from the old kSdpaCmInvocations constant).
    if (is_sdpa_coopmat(shader)) {
      return {sdpa_wg_size(sdpa_attn_tile_dims()), 1, 1};
    }
    const bool use_coop_algorithm =
        shader.kernel_name.find("_coop") != std::string::npos;
    if (use_coop_algorithm) {
      return {1, 64, 1};
    }
    return pick_hw_square_wg_size(
        graph, shader, global_workgroup_size, args, resize_args);
  }
  return default_pick_local_wg_size(
      graph, shader, global_workgroup_size, args, resize_args);
}

utils::uvec3 pick_sdpa_softmax_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  const SDPAMode mode = mode_of(resize_args);
  const ValueRef q = resize_args.at(0);
  // LLM reads H from axis -2, fused from axis -3 (handled by
  // compute_sdpa_dims).
  const int64_t num_q_heads = (mode == SDPAMode::LLM)
      ? graph->size_at<int64_t>(-2, q)
      : graph->size_at<int64_t>(-3, q);
  const int64_t seq_len = (mode == SDPAMode::LLM)
      ? graph->size_at<int64_t>(-3, q)
      : graph->size_at<int64_t>(-2, q);
  const int64_t B =
      (mode == SDPAMode::LLM) ? 1 : graph->size_at<int64_t>(-4, q);
  return {
      1,
      static_cast<uint32_t>(seq_len),
      static_cast<uint32_t>(num_q_heads * B)};
}

utils::uvec3 pick_sdpa_softmax_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)global_workgroup_size;
  (void)args;
  (void)resize_args;
  return {64, 1, 1};
}

vkapi::ShaderInfo pick_sdpa_av_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const SDPAMode mode = mode_of(resize_args);
  if (mode == SDPAMode::LLM) {
    const ValueRef out = args.at(0).refs.at(0);
    const ValueRef attn_weights_softmax = args.at(1).refs.at(0);
    const ValueRef v_cache = args.at(1).refs.at(1);
    const ValueRef q_projected = resize_args.at(0);
    const bool is_gemv = is_single_token(graph, q_projected);

    // Prefill WMMA path: P @ V with K = context_len, N = head_dim, M = S.
    if (!is_gemv && sdpa_coopmat_device_ok(graph) &&
        sdpa_buf_half(graph, out) &&
        sdpa_buf_half(graph, attn_weights_softmax) &&
        sdpa_buf_half(graph, v_cache)) {
      const SDPADims d = compute_sdpa_dims(
          *graph,
          q_projected,
          resize_args.at(1),
          resize_args.at(2),
          SDPAMode::LLM);
      const SdpaCoopmatTileDims out_dims = sdpa_out_tile_dims();
      if (sdpa_cm_aligned(
              /*m=*/d.S, /*n=*/d.D, /*k=*/d.context_len, out_dims)) {
        std::string shader_name =
            "sdpa_compute_out_coopmat_" + sdpa_out_coopmat_variant();
        add_storage_type_suffix(shader_name, graph->storage_type_of(out));
        add_storage_type_suffix(shader_name, graph->storage_type_of(v_cache));
        add_dtype_suffix(shader_name, graph->dtype_of(out));
        return VK_KERNEL_FROM_STR(shader_name);
      }
    }

    std::string shader_name = "sdpa_compute_out";
    shader_name += is_gemv ? "_coop" : "_tiled";
    add_storage_type_suffix(shader_name, graph->storage_type_of(out));
    add_storage_type_suffix(shader_name, graph->storage_type_of(v_cache));
    add_dtype_suffix(shader_name, graph->dtype_of(out));
    return VK_KERNEL_FROM_STR(shader_name);
  } else {
    const ValueRef out = args.at(0).refs.at(0);
    const ValueRef v = args.at(1).refs.at(1);
    std::string shader_name = "fused_sdpa_av_tiled";
    add_storage_type_suffix(shader_name, graph->storage_type_of(out));
    add_storage_type_suffix(shader_name, graph->storage_type_of(v));
    add_dtype_suffix(shader_name, graph->dtype_of(out));
    return VK_KERNEL_FROM_STR(shader_name);
  }
}

utils::uvec3 pick_sdpa_av_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const SDPAMode mode = mode_of(resize_args);
  const ValueRef q = resize_args.at(0);
  const ValueRef k = resize_args.at(1);
  const ValueRef input_pos_symint = resize_args.at(2);
  const SDPADims d = compute_sdpa_dims(*graph, q, k, input_pos_symint, mode);

  if (is_sdpa_coopmat(shader)) {
    // One workgroup per output tile (N = head_dim, M = S). z = head. Tile
    // dims come from the active ET_VK_SDPA_OUT_COOPMAT_VARIANT (default:
    // unchanged from the old kSdpaCmTileM/kSdpaCmTileN constants).
    const SdpaCoopmatTileDims dims = sdpa_out_tile_dims();
    const uint32_t num_tiles_n =
        utils::div_up(static_cast<uint32_t>(d.D), dims.n);
    const uint32_t num_tiles_m =
        utils::div_up(static_cast<uint32_t>(d.S), dims.m);
    return {
        num_tiles_n * sdpa_wg_size(dims),
        num_tiles_m,
        static_cast<uint32_t>(d.H * d.B)};
  }

  const uint32_t N4 = utils::div_up_4(static_cast<uint32_t>(d.D));
  const uint32_t M4 = utils::div_up_4(static_cast<uint32_t>(d.S));
  return {N4, M4, static_cast<uint32_t>(d.H * d.B)};
}

utils::uvec3 pick_sdpa_av_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const SDPAMode mode = mode_of(resize_args);
  if (mode == SDPAMode::LLM) {
    // _coopmat must be checked before _coop (the former contains the latter as
    // a substring); the coopmat shaders use a flat workgroup sized by the
    // active ET_VK_SDPA_OUT_COOPMAT_VARIANT (default: 256 = 2x2 subgroups x
    // 64, unchanged from the old kSdpaCmInvocations constant).
    if (is_sdpa_coopmat(shader)) {
      return {sdpa_wg_size(sdpa_out_tile_dims()), 1, 1};
    }
    const bool use_coop_algorithm =
        shader.kernel_name.find("_coop") != std::string::npos;
    if (use_coop_algorithm) {
      return {1, 64, 1};
    }
    return pick_hw_square_wg_size(
        graph, shader, global_workgroup_size, args, resize_args);
  }
  return default_pick_local_wg_size(
      graph, shader, global_workgroup_size, args, resize_args);
}

//
// Dispatch nodes
//

void add_sdpa_kv_cache_update_node(
    ComputeGraph& graph,
    const ValueRef input_pos_symint,
    const ValueRef projected,
    const ValueRef cache) {
  std::string kernel_name("sdpa_kv_cache_update");
  add_storage_type_suffix(kernel_name, graph.storage_type_of(cache));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(projected));
  add_dtype_suffix(kernel_name, graph.dtype_of(projected));

  vkapi::ParamsBindList param_ubos = {
      graph.sizes_ubo(cache),
      graph.sizes_ubo(projected),
      graph.get_or_create_int_param_buffer(input_pos_symint)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      kv_cache_update_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{cache, vkapi::kWrite}, {projected, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {input_pos_symint},
      // Resizing Logic
      nullptr));
}

// Unified QK node (attn_weights = scale * Q @ K^T [+ bias]).
// LLM: pass input_pos_symint (real symint), attn_mask = kDummyValueRef.
// FUSED: pass input_pos_symint = kDummyValueRef, attn_mask = valid ref or
//        kDummyValueRef to indicate no bias. scale_val is always passed as
//        a spec const; the LLM path computes it per head_dim and FUSED may
//        inherit from the caller-supplied scale.
void add_sdpa_compute_attn_weights_node(
    ComputeGraph& graph,
    const ValueRef q,
    const ValueRef k,
    const ValueRef input_pos_symint,
    const ValueRef attn_mask,
    const float scale_val,
    const ValueRef attn_weights,
    const SDPAMode mode) {
  vkapi::ParamsBindList param_ubos = {
      graph.sizes_ubo(q),
      graph.sizes_ubo(k),
  };
  std::vector<ValueRef> read_inputs = {q, k};

  if (mode == SDPAMode::LLM) {
    param_ubos.append(graph.get_or_create_int_param_buffer(input_pos_symint));
  } else if (is_valid(attn_mask)) {
    param_ubos.append(graph.sizes_ubo(attn_mask));
    read_inputs.push_back(attn_mask);
  }

  const ValueRef mode_ref = static_cast<ValueRef>(mode);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_sdpa_qk_shader,
      pick_sdpa_qk_global_wg_size,
      pick_sdpa_qk_local_wg_size,
      // Inputs and Outputs
      {{attn_weights, vkapi::kWrite}, {read_inputs, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants: {inv_scale (id 3), num_k_chunks (id 4)}.
      // num_k_chunks = head_dim / WG_TILE_K is static and consumed only by the
      // coopmat QK^T variant (WG_TILE_K from the active
      // ET_VK_SDPA_ATTN_COOPMAT_VARIANT); the tiled/coop variants declare
      // only id 3 and ignore the trailing entry -- safe to compute
      // unconditionally even when coopmat doesn't end up firing.
      {scale_val,
       graph.size_at<int32_t>(-1, q) /
           static_cast<int32_t>(sdpa_attn_tile_dims().k)},
      // Resize Args: [q, k, input_pos_symint_or_dummy, mode]
      {q, k, input_pos_symint, mode_ref},
      // Resizing Logic
      resize_sdpa_attn_weights_node));
}

void add_sdpa_attn_weights_softmax_node(
    ComputeGraph& graph,
    const ValueRef attn_weights,
    const ValueRef q,
    const ValueRef k,
    const ValueRef input_pos_symint,
    const ValueRef attn_weights_softmax,
    const SDPAMode mode) {
  std::string shader_name;
  if (mode == SDPAMode::LLM) {
    shader_name = "sdpa_attn_weights_softmax";
    add_storage_type_suffix(
        shader_name, graph.storage_type_of(attn_weights_softmax));
    add_dtype_suffix(shader_name, graph.dtype_of(attn_weights_softmax));
  } else {
    shader_name = "fused_sdpa_softmax";
    add_storage_type_suffix(
        shader_name, graph.storage_type_of(attn_weights_softmax));
    add_dtype_suffix(shader_name, graph.dtype_of(attn_weights_softmax));
  }

  vkapi::ParamsBindList param_ubos;
  if (mode == SDPAMode::LLM) {
    param_ubos = {
        graph.sizes_ubo(q),
        graph.sizes_ubo(k),
        graph.get_or_create_int_param_buffer(input_pos_symint)};
  } else {
    param_ubos = {graph.sizes_ubo(q), graph.sizes_ubo(k)};
  }

  const ValueRef mode_ref = static_cast<ValueRef>(mode);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(shader_name),
      pick_sdpa_softmax_global_wg_size,
      pick_sdpa_softmax_local_wg_size,
      // Inputs and Outputs
      {{attn_weights_softmax, vkapi::kWrite}, {attn_weights, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args: [q, k, input_pos_symint_or_dummy, mode]
      {q, k, input_pos_symint, mode_ref},
      // Resizing Logic
      resize_sdpa_attn_weights_softmax_node));
}

void add_sdpa_compute_out_node(
    ComputeGraph& graph,
    const ValueRef attn_weights_softmax,
    const ValueRef v,
    const ValueRef q,
    const ValueRef k,
    const ValueRef input_pos_symint,
    const ValueRef out,
    const SDPAMode mode) {
  vkapi::ParamsBindList param_ubos;
  if (mode == SDPAMode::LLM) {
    param_ubos = {
        graph.sizes_ubo(q),
        graph.sizes_ubo(v),
        graph.get_or_create_int_param_buffer(input_pos_symint)};
  } else {
    param_ubos = {graph.sizes_ubo(q), graph.sizes_ubo(v)};
  }

  const ValueRef mode_ref = static_cast<ValueRef>(mode);

  // Coopmat attn*V spec constants (static; consumed only by the coopmat
  // variant — the tiled/coop variants ignore the trailing entries, and id 3 is
  // the inv_scale slot the decode _coop shader reads, kept at 1.0 = no-op).
  // num_k_chunks uses max_context_len (the loop bound is a spec const per the
  // Xclipse bug) and the active ET_VK_SDPA_OUT_COOPMAT_VARIANT's WG_TILE_K
  // (independent of QK^T's own K-tile -- these are two separately-swept
  // shaders); beyond-context chunks are zero-staged in the shader. Values
  // are meaningful only in LLM mode; in FUSED they are ignored.
  const int32_t cm_head_dim = graph.size_at<int32_t>(-1, q);
  const int32_t cm_num_q_heads = graph.size_at<int32_t>(-2, q);
  const int32_t cm_max_context = graph.size_at<int32_t>(-3, v);
  const int32_t cm_out_tile_k = static_cast<int32_t>(sdpa_out_tile_dims().k);
  const int32_t cm_num_k_chunks =
      (cm_max_context + cm_out_tile_k - 1) / cm_out_tile_k;
  const int32_t cm_out_row_stride = cm_num_q_heads * cm_head_dim;

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_sdpa_av_shader,
      pick_sdpa_av_global_wg_size,
      pick_sdpa_av_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{attn_weights_softmax, v}, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants:
      // {inv_scale slot (id 3), num_k_chunks (id 4), out_row_stride (id 5),
      //  head_dim (id 6)}.
      {1.0f, cm_num_k_chunks, cm_out_row_stride, cm_head_dim},
      // Resize Args: [q, k, input_pos_symint_or_dummy, mode]
      {q, k, input_pos_symint, mode_ref},
      // Resizing Logic
      resize_sdpa_out_node));
}

//
// High level operator impl
//

void update_cache_impl(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef value = args[arg_idx++];
  const ValueRef cache = args[arg_idx++];
  const ValueRef input_pos_symint = args[arg_idx++];
  const ValueRef out = args[arg_idx++];

  // Unused variables
  (void)out;

  VK_CHECK_COND(graph.size_at<int32_t>(-4, value) == 1);
  VK_CHECK_COND(graph.size_at<int32_t>(-4, cache) == 1);
  VK_CHECK_COND(
      graph.size_at<int32_t>(-1, value) == graph.size_at<int32_t>(-1, cache));
  VK_CHECK_COND(
      graph.size_at<int32_t>(-2, value) == graph.size_at<int32_t>(-2, cache));

  add_sdpa_kv_cache_update_node(graph, input_pos_symint, value, cache);
}

void sdpa_impl(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef q_projected = args[arg_idx++];
  const ValueRef k_cache = args[arg_idx++];
  const ValueRef v_cache = args[arg_idx++];
  const ValueRef input_pos_symint = args[arg_idx++];
  const ValueRef attn_mask = args[arg_idx++];
  const ValueRef dropout_p = args[arg_idx++];
  const ValueRef is_causal = args[arg_idx++];
  const ValueRef scale = args[arg_idx++];

  // Output tensors
  const ValueRef out = args[arg_idx++];

  // Batches must be 1
  VK_CHECK_COND(graph.size_at<int32_t>(-4, q_projected) == 1);
  VK_CHECK_COND(graph.size_at<int32_t>(-4, k_cache) == 1);
  VK_CHECK_COND(graph.size_at<int32_t>(-4, v_cache) == 1);
  // k and v projected must have the same shape
  VK_CHECK_COND(graph.sizes_of(k_cache) == graph.sizes_of(v_cache));
  // head dim must match between tensors
  VK_CHECK_COND(
      graph.size_at<int32_t>(-1, q_projected) ==
      graph.size_at<int32_t>(-1, k_cache));
  // All tensors must have the packed dim be the width (head) dimension
  VK_CHECK_COND(graph.packed_dim_of(q_projected) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(k_cache) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(v_cache) == WHCN::kWidthDim);
  // Some variables are not supported yet
  VK_CHECK_COND(
      graph.val_is_none(dropout_p) ||
      graph.extract_scalar<double>(dropout_p) == 0);
  VK_CHECK_COND(graph.val_is_none(scale));
  // is_causal is assumed to be true in the current implementation.
  VK_CHECK_COND(
      graph.val_is_none(is_causal) || graph.extract_scalar<bool>(is_causal));
  VK_CHECK_COND(graph.val_is_none(attn_mask));

  const int64_t num_q_heads = graph.size_at<int64_t>(-2, q_projected);
  int64_t max_seq_len = graph.size_at<int64_t>(-3, q_projected);
  const int64_t max_context_len = graph.size_at<int32_t>(-3, k_cache);

  const utils::StorageType attn_weights_storage =
      graph.storage_type_of(q_projected);

  // If using buffer storage for attn weights, we need to ensure that the buffer
  // numel limit is not exceeded. If needed, manually adjust max_seq_len based
  // on the buffer numel limit.
  if (attn_weights_storage == utils::kBuffer) {
    const int64_t max_buffer_numel = graph.max_buffer_numel();
    if (num_q_heads * max_seq_len * max_context_len >= max_buffer_numel) {
      // Compute the maximum possible value for max_seq_len that will hit
      // the buffer numel limit.
      max_seq_len = max_buffer_numel / (num_q_heads * max_context_len);
      // Adjust down to the nearest multiple of 4 to make sure the limit is
      // not hit.
      if (max_seq_len % 4 != 0) {
        max_seq_len = (max_seq_len / 4) * 4;
      } else {
        max_seq_len -= 4;
      }
    }
  }

  std::vector<int64_t> attn_weight_full_sizes = {
      1, // batch
      num_q_heads,
      max_seq_len,
      max_context_len};

  TmpTensor attn_weights(
      &graph,
      attn_weight_full_sizes,
      graph.dtype_of(q_projected),
      attn_weights_storage,
      utils::kWidthPacked);

  TmpTensor attn_weights_softmax(
      &graph,
      attn_weight_full_sizes,
      graph.dtype_of(q_projected),
      attn_weights_storage,
      utils::kWidthPacked);

  const int32_t head_dim_size = graph.size_at<int32_t>(-1, q_projected);
  const float scale_val = 1.0f / std::sqrt(static_cast<float>(head_dim_size));

  add_sdpa_compute_attn_weights_node(
      graph,
      q_projected,
      k_cache,
      input_pos_symint,
      /*attn_mask=*/kDummyValueRef,
      scale_val,
      attn_weights,
      SDPAMode::LLM);

  add_sdpa_attn_weights_softmax_node(
      graph,
      attn_weights,
      q_projected,
      k_cache,
      input_pos_symint,
      attn_weights_softmax,
      SDPAMode::LLM);

  add_sdpa_compute_out_node(
      graph,
      attn_weights_softmax,
      v_cache,
      q_projected,
      /*k=*/kDummyValueRef,
      input_pos_symint,
      out,
      SDPAMode::LLM);
}

void sdpa_with_kv_cache_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef q_projected = args[arg_idx++];
  const ValueRef k_projected = args[arg_idx++];
  const ValueRef v_projected = args[arg_idx++];
  const ValueRef k_cache_data = args[arg_idx++];
  const ValueRef v_cache_data = args[arg_idx++];
  const ValueRef input_pos_symint = args[arg_idx++];
  const ValueRef sequence_len = args[arg_idx++];
  const ValueRef attn_mask = args[arg_idx++];
  const ValueRef dropout_p = args[arg_idx++];
  const ValueRef is_causal = args[arg_idx++];
  const ValueRef scale = args[arg_idx++];

  // Output tensors
  const ValueRef out = args[arg_idx];

  (void)sequence_len;

  utils::StorageType cache_storage = graph.storage_type_of(q_projected);
  const ValueRef k_cache =
      graph.add_tensor_like(k_cache_data, cache_storage, utils::kWidthPacked);
  const ValueRef v_cache =
      graph.add_tensor_like(v_cache_data, cache_storage, utils::kWidthPacked);

  update_cache_impl(graph, {k_projected, k_cache, input_pos_symint, -1});
  update_cache_impl(graph, {v_projected, v_cache, input_pos_symint, -1});

  sdpa_impl(
      graph,
      {q_projected,
       k_cache,
       v_cache,
       input_pos_symint,
       attn_mask,
       dropout_p,
       is_causal,
       scale,
       out});
}

void compute_attn_weight_with_kv_cache_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef q_projected = args[arg_idx++];
  const ValueRef k_projected = args[arg_idx++];
  const ValueRef v_projected = args[arg_idx++];
  const ValueRef k_cache_data = args[arg_idx++];
  const ValueRef v_cache_data = args[arg_idx++];
  const ValueRef input_pos_symint = args[arg_idx++];
  const ValueRef sequence_len = args[arg_idx++];
  const ValueRef attn_mask = args[arg_idx++];
  (void)attn_mask;
  const ValueRef dropout_p = args[arg_idx++];
  (void)dropout_p;
  const ValueRef is_causal = args[arg_idx++];
  (void)is_causal;
  const ValueRef scale = args[arg_idx++];
  (void)scale;

  // Output tensors
  const ValueRef out = args[arg_idx++];

  (void)sequence_len;

  const utils::StorageType cache_storage = graph.storage_type_of(q_projected);
  const ValueRef k_cache =
      graph.add_tensor_like(k_cache_data, cache_storage, utils::kWidthPacked);
  const ValueRef v_cache =
      graph.add_tensor_like(v_cache_data, cache_storage, utils::kWidthPacked);

  update_cache_impl(graph, {k_projected, k_cache, input_pos_symint, -1});
  update_cache_impl(graph, {v_projected, v_cache, input_pos_symint, -1});

  const int32_t head_dim_size = graph.size_at<int32_t>(-1, q_projected);
  const float scale_val = 1.0f / std::sqrt(static_cast<float>(head_dim_size));

  add_sdpa_compute_attn_weights_node(
      graph,
      q_projected,
      k_cache,
      input_pos_symint,
      /*attn_mask=*/kDummyValueRef,
      scale_val,
      out,
      SDPAMode::LLM);
}

//
// Fused SDPA entry point (et_vk.sdpa.default).
//
// Accepts pre-reshaped [B, H, S, D] tensors (DSHB) plus optional additive
// attn_mask and optional scale scalar. No KV cache; this is the general SDPA
// fused op used by non-LLM models.
//
void fused_sdpa_impl(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef q = args[arg_idx++];
  const ValueRef k = args[arg_idx++];
  const ValueRef v = args[arg_idx++];
  const ValueRef attn_mask = args[arg_idx++];
  const ValueRef scale_ref = args[arg_idx++];
  const ValueRef out = args[arg_idx];

  // Validate inputs
  VK_CHECK_COND(graph.dim_of(q) == 4);
  VK_CHECK_COND(graph.dim_of(k) == 4);
  VK_CHECK_COND(graph.dim_of(v) == 4);
  // Head dim must match between Q and K
  VK_CHECK_COND(graph.size_at<int32_t>(-1, q) == graph.size_at<int32_t>(-1, k));
  // K and V must have same sequence length
  VK_CHECK_COND(graph.size_at<int32_t>(-2, k) == graph.size_at<int32_t>(-2, v));
  // All tensors must be width-packed
  VK_CHECK_COND(graph.packed_dim_of(q) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(k) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(v) == WHCN::kWidthDim);

  // Compute scale
  const int32_t head_dim = graph.size_at<int32_t>(-1, q);
  float scale_val;
  if (graph.val_is_none(scale_ref)) {
    scale_val = 1.0f / std::sqrt(static_cast<float>(head_dim));
  } else {
    scale_val = graph.extract_scalar<float>(scale_ref);
  }

  // Resolve attn_mask: a None value is normalized to kDummyValueRef so the
  // unified helpers can branch with a single `is_valid()` check.
  const ValueRef attn_mask_ref =
      graph.val_is_none(attn_mask) ? kDummyValueRef : attn_mask;

  // Get dimensions for intermediate allocation
  const int64_t B = graph.size_at<int64_t>(-4, q);
  const int64_t H = graph.size_at<int64_t>(-3, q);
  const int64_t S = graph.size_at<int64_t>(-2, q);
  const int64_t L = graph.size_at<int64_t>(-2, k);

  std::vector<int64_t> attn_weight_sizes = {B, H, S, L};

  // attn_weights and attn_weights_softmax follow the output's storage so the
  // entire fused SDPA pipeline uses a uniform storage type. attn_weights stays
  // in fp32 for numerical stability of the Q@K^T accumulation.
  const utils::StorageType attn_storage = graph.storage_type_of(out);

  TmpTensor attn_weights(
      &graph,
      attn_weight_sizes,
      vkapi::ScalarType::Float,
      attn_storage,
      utils::kWidthPacked);

  TmpTensor attn_weights_softmax(
      &graph,
      attn_weight_sizes,
      graph.dtype_of(q),
      attn_storage,
      utils::kWidthPacked);

  // Phase 1: Q @ K^T with fp32 accumulation, apply scale and optional bias
  add_sdpa_compute_attn_weights_node(
      graph,
      q,
      k,
      /*input_pos_symint=*/kDummyValueRef,
      attn_mask_ref,
      scale_val,
      attn_weights,
      SDPAMode::FUSED);

  // Phase 2: Softmax in fp32, output in input dtype
  add_sdpa_attn_weights_softmax_node(
      graph,
      attn_weights,
      q,
      k,
      /*input_pos_symint=*/kDummyValueRef,
      attn_weights_softmax,
      SDPAMode::FUSED);

  // Phase 3: attn_weights_softmax @ V
  add_sdpa_compute_out_node(
      graph,
      attn_weights_softmax,
      v,
      q,
      k,
      /*input_pos_symint=*/kDummyValueRef,
      out,
      SDPAMode::FUSED);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(sdpa_with_kv_cache.default, sdpa_with_kv_cache_impl);
  VK_REGISTER_OP(update_cache.default, update_cache_impl);
  VK_REGISTER_OP(llama.custom_sdpa.default, sdpa_impl);
  VK_REGISTER_OP(
      testing.compute_attn_weight_with_kv_cache.default,
      compute_attn_weight_with_kv_cache_impl);
  VK_REGISTER_OP(et_vk.sdpa.default, fused_sdpa_impl);
}

} // namespace vkcompute
