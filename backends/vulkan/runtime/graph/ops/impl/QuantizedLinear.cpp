/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <cstring>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/GemmCoopmat.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizeDequantize.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

//
// Shader dispatch utilities
//

void resize_linear_qw_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;

  ValueRef output = args.at(0).refs.at(0);
  ValueRef fp_input = args.at(1).refs.at(0);
  ValueRef weight_data = extra_args.at(1);

  std::vector<int64_t> mat1_sizes = graph->sizes_of(fp_input);
  std::vector<int64_t> mat2_sizes = graph->sizes_of(weight_data);

  const int64_t out_cols = utils::val_at(-2, mat1_sizes);
  const int64_t out_rows = utils::val_at(-2, mat2_sizes);

  std::vector<int64_t> new_out_sizes(3);
  if (mat1_sizes.size() == 2) {
    new_out_sizes.resize(2);
    new_out_sizes.at(0) = out_cols;
    new_out_sizes.at(1) = out_rows;
  } else {
    new_out_sizes.at(0) = mat1_sizes.at(0);
    new_out_sizes.at(1) = out_cols;
    new_out_sizes.at(2) = out_rows;
  }

  graph->virtual_resize(output, new_out_sizes);
}

// Per-shader coopmat tile geometry (must match each shader's yaml).
// Workgroup size (wg_size) = SG_GRID_X * SG_GRID_Y * SUBGROUP_SIZE.
//   linear_q4gsw_coopmat       128x128x16, 2x2 subgroups x 32 (forced) -> 128
//   linear_dq8ca_q4gsw_coopmat 64x32x32,   1x2 subgroups x 64          -> 128
// (specs/027-e2e-tile-sweep: dq8ca_q4gsw tile updated from 128x64x32/2x2 to
// this e2e-ranked winner. int8-MMA stays on wave64 at this tile -- specs/026
// found subgroup=32 legal-but-shape-dependently-incorrect, not verified
// correct at this specific tile. specs/036-portable-device-sweep: q4gsw tile
// updated 2026-07-24 from 128x64x16/2x2 to this e2e-ranked winner, +6.8/+7.7/
// +10.1% on 1B/3B/8B prefill tok/s -- N doubled, grid/subgroup unchanged so
// wg_size is unaffected.)
struct CoopmatTileDims {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  // Threads per workgroup = SG_GRID_X * SG_GRID_Y * SUBGROUP_SIZE. MUST match
  // the WG_SIZE the shader yaml resolves to, or the launched thread count won't
  // match the shader's staging passes (out-of-bounds).
  uint32_t wg_size;
  // Only needed for the texture-IO shared-memory budget below; the buffer path
  // never reads it. 0 = "unknown / shipped default".
  uint32_t sg_grid_y;
};
// linear_qw_coopmat.yaml: 128x128, 2x2 subgroup grid, sg32 -> WG_SIZE 128.
constexpr CoopmatTileDims kQ4gswCoopmatDims = {128, 128, 16, 128, 2};
// linear_dq8ca_qw_coopmat.yaml: 64x32, 1x2 grid, sg64 -> WG_SIZE 128
// (specs/027-e2e-tile-sweep winner, was 128x64x32/256).
constexpr CoopmatTileDims kDq8caQ4gswCoopmatDims = {64, 32, 32, 128, 2};

// specs/028-4w-e2e-tile-sweep / specs/041-dbuf4-tile-sweep:
// ET_VK_Q4GSW_COOPMAT_VARIANT / ET_VK_DQ8CA_COOPMAT_VARIANT can swap the
// coopmat dispatch to a specific tile/subgroup-grid/loop-structure variant for
// sweeping. A "tsweep_dbuf<N>_t..." token additionally selects a loop-structure
// variant (1-4); "tsweep_t..." is the original (production winner's own)
// namespace. Unset/unrecognized = shipped dispatch, unchanged. The five
// prefixes are mutually exclusive by construction (position 7 is 'd' vs 't').
static const char* const kTsweepPrefixes[] = {
    "tsweep_dbuf1_t",
    "tsweep_dbuf2_t",
    "tsweep_dbuf3_t",
    "tsweep_dbuf4_t",
    "tsweep_t",
};

static bool is_recognized_coopmat_variant_token(const std::string& v) {
  for (const char* prefix : kTsweepPrefixes) {
    if (v.rfind(prefix, 0) == 0) {
      return true;
    }
  }
  return false;
}

static const std::string& q4gsw_coopmat_variant() {
  static const std::string variant = [] {
    const char* env = std::getenv("ET_VK_Q4GSW_COOPMAT_VARIANT");
    if (!env) {
      return std::string();
    }
    const std::string v(env);
    if (is_recognized_coopmat_variant_token(v)) {
      return v;
    }
    return std::string();
  }();
  return variant;
}

static const std::string& dq8ca_coopmat_variant() {
  // Default (no ET_VK_DQ8CA_COOPMAT_VARIANT set):
  // tsweep_dbuf4_t128x16k64g12s64, this workspace's texture-IO tile sweep
  // winner on M51 (2026-08-18) -- 11.8-17.3% faster than the prior
  // t64x32k32g12s64 default across 1B/3B/8B, e2e-confirmed (not microbench) and
  // ETDump-confirmed to actually dispatch coopmat on the real prefill path.
  // Only takes effect when ET_VK_TEXTURE_COOPMAT=1 is also set -- that master
  // switch stays opt-in.
  static const std::string variant = [] {
    const char* env = std::getenv("ET_VK_DQ8CA_COOPMAT_VARIANT");
    if (!env) {
      return std::string("tsweep_dbuf4_t128x16k64g12s64");
    }
    const std::string v(env);
    if (is_recognized_coopmat_variant_token(v)) {
      return v;
    }
    return std::string("tsweep_dbuf4_t128x16k64g12s64");
  }();
  return variant;
}

// Parses "tsweep_t<M>x<N>k<K>g<SGX><SGY>s<sub>" or
// "tsweep_dbuf<N>_t<M>x<N>k<K>g<SGX><SGY>s<sub>" -> {M, N, K, SGX*SGY*sub,
// SGY}. Returns fallback unchanged if the token matches none of
// kTsweepPrefixes.
static CoopmatTileDims parse_tsweep_tile(
    const std::string& variant,
    const CoopmatTileDims& fallback) {
  size_t t_pos = std::string::npos;
  for (const char* prefix : kTsweepPrefixes) {
    if (variant.rfind(prefix, 0) == 0) {
      t_pos = std::strlen(prefix);
      break;
    }
  }
  if (t_pos == std::string::npos) {
    return fallback;
  }
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
  return {m, n, k, sgx * sgy * sub, sgy};
}

static CoopmatTileDims parse_q4gsw_tsweep_tile(const std::string& variant) {
  return parse_tsweep_tile(variant, kQ4gswCoopmatDims);
}

static CoopmatTileDims coopmat_tile_dims(const std::string& kernel_name) {
  // Exact prefix matches (the "linear_dq8ca_*" names must not match the
  // weight-only entries). Order matters: check dq8ca first.
  if (kernel_name.rfind("linear_dq8ca_q4gsw_coopmat", 0) == 0) {
    return parse_tsweep_tile(dq8ca_coopmat_variant(), kDq8caQ4gswCoopmatDims);
  }
  if (kernel_name.rfind("linear_q4gsw_coopmat", 0) == 0) {
    return parse_q4gsw_tsweep_tile(q4gsw_coopmat_variant());
  }
  return {kCoopmatTileM, kCoopmatTileN, kCoopmatTileK, kCoopmatInvocations};
}

utils::uvec3 quantized_linear_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);

  std::vector<int64_t> out_sizes = graph->sizes_of(out);
  // width
  const uint32_t N = utils::val_at(-1, out_sizes);
  // height
  const uint32_t M = utils::val_at(-2, out_sizes);

  // Coopmat variants dispatch a 256-thread WG per 64x64 output tile.  Mirrors
  // GemmCoopmat.cpp's pick_linear_coopmat_global_wg_size — the multiplication
  // by kCoopmatInvocations cancels the framework's div_up, since
  // local_wg = {256, 1, 1}.
  if (shader.kernel_name.find("_coopmat") != std::string::npos) {
    const CoopmatTileDims dims = coopmat_tile_dims(shader.kernel_name);
    const uint32_t num_tiles_n = utils::div_up(N, dims.n);
    const uint32_t num_tiles_m = utils::div_up(M, dims.m);
    return {num_tiles_n * dims.wg_size, num_tiles_m, 1};
  }

  uint32_t N_per_tile = 4;
  uint32_t M_per_tile = 4;

  // For 4-bit weights, each output tile contains 8 columns
  if (shader.kernel_name.find("q4") != std::string::npos) {
    N_per_tile = 8;
  }
  if (shader.kernel_name.find("coop") != std::string::npos) {
    M_per_tile = 1;
  }

  if (shader.kernel_name.find("q8ta_q8csw_tiled") != std::string::npos) {
    N_per_tile = 8;
  }

  const uint32_t num_N_tiles = utils::div_up(N, N_per_tile);
  const uint32_t num_M_tiles = utils::div_up(M, M_per_tile);

  // Otherwise, each output tile contains 4 columns and 4 rows
  return {num_N_tiles, num_M_tiles, 1};
}

utils::uvec3 quantized_linear_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  // Coopmat variants use a per-shader workgroup size (q4gsw/q8csw = 128,
  // dq8ca = 256) — must match the WG_SIZE the shader yaml resolves to.
  if (shader.kernel_name.find("_coopmat") != std::string::npos) {
    return {coopmat_tile_dims(shader.kernel_name).wg_size, 1, 1};
  }

  const bool use_coop_algorithm =
      shader.kernel_name.find("_coop") != std::string::npos;

  if (use_coop_algorithm) {
    return {1, 1, 64};
  } else {
    return pick_hw_square_wg_size(
        graph, shader, global_workgroup_size, args, resize_args);
  }
}

// Experiment hook (specs/040/041): allows the *_texture3d_* variants, which
// stage the result tile through shared memory and imageStore it instead of
// coopMatStore-ing straight to an SSBO. Off by default, so buffer dispatch is
// byte-identical.
static bool texture_coopmat_enabled() {
  static const bool enabled = std::getenv("ET_VK_TEXTURE_COOPMAT") != nullptr;
  return enabled;
}

// Returns true when the q4gsw coopmat shader can be dispatched for this
// (M, N, K, dtype, output_storage, group_size) tuple. Preconditions match what
// linear_q4gsw_coopmat.glsl assumes; the subgroup_size == 64 check scopes this
// to wave64 devices (e.g. AMD RDNA), which the coopmat tiling is tuned for.
static bool can_use_q4gsw_coopmat(
    ComputeGraph* graph,
    const ValueRef output,
    const ValueRef fp_input,
    int64_t group_size,
    const ValueRef bias,
    int64_t tile_m = kCoopmatTileM,
    int64_t tile_n = kCoopmatTileN,
    int64_t tile_k = kCoopmatTileK,
    bool allow_texture_io = false,
    uint32_t sg_grid_y = 0) {
  // Baseline-measurement escape hatch: forces every dispatch through this
  // function to the tiled fallback, regardless of eligibility. Off by
  // default (unset), so production behavior is unchanged.
  if (std::getenv("ET_VK_FORCE_TILED_LINEAR") != nullptr) {
    return false;
  }
  // The coopmat shaders only build HAS_BIAS=false variants, so they would
  // silently drop a bias. Fall back to the tiled path (which applies bias at
  // runtime via the apply_bias spec constant) whenever a bias is present.
  if (!graph->val_is_none(bias)) {
    return false;
  }
  const auto* adapter = graph->context()->adapter_ptr();
  if (!adapter->supports_cooperative_matrix()) {
    return false;
  }
  if (adapter->subgroup_size() != 64) {
    return false;
  }
  // These coopmat shaders have only been validated on AMD-RDNA GPUs (Samsung
  // Xclipse and AMD Radeon). Gate to those families so the path stays off on
  // other devices that advertise cooperative matrix support but have not been
  // validated.
  if (!graph->device_is_amd()) {
    return false;
  }
  // Coopmat shaders dispatch over gl_WorkGroupID.xy only, sized purely from
  // the output's trailing two dims; neither that sizing nor the shaders
  // themselves ever read a leading dim. A genuine batch (any leading-dim
  // product != 1) would silently miscompute all slices beyond the first --
  // but a size-1 leading dim (the real exported model's rank-3 [1, M, K]
  // activations, never squeezed) is safe: a contiguous Buffer's [1, M, N]
  // layout is bit-identical to [M, N] when the leading dim is 1, so the
  // existing 2D dispatch grid already covers 100% of the data. Reject only a
  // real batch.
  const std::vector<int64_t> out_sizes = graph->sizes_of(output);
  int64_t leading_dims_numel = 1;
  for (int64_t d = 0; d < graph->dim_of(output) - 2; d++) {
    leading_dims_numel *= utils::val_at(d, out_sizes);
  }
  if (leading_dims_numel != 1) {
    return false;
  }
  if (graph->storage_type_of(output) != utils::kBuffer) {
    // One IO_STORAGE param is shared across t_input/t_output, so BOTH must be
    // texture3d; the imageStore epilogue and texelFetch A-stage assume
    // width-packed texels.
    if (!allow_texture_io || !texture_coopmat_enabled()) {
      return false;
    }
    if (graph->storage_type_of(output) != utils::kTexture3D ||
        graph->storage_type_of(fp_input) != utils::kTexture3D) {
      return false;
    }
    if (graph->packed_dim_of(output) != WHCN::kWidthDim ||
        graph->packed_dim_of(fp_input) != WHCN::kWidthDim) {
      return false;
    }
    // The texture epilogue needs a Csh staging array ON TOP OF the Ash/Bsh the
    // buffer path already allocates: SG_GRID_Y * MMA_M rows x WG_TILE_N fp16.
    // That term is absent from the offline tile_constraints model, so a tile
    // that is legal for buffer can exceed the shared-memory limit at texture
    // IO -- one specific large tile hung the GPU and rebooted the board
    // instead of failing pipeline creation (2026-08-09). Reject here so it
    // falls back to tiled.
    if (sg_grid_y > 0) {
      constexpr int64_t kMmaM = 16; // MMA_M, fixed across every coopmat yaml
      const int64_t csh_bytes =
          int64_t(sg_grid_y) * kMmaM * tile_n * int64_t(sizeof(uint16_t));
      const int64_t limit =
          graph->context()->adapter_ptr()->max_compute_shared_memory_size();
      if (csh_bytes >= limit) {
        return false;
      }
    }
  }
  if (graph->dtype_of(output) != vkapi::kHalf) {
    return false;
  }

  const int64_t N = utils::val_at(-1, out_sizes);
  const int64_t M = utils::val_at(-2, out_sizes);
  const std::vector<int64_t> in_sizes = graph->sizes_of(fp_input);
  const int64_t K = utils::val_at(-1, in_sizes);

  if (M % tile_m != 0) {
    return false;
  }
  if (N % tile_n != 0) {
    return false;
  }
  if (K % tile_k != 0) {
    return false;
  }
  if (group_size % tile_k != 0) {
    return false;
  }
  return true;
}

vkapi::ShaderInfo pick_linear_qw_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;

  const ValueRef output = args.at(0).refs.at(0);
  const ValueRef fp_input = args.at(1).refs.at(0);
  const ValueRef packed_int_weight = args.at(1).refs.at(1);

  const bool weight_is_4bit = resize_args.at(0) != kDummyValueRef;
  const bool is_gemv_case = is_gemv(graph, fp_input);

  // Use the coopmat shader for 4-bit, non-gemv, buffer-output, half-dtype
  // dispatches when shape alignment allows; tiled remains the fallback.
  if (weight_is_4bit && !is_gemv_case) {
    const int64_t group_size =
        graph->extract_scalar<int64_t>(resize_args.at(0));
    // A tsweep_* variant has different tile dims than the shipped
    // kQ4gswCoopmatDims, so the eligibility check's alignment gate must use
    // the ACTIVE variant's own dims, not the shipped constant.
    const CoopmatTileDims active_dims =
        parse_q4gsw_tsweep_tile(q4gsw_coopmat_variant());
    if (can_use_q4gsw_coopmat(
            graph,
            output,
            fp_input,
            group_size,
            resize_args.at(2),
            active_dims.m,
            active_dims.n,
            active_dims.k,
            /*allow_texture_io=*/true,
            active_dims.sg_grid_y)) {
      std::string kernel_name = "linear_q4gsw_coopmat";
      const std::string& variant = q4gsw_coopmat_variant();
      if (!variant.empty()) {
        kernel_name += "_" + variant;
      }
      // Output storage is buffer or texture3d (gated above); weight storage
      // matches the existing variants.
      add_storage_type_suffix(kernel_name, graph->storage_type_of(output));
      add_storage_type_suffix(
          kernel_name, graph->storage_type_of(packed_int_weight));
      add_dtype_suffix(kernel_name, graph->dtype_of(output));
      return VK_KERNEL_FROM_STR(kernel_name);
    }
  }

  std::string kernel_name = "linear_";
  if (weight_is_4bit) {
    kernel_name += "q4gsw";
  } else {
    kernel_name += "q8csw";
  }

  if (weight_is_4bit && is_gemv_case) {
    kernel_name += "_coop";
  } else {
    kernel_name += "_tiled";
  }
  add_storage_type_suffix(kernel_name, graph->storage_type_of(output));
  add_storage_type_suffix(
      kernel_name, graph->storage_type_of(packed_int_weight));
  add_dtype_suffix(kernel_name, graph->dtype_of(output));

  return VK_KERNEL_FROM_STR(kernel_name);
}

vkapi::ShaderInfo pick_linear_dqa_qw_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;

  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef fp_input = args.at(1).refs.at(0);
  const ValueRef int_input = args.at(1).refs.at(1);
  (void)int_input;
  const ValueRef input_zp = args.at(1).refs.at(4);
  const ValueRef int_weight = args.at(1).refs.at(5);

  const bool weight_is_4bit = resize_args.at(0) != kDummyValueRef;
  const bool is_gemv_case = is_gemv(graph, fp_input);

  // Use the coopmat<int8> shader for 4-bit dq8ca dispatches when the device
  // enumerates VK_COMPONENT_TYPE_SINT8_KHR in its cooperative matrix property
  // list and the shape aligns; tiled otherwise.
  if (weight_is_4bit && !is_gemv_case &&
      graph->context()->adapter_ptr()->supports_int8_cooperative_matrix()) {
    const int64_t group_size =
        graph->extract_scalar<int64_t>(resize_args.at(0));
    // Alignment gate must use the ACTIVE sweep variant's own tile dims (same
    // rationale as the q4gsw tsweep hook above).
    const CoopmatTileDims active_dq8ca_dims =
        parse_tsweep_tile(dq8ca_coopmat_variant(), kDq8caQ4gswCoopmatDims);
    // The dq8ca texture-IO shader declares t_input (fp_input) with an
    // IO_STORAGE-typed binding even though it's never read in the shader body
    // (activations arrive pre-quantized in t_packed_int8_input instead) --
    // Vulkan still requires the bound resource's storage type to match the
    // declared binding type, so fp_input must genuinely be texture3d too when
    // texture IO is active. Same requirement as q4gsw; no separate check
    // needed.
    if (can_use_q4gsw_coopmat(
            graph,
            out,
            fp_input,
            group_size,
            resize_args.at(2),
            active_dq8ca_dims.m,
            active_dq8ca_dims.n,
            active_dq8ca_dims.k,
            /*allow_texture_io=*/true,
            active_dq8ca_dims.sg_grid_y)) {
      std::string kernel_name = "linear_dq8ca_q4gsw_coopmat";
      const std::string& dq8ca_variant = dq8ca_coopmat_variant();
      if (!dq8ca_variant.empty()) {
        kernel_name += "_" + dq8ca_variant;
      }
      add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
      add_storage_type_suffix(kernel_name, graph->storage_type_of(int_weight));
      add_dtype_suffix(kernel_name, graph->dtype_of(out));
      return VK_KERNEL_FROM_STR(kernel_name);
    }
  }

  std::string kernel_name = "linear_dq8ca_q4gsw";
  kernel_name += is_gemv_case ? "_coop" : "_tiled";
  add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
  add_storage_type_suffix(kernel_name, graph->storage_type_of(int_weight));
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  add_zp_dtype_mode_suffix(kernel_name, graph->dtype_of(input_zp));

  return VK_KERNEL_FROM_STR(kernel_name);
}

//
// Prepacking nodes
//

ValueRef prepack_quantized_linear_weight(
    ComputeGraph& graph,
    const QuantizationConfig& weight_quant_config,
    const ValueRef qmat2_data) {
  VK_CHECK_COND(
      weight_quant_config.nbits == 8 || weight_quant_config.nbits == 4);

  std::vector<int64_t> qmat2_orig_sizes = graph.sizes_of(qmat2_data);
  const int64_t ndim = graph.dim_of(qmat2_data);

  int64_t qmat2_width = qmat2_orig_sizes.at(ndim - 1);
  int64_t qmat2_height = qmat2_orig_sizes.at(ndim - 2);

  int64_t K;
  int64_t N;
  if (weight_quant_config.nbits == 4) {
    // For 4-bit quantization, weight source data has shape [N, K/2]. Each byte
    // contains 2 * 4-bit values.
    K = qmat2_width * 2;
    N = qmat2_height;
  } else {
    // For 8-bit quantization, the weight source data has shape [N, K]
    K = qmat2_width;
    N = qmat2_height;
  }

  // Sanity check that assumptions are correct. Data loads along the innermost
  // dimension must be well aligned along texel boundaries.
  if (weight_quant_config.nbits == 4) {
    VK_CHECK_COND(K % 8 == 0);
  } else {
    VK_CHECK_COND(K % 4 == 0);
  }

  // The packing format packs the weight tensor into blocks of 4 columns (K) and
  // 4 rows (N)
  int64_t N_per_block = 4;
  int64_t K_per_block = 4;

  // For 4 bit, quantization, the amount of information contained in one block
  // can be doubled. Each block will contain data for 8 rows (N) instead of the
  // usual 4.
  if (weight_quant_config.nbits == 4) {
    N_per_block = 8;
  }

  // To figure out the size of the output tensor, determine the number of blocks
  // along each dimension.
  const int64_t num_blocks_K = utils::div_up(K, K_per_block);
  const int64_t num_blocks_N = utils::div_up(N, N_per_block);

  // The blocks are arranged in a transposed manner, such that the transposed
  // weight block is indexed like packed_weights[k4][n4] - this is to allow for
  // optimal memory coalescing when computing GEMM.
  int64_t output_height = num_blocks_K;
  // The base dtype of the packed tensor is int32 (each int32 contains 4x 8bit
  // values) and each block is represented as a ivec4. Therefore the width dim
  // of the packed tensor is multiplied by 4.
  int64_t output_width = num_blocks_N * 4;

  // For 4 bit quantization, The blocks are arranged without the transposition,
  // such that a weight block is accessed like packed_weights[n8][k4]. This is
  // an optimization targeted for LLMs, which need to compute GEMV as well as
  // GEMM. This memory layout provides better performance for the co-operative
  // algorithm used to compute GEMV, at the cost of slightly reducing GEMM
  // performance.
  if (weight_quant_config.nbits == 4) {
    output_height = num_blocks_N;
    output_width = num_blocks_K * 4;
  }

  // Store the original sizes of the weight data to pass to the shader
  utils::ivec2 orig_sizes = {
      utils::safe_downcast<int32_t>(K), utils::safe_downcast<int32_t>(N)};

  std::vector<int64_t> qmat2_sizes{output_height, output_width};

  utils::StorageType storage_type = utils::kTexture2D;
  uint32_t max_extent = graph.context()->adapter_ptr()->max_texture2d_dim();
  if (output_width > max_extent * 4 || output_height > max_extent) {
    storage_type = utils::kBuffer;
  }

  std::string kernel_name = weight_quant_config.nbits == 4
      ? "pack_q4_linear_weight"
      : "pack_q8_linear_weight";
  add_storage_type_suffix(kernel_name, storage_type);

  // Check prepack cache before creating a new prepack node. This avoids
  // allocating a duplicate output tensor when the same weight data has already
  // been prepacked with the same kernel (e.g. tied embedding/linear weights).
  ValueRef cached = graph.get_cached_prepack(qmat2_data, kernel_name);
  if (is_valid(cached)) {
    return cached;
  }

  ValueRef qmat2 = graph.add_tensor(
      qmat2_sizes, vkcompute::vkapi::kInt, storage_type, utils::kWidthPacked);

  utils::uvec3 global_wg_size;
  if (weight_quant_config.nbits == 4) {
    // For 4-bit quantization, each thread writes out two adjacent blocks
    global_wg_size = {
        utils::safe_downcast<uint32_t>(utils::div_up(num_blocks_K, int64_t(2))),
        utils::safe_downcast<uint32_t>(num_blocks_N),
        1u};
  } else {
    global_wg_size = {
        utils::safe_downcast<uint32_t>(num_blocks_N),
        utils::safe_downcast<uint32_t>(num_blocks_K),
        1u};
  }

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg_size,
      graph.create_local_wg_size(global_wg_size),
      // Inputs and Outputs
      qmat2_data,
      qmat2,
      // UBOs
      {},
      // Specialization Constants
      {},
      // Push Constants
      {graph.sizes_pc_of(qmat2),
       PushConstantDataInfo(&orig_sizes, sizeof(utils::ivec2))}));

  graph.cache_prepack(qmat2_data, kernel_name, qmat2);
  return qmat2;
}

//
// Dispatch nodes
//

/*
 * Shader dispatch for linear with quantized weight but fp activations.
 */
void add_linear_qw_node(
    ComputeGraph& graph,
    const QuantizationConfig& weight_quant_config,
    const ValueRef fp_input,
    const ValueRef weight_data,
    const ValueRef packed_weight,
    const ValueRef packed_weight_scales,
    const ValueRef packed_weight_zeros,
    const ValueRef group_size,
    const ValueRef bias_data,
    const ValueRef packed_bias,
    const ValueRef output) {
  // Only certain quantization types supported at the moment
  VK_CHECK_COND(
      weight_quant_config.granularity == kPerChannel ||
      weight_quant_config.granularity == kPerGroup);
  VK_CHECK_COND(weight_quant_config.is_symmetric);
  VK_CHECK_COND(
      weight_quant_config.nbits == 8 || weight_quant_config.nbits == 4);

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(output), graph.sizes_ubo(fp_input)};

  uint32_t apply_bias = 1;
  if (graph.val_is_none(bias_data)) {
    apply_bias = 0;
  }

  int32_t K4_per_group = 0;
  // 3rd coopmat spec const: num_groups (trip count of the coopmat loop),
  // passed as a spec constant to avoid the Xclipse UBO-derived bounds crash.
  int32_t num_groups = 0;
  if (weight_quant_config.nbits == 4) {
    int32_t group_size_val = graph.extract_scalar<int32_t>(group_size);
    K4_per_group = utils::div_up(group_size_val, int32_t(4));
    num_groups = graph.size_at<int32_t>(-1, fp_input) / group_size_val;
  }

  const ValueRef is_4bit_flag =
      weight_quant_config.nbits == 4 ? group_size : kDummyValueRef;

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_linear_qw_shader,
      quantized_linear_global_wg_size,
      quantized_linear_local_wg_size,
      // Inputs and Outputs
      {{output, vkapi::kWrite},
       {{fp_input, packed_weight, packed_weight_scales, packed_bias},
        vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      {},
      // Specialization Constants
      // 4th spec const: output width N. The coopmat shaders must take N for
      // coopMatStore address math from a spec constant, not the sizes UBO
      // (Xclipse driver miscompiles UBO-derived store offsets/strides).
      {apply_bias,
       K4_per_group,
       num_groups,
       graph.size_at<int32_t>(-1, output)},
      // Resize args (resize_args.at(2) = bias_data, read by the coopmat gate)
      {is_4bit_flag, weight_data, bias_data},
      // Resizing Logic
      resize_linear_qw_node));
}

void add_linear_qa_qw_node(
    ComputeGraph& graph,
    const QuantizationConfig& input_quant_config,
    const QuantizationConfig& weight_quant_config,
    const ValueRef fp_input,
    const ValueRef packed_int_input,
    const ValueRef packed_input_scale,
    const ValueRef packed_input_zp,
    const ValueRef input_scale_data,
    const ValueRef input_zp_data,
    const ValueRef weight_data,
    const ValueRef packed_weight,
    const ValueRef packed_weight_sums,
    const ValueRef packed_weight_scales,
    const ValueRef group_size,
    const ValueRef bias_data,
    const ValueRef packed_bias,
    const ValueRef output) {
  VK_CHECK_COND(input_quant_config.granularity == kPerTensor);
  VK_CHECK_COND(input_quant_config.nbits == 8);
  VK_CHECK_COND(weight_quant_config.granularity == kPerChannel);
  VK_CHECK_COND(weight_quant_config.is_symmetric);
  VK_CHECK_COND(weight_quant_config.nbits == 8);

  float scale = graph.extract_scalar<float>(input_scale_data);
  int32_t zp = graph.extract_scalar<int32_t>(input_zp_data);

  // Get shader for quantized linear
  std::string kernel_name = "linear_q8ta_q8csw_tiled";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(output));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(packed_int_input));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(packed_weight));
  add_dtype_suffix(kernel_name, graph.dtype_of(output));
  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(output), graph.sizes_ubo(packed_int_input)};

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&scale, sizeof(scale)),
      PushConstantDataInfo(&zp, sizeof(zp)),
  };

  uint32_t apply_bias = 1;
  if (graph.val_is_none(bias_data)) {
    apply_bias = 0;
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      quantized_linear_global_wg_size,
      quantized_linear_local_wg_size,
      // Inputs and Outputs
      {{output, vkapi::kWrite},
       {{packed_int_input,
         packed_weight,
         packed_weight_sums,
         packed_weight_scales,
         packed_bias},
        vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      push_constants,
      // Specialization Constants
      {apply_bias},
      // Resize args
      {fp_input},
      // Resizing Logic
      nullptr));
}

void add_linear_dqa_qw_node(
    ComputeGraph& graph,
    const QuantizationConfig& input_quant_config,
    const QuantizationConfig& weight_quant_config,
    const ValueRef fp_input,
    const ValueRef packed_int_input,
    const ValueRef int_input_sums,
    const ValueRef packed_input_scale,
    const ValueRef packed_input_zp,
    const ValueRef input_scale_data,
    const ValueRef input_zp_data,
    const ValueRef weight_data,
    const ValueRef packed_weight,
    const ValueRef packed_weight_sums,
    const ValueRef packed_weight_scales,
    const ValueRef group_size,
    const ValueRef bias_data,
    const ValueRef packed_bias,
    const ValueRef output) {
  VK_CHECK_COND(input_quant_config.granularity == kPerChannel);
  VK_CHECK_COND(input_quant_config.nbits == 8);
  VK_CHECK_COND(input_quant_config.is_dynamic);

  VK_CHECK_COND(weight_quant_config.granularity == kPerGroup);
  VK_CHECK_COND(weight_quant_config.is_symmetric);
  VK_CHECK_COND(weight_quant_config.nbits == 4);

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(output), graph.sizes_ubo(fp_input)};

  uint32_t apply_bias = 1;
  if (graph.val_is_none(bias_data)) {
    apply_bias = 0;
  }

  int32_t K4_per_group = 0;
  int32_t coopmat_k_iters = 0;
  const int32_t K_dim = graph.size_at<int32_t>(-1, fp_input);
  if (weight_quant_config.nbits == 4) {
    int32_t group_size_val = graph.extract_scalar<int32_t>(group_size);
    K4_per_group = utils::div_up(group_size_val, int32_t(4));
    coopmat_k_iters = K_dim / group_size_val;
  }

  const ValueRef is_4bit_flag =
      weight_quant_config.nbits == 4 ? group_size : kDummyValueRef;

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_linear_dqa_qw_shader,
      quantized_linear_global_wg_size,
      quantized_linear_local_wg_size,
      // Inputs and Outputs
      {{output, vkapi::kWrite},
       {{fp_input,
         packed_int_input,
         int_input_sums,
         packed_input_scale,
         packed_input_zp,
         packed_weight,
         packed_weight_sums,
         packed_weight_scales,
         packed_bias},
        vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      {},
      // Specialization Constants
      // 4th spec const: output width N for coopMatStore (see
      // add_linear_qw_node).
      {apply_bias,
       K4_per_group,
       coopmat_k_iters,
       graph.size_at<int32_t>(-1, output)},
      // Resize args (resize_args.at(2) = bias_data, read by the coopmat gate)
      {is_4bit_flag, weight_data, bias_data},
      // Resizing Logic
      resize_linear_qw_node));
}

//
// High level operator impl
//

void quantized_linear_impl(
    ComputeGraph& graph,
    const QuantizationConfig& input_quant_config,
    const QuantizationConfig& weight_quant_config,
    const ValueRef fp_input,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef weight_data,
    const ValueRef weight_sums_data,
    const ValueRef weight_scales_data,
    const ValueRef weight_zeros_data,
    const ValueRef group_size,
    const ValueRef bias_data,
    const ValueRef output) {
  std::vector<int64_t> input_sizes = graph.sizes_of(fp_input);
  std::vector<int64_t> weight_sizes = graph.sizes_of(weight_data);

  const int64_t K = utils::val_at(-1, input_sizes);
  // K (input channels) must be a multiple of 4 to ensure that reading a group
  // of 4 input channels from the input tensor will be aligned on a texel
  // boundary.
  VK_CHECK_COND(K % 4 == 0);

  // Prepack weight data

  const ValueRef packed_weight =
      prepack_quantized_linear_weight(graph, weight_quant_config, weight_data);
  const ValueRef packed_weight_scales = prepack_standard(
      graph, weight_scales_data, utils::kBuffer, utils::kWidthPacked);
  // Weight affine quant not supported at the moment
  const ValueRef packed_weight_zeros = kDummyValueRef;

  // Prepack bias data

  // Create a dummy tensor to fill the binding slot of the bias tensor if it is
  // not provided. This helps simplify dispatch logic and makes it so that
  // fewer shdaer variants need to be generated.
  TmpTensor dummy_bias(
      &graph, {}, graph.dtype_of(output), utils::kBuffer, utils::kWidthPacked);

  ValueRef packed_bias = dummy_bias.vref;
  if (graph.val_is_not_none(bias_data)) {
    packed_bias =
        prepack_standard(graph, bias_data, utils::kBuffer, utils::kWidthPacked);
  }

  // Use weight only quantized linear if at least one is true:
  // 1. Device does not support int8 dot product
  // 2. Input is not quantized
  if (!graph.can_use_int8_dot_product() ||
      input_quant_config.granularity == kNoQuantization) {
    add_linear_qw_node(
        graph,
        weight_quant_config,
        fp_input,
        weight_data,
        packed_weight,
        packed_weight_scales,
        packed_weight_zeros,
        group_size,
        bias_data,
        packed_bias,
        output);

    return;
  }
  // Otherwise, use input and weight quantized linear computed with integer
  // accumulation

  // Input scale/zero point only used for activation & weight quantized linear
  ValueRef packed_input_scale = input_scale;
  ValueRef packed_input_zp = input_zp;
  if (graph.val_is_tref(input_scale)) {
    VK_CHECK_COND(graph.val_is_tref(packed_input_zp));
    packed_input_scale = prepack_standard(
        graph, input_scale, utils::kTexture3D, utils::kWidthPacked);
    packed_input_zp = prepack_standard(
        graph, input_zp, utils::kTexture3D, utils::kWidthPacked);
  }

  // Pre-computed per quant group weight sums are needed for int accumulation,
  // but not for weight only
  const ValueRef packed_weight_sums = prepack_standard(
      graph, weight_sums_data, utils::kBuffer, utils::kWidthPacked);

  // Allocate temporary tensor to store quantized and packed input
  TmpTensor packed_int_input(
      &graph,
      graph.sizes_of(fp_input),
      vkapi::kInt8x4,
      utils::kBuffer,
      utils::kPackedInt8_4H4W);

  // Non dynamically quantized input case
  if (!input_quant_config.is_dynamic) {
    add_quantize_and_pack_4h4w_node(
        graph,
        input_quant_config,
        fp_input,
        packed_input_scale,
        packed_input_zp,
        input_scale,
        input_zp,
        packed_int_input,
        group_size);

    add_linear_qa_qw_node(
        graph,
        input_quant_config,
        weight_quant_config,
        fp_input,
        packed_int_input,
        packed_input_scale,
        packed_input_zp,
        input_scale,
        input_zp,
        weight_data,
        packed_weight,
        packed_weight_sums,
        packed_weight_scales,
        group_size,
        bias_data,
        packed_bias,
        output);

    return;
  }

  // Otherwise, input is dynamically quantized. Currently only per group 4-bit
  // quantized weights is supported for this mode.
  VK_CHECK_COND(weight_quant_config.nbits == 4);

  int64_t num_groups = 1;
  if (weight_quant_config.granularity == kPerGroup) {
    num_groups = graph.size_at<int64_t>(-2, weight_scales_data);
  }

  // Per-group int8 input sums buffer, indexed as ivec4[group_idx * M4 + m4]
  // by both the producer (quantize_and_pack_4h4w_with_group_sums.glsl) and the
  // consumer (linear_int8_input_sums_load.glslh). Capacity must therefore be
  // num_groups * M4 ivec4 texels, sized by the input row count M -- NOT K.
  // dtype is kInt to match the shaders' `int`/ivec4 binding (each texel is 4
  // int32 sums = 16 bytes).
  const int64_t M = utils::val_at(-2, input_sizes);
  const int64_t M4 = utils::div_up(M, int64_t(4));
  TmpTensor int_input_sums(
      &graph,
      {num_groups * M4 * 4},
      vkapi::kInt,
      utils::kBuffer,
      utils::kWidthPacked);

  add_quantize_and_pack_4h4w_with_group_sums_node(
      graph,
      input_quant_config,
      fp_input,
      int_input_sums,
      packed_input_scale,
      packed_input_zp,
      packed_int_input,
      group_size);

  add_linear_dqa_qw_node(
      graph,
      input_quant_config,
      weight_quant_config,
      fp_input,
      packed_int_input,
      int_input_sums,
      packed_input_scale,
      packed_input_zp,
      input_scale,
      input_zp,
      weight_data,
      packed_weight,
      packed_weight_sums,
      packed_weight_scales,
      group_size,
      bias_data,
      packed_bias,
      output);
}

void linear_q8ta_q8csw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_sums_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef output = args.at(idx++);

  const int64_t K = graph.size_at<int64_t>(-1, fp_input);

  QuantizationConfig input_quant_config(8, kPerTensor, {}, false);
  QuantizationConfig weight_quant_config(8, kPerChannel, {K});

  quantized_linear_impl(
      graph,
      input_quant_config,
      weight_quant_config,
      fp_input,
      input_scale,
      input_zp,
      weight_data,
      weight_sums_data,
      weight_scales_data,
      kDummyValueRef, // weight_zeros_data
      kDummyValueRef, // group_size
      bias_data,
      output);
}

void linear_q8csw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef output = args.at(idx++);

  const int64_t K = graph.size_at<int64_t>(-1, fp_input);

  QuantizationConfig input_quant_config(32, kNoQuantization, {});
  QuantizationConfig weight_quant_config(8, kPerChannel, {K});

  quantized_linear_impl(
      graph,
      input_quant_config,
      weight_quant_config,
      fp_input,
      kDummyValueRef, // input scale
      kDummyValueRef, // input zp
      weight_data,
      kDummyValueRef, // weight sums
      weight_scales_data,
      kDummyValueRef, // weight zeros
      kDummyValueRef, // group size
      bias_data,
      output);
}

// Registered below as et_vk.linear_q4gsw.default -- takes over from
// Q4gswLinear.cpp's own registration of the same op name (commented out
// there) so that add_linear_qw_node / linear_q4gsw_coopmat are actually
// reachable. See memory quant-perf-rebase-orphaned-4w-coopmat: upstream's
// Q4gswLinear.cpp silently hijacks this exact op name with its own
// q4gsw_linear_gemm__* tiled-only shaders, and QuantizedLinear.cpp's coopmat
// path (present and correctly built) is unreachable for any 4w PTE until
// this registration is restored.
void linear_q4gsw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef group_size = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef output = args.at(idx++);

  const int64_t group_size_val = graph.extract_scalar<int64_t>(group_size);

  QuantizationConfig input_quant_config(32, kNoQuantization, {});
  QuantizationConfig weight_quant_config(4, kPerGroup, {group_size_val});

  quantized_linear_impl(
      graph,
      input_quant_config,
      weight_quant_config,
      fp_input,
      kDummyValueRef, // input scale
      kDummyValueRef, // input zp
      weight_data,
      kDummyValueRef, // weight sums
      weight_scales_data,
      kDummyValueRef, // weight zeros
      group_size, // group size
      bias_data,
      output);
}

void linear_dq8ca_q4gsw(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_sums_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef group_size = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef output = args.at(idx++);

  const int64_t group_size_val = graph.extract_scalar<int64_t>(group_size);

  QuantizationConfig input_quant_config(8, kPerChannel, {}, false, true);
  QuantizationConfig weight_quant_config(4, kPerGroup, {group_size_val});

  quantized_linear_impl(
      graph,
      input_quant_config,
      weight_quant_config,
      fp_input,
      input_scale,
      input_zp,
      weight_data,
      weight_sums_data,
      weight_scales_data,
      kDummyValueRef, // weight_zeros_data
      group_size, // group_size
      bias_data,
      output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.linear_q8ta_q8csw.default, linear_q8ta_q8csw);
  VK_REGISTER_OP(et_vk.linear_q8csw.default, linear_q8csw);
  VK_REGISTER_OP(et_vk.linear_q4gsw.default, linear_q4gsw);
  VK_REGISTER_OP(et_vk.linear_dq8ca_q4gsw.default, linear_dq8ca_q4gsw);
}

} // namespace vkcompute
