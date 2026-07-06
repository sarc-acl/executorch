/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Test-only op `test_etvk.dq8ca_tile_sweep.default`, driven by
// test_dq8ca_tile_sweep.cpp (specs/008-8da4w-parameter-sweep). Forces
// dispatch to one of the 12 tile-shape/subgroup-size variants of
// dq8ca_q4gsw_coopmat_sweep.glsl (a test-owned copy of the production
// linear_dq8ca_qw_coopmat.glsl -- see that .glsl's own header comment)
// selected by a `config_id` arg, instead of going through the production
// eligibility-gated picker (pick_linear_dqa_qw_shader in
// QuantizedLinear.cpp, left untouched). Config 0 (the shipped baseline)
// is not represented here at all -- its numbers are reused directly from
// 007's data. Config 12 is a deliberate negative test (WG_TILE_K=64,
// research.md Decision 4 / finding G1), expected to fail correctness.

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizeDequantize.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/QuantizationConfig.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

namespace {

// Mirrors QuantizedLinear.cpp's CoopmatTileDims/kDq8caQ4gswCoopmatDims, but
// keyed by config_id (1-12, research.md Decision 4) rather than by
// kernel-name prefix matching -- that production lookup does not recognize
// this feature's kernel names and would silently resolve to the wrong (fp16
// q4gsw) tile dims if reused naively (research.md Decision 2 correction).
struct SweepTileDims {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t wg_size; // SG_GRID_X * SG_GRID_Y * SUBGROUP_SIZE; SG_GRID is 2x2
                    // for every config in this curated set (Decision 4).
  const char* kernel_name;
};

// clang-format off
constexpr SweepTileDims kSweepDims[13] = {
    /*0 (unused; config 0 is the reused shipped baseline, never built here)*/
    {0, 0, 0, 0, ""},
    /*1*/  {128, 64, 32, 128, "dq8ca_q4gsw_coopmat_sweep_cfg1_buffer_texture2d_half"},
    /*2*/  {64,  64, 32, 256, "dq8ca_q4gsw_coopmat_sweep_cfg2_buffer_texture2d_half"},
    /*3*/  {64,  64, 32, 128, "dq8ca_q4gsw_coopmat_sweep_cfg3_buffer_texture2d_half"},
    /*4*/  {128, 64, 16, 256, "dq8ca_q4gsw_coopmat_sweep_cfg4_buffer_texture2d_half"},
    /*5*/  {128, 64, 16, 128, "dq8ca_q4gsw_coopmat_sweep_cfg5_buffer_texture2d_half"},
    /*6*/  {64,  64, 16, 256, "dq8ca_q4gsw_coopmat_sweep_cfg6_buffer_texture2d_half"},
    /*7*/  {64,  64, 16, 128, "dq8ca_q4gsw_coopmat_sweep_cfg7_buffer_texture2d_half"},
    /*8*/  {256, 64, 32, 256, "dq8ca_q4gsw_coopmat_sweep_cfg8_buffer_texture2d_half"},
    /*9*/  {256, 64, 32, 128, "dq8ca_q4gsw_coopmat_sweep_cfg9_buffer_texture2d_half"},
    /*10*/ {128, 128, 32, 256, "dq8ca_q4gsw_coopmat_sweep_cfg10_buffer_texture2d_half"},
    /*11*/ {128, 128, 32, 128, "dq8ca_q4gsw_coopmat_sweep_cfg11_buffer_texture2d_half"},
    /*12*/ {128, 64, 64, 256, "dq8ca_q4gsw_coopmat_sweep_cfg12_buffer_texture2d_half"},
};
// clang-format on

template <int32_t CONFIG_ID>
vkapi::ShaderInfo pick_sweep_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)args;
  (void)resize_args;
  return VK_KERNEL_FROM_STR(std::string(kSweepDims[CONFIG_ID].kernel_name));
}

template <int32_t CONFIG_ID>
utils::uvec3 pick_sweep_global_wg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  std::vector<int64_t> out_sizes = graph->sizes_of(out);
  const uint32_t N = utils::val_at(-1, out_sizes);
  const uint32_t M = utils::val_at(-2, out_sizes);
  const SweepTileDims& dims = kSweepDims[CONFIG_ID];
  const uint32_t num_tiles_n = utils::div_up(N, dims.n);
  const uint32_t num_tiles_m = utils::div_up(M, dims.m);
  return {num_tiles_n * dims.wg_size, num_tiles_m, 1};
}

template <int32_t CONFIG_ID>
utils::uvec3 pick_sweep_local_wg(
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
  return {kSweepDims[CONFIG_ID].wg_size, 1, 1};
}

using PickShaderFn = vkapi::ShaderInfo (*)(
    ComputeGraph*,
    const std::vector<ArgGroup>&,
    const std::vector<ValueRef>&);
using PickGlobalWgFn = utils::uvec3 (*)(
    ComputeGraph*,
    const vkapi::ShaderInfo&,
    const std::vector<ArgGroup>&,
    const std::vector<ValueRef>&);
using PickLocalWgFn = utils::uvec3 (*)(
    ComputeGraph*,
    const vkapi::ShaderInfo&,
    const utils::uvec3&,
    const std::vector<ArgGroup>&,
    const std::vector<ValueRef>&);

// Maps a runtime config_id (1-12) to its compile-time-templated picker
// function pointers. A plain switch, mirroring
// TestFpaQ4gswLinear.cpp's `pick_forced_shader<KIND>` selection pattern.
void pick_fns_for_config(
    int32_t config_id,
    PickShaderFn& pick_shader,
    PickGlobalWgFn& pick_global,
    PickLocalWgFn& pick_local) {
  switch (config_id) {
    case 1:
      pick_shader = pick_sweep_shader<1>;
      pick_global = pick_sweep_global_wg<1>;
      pick_local = pick_sweep_local_wg<1>;
      break;
    case 2:
      pick_shader = pick_sweep_shader<2>;
      pick_global = pick_sweep_global_wg<2>;
      pick_local = pick_sweep_local_wg<2>;
      break;
    case 3:
      pick_shader = pick_sweep_shader<3>;
      pick_global = pick_sweep_global_wg<3>;
      pick_local = pick_sweep_local_wg<3>;
      break;
    case 4:
      pick_shader = pick_sweep_shader<4>;
      pick_global = pick_sweep_global_wg<4>;
      pick_local = pick_sweep_local_wg<4>;
      break;
    case 5:
      pick_shader = pick_sweep_shader<5>;
      pick_global = pick_sweep_global_wg<5>;
      pick_local = pick_sweep_local_wg<5>;
      break;
    case 6:
      pick_shader = pick_sweep_shader<6>;
      pick_global = pick_sweep_global_wg<6>;
      pick_local = pick_sweep_local_wg<6>;
      break;
    case 7:
      pick_shader = pick_sweep_shader<7>;
      pick_global = pick_sweep_global_wg<7>;
      pick_local = pick_sweep_local_wg<7>;
      break;
    case 8:
      pick_shader = pick_sweep_shader<8>;
      pick_global = pick_sweep_global_wg<8>;
      pick_local = pick_sweep_local_wg<8>;
      break;
    case 9:
      pick_shader = pick_sweep_shader<9>;
      pick_global = pick_sweep_global_wg<9>;
      pick_local = pick_sweep_local_wg<9>;
      break;
    case 10:
      pick_shader = pick_sweep_shader<10>;
      pick_global = pick_sweep_global_wg<10>;
      pick_local = pick_sweep_local_wg<10>;
      break;
    case 11:
      pick_shader = pick_sweep_shader<11>;
      pick_global = pick_sweep_global_wg<11>;
      pick_local = pick_sweep_local_wg<11>;
      break;
    case 12:
      pick_shader = pick_sweep_shader<12>;
      pick_global = pick_sweep_global_wg<12>;
      pick_local = pick_sweep_local_wg<12>;
      break;
    default:
      VK_THROW("dq8ca_tile_sweep: config_id must be 1-12, got ", config_id);
  }
}

// Local copy of QuantizedLinear.cpp's resize_linear_qw_node (not exposed
// via any header; simple output-shape-only logic, no coopmat-specific
// content to get wrong by reimplementing).
void resize_linear_qw_node_local(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  ValueRef output = args.at(0).refs.at(0);
  ValueRef fp_input = args.at(1).refs.at(0);
  ValueRef weight_data = extra_args.at(0);

  std::vector<int64_t> mat1_sizes = graph->sizes_of(fp_input);
  std::vector<int64_t> mat2_sizes = graph->sizes_of(weight_data);

  const int64_t out_cols = utils::val_at(-2, mat1_sizes);
  const int64_t out_rows = utils::val_at(-2, mat2_sizes);

  std::vector<int64_t> new_out_sizes(mat1_sizes.size());
  if (mat1_sizes.size() == 2) {
    new_out_sizes.at(0) = out_cols;
    new_out_sizes.at(1) = out_rows;
  } else {
    new_out_sizes.at(0) = mat1_sizes.at(0);
    new_out_sizes.at(1) = out_cols;
    new_out_sizes.at(2) = out_rows;
  }
  graph->virtual_resize(output, new_out_sizes);
}

// Mirrors add_linear_dqa_qw_node's DynamicDispatchNode construction
// (QuantizedLinear.cpp:616-687) exactly -- same bindings, same
// spec-constant list -- replacing pick_linear_dqa_qw_shader (and its
// eligibility gate) with a fixed picker for the requested config_id.
void add_dq8ca_sweep_node(
    ComputeGraph& graph,
    const ValueRef fp_input,
    const ValueRef packed_int_input,
    const ValueRef int_input_sums,
    const ValueRef packed_input_scale,
    const ValueRef packed_input_zp,
    const ValueRef weight_data,
    const ValueRef packed_weight,
    const ValueRef packed_weight_sums,
    const ValueRef packed_weight_scales,
    const ValueRef group_size,
    const ValueRef bias_data,
    const ValueRef packed_bias,
    const ValueRef output,
    int32_t config_id) {
  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(output), graph.sizes_ubo(fp_input)};

  uint32_t apply_bias = 0;
  if (graph.val_is_not_none(bias_data)) {
    apply_bias = 1;
  }

  int32_t group_size_val = graph.extract_scalar<int32_t>(group_size);
  int32_t K4_per_group = utils::div_up(group_size_val, int32_t(4));
  const int32_t K_dim = graph.size_at<int32_t>(-1, fp_input);
  int32_t coopmat_k_iters = K_dim / group_size_val;

  PickShaderFn pick_shader = nullptr;
  PickGlobalWgFn pick_global = nullptr;
  PickLocalWgFn pick_local = nullptr;
  pick_fns_for_config(config_id, pick_shader, pick_global, pick_local);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_shader,
      pick_global,
      pick_local,
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
      param_buffers,
      {},
      {apply_bias,
       K4_per_group,
       coopmat_k_iters,
       graph.size_at<int32_t>(-1, output)},
      {weight_data, fp_input},
      resize_linear_qw_node_local));
}

} // namespace

// Registered test op. Args mirror et_vk.linear_dq8ca_q4gsw.default exactly
// (fp_input, input_scale, input_zp, weight_data, weight_sums_data,
// weight_scales_data, group_size, bias_data), plus one extra trailing
// config_id scalar arg selecting which of the 12 sweep variants to force.
void test_dq8ca_tile_sweep_op(
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
  const ValueRef config_id_ref = args.at(idx++);
  const ValueRef output = args.at(idx++);

  // config_id_ref carries `config_id * 10000 + salt` -- the salt only
  // exists to keep this op's TestCases from colliding in
  // execute_test_cases's shape-based reference-cache grouping across
  // different ops that happen to share an (M,K,N) shape (see
  // test_dq8ca_tile_sweep.cpp's encode_config_id_arg); it carries no
  // other meaning here.
  const int32_t config_id =
      graph.extract_scalar<int32_t>(config_id_ref) / 10000;
  const int64_t group_size_val = graph.extract_scalar<int64_t>(group_size);

  // Mirrors quantized_linear_impl's dynamic-quant (dq8ca) branch
  // (QuantizedLinear.cpp:700-786) exactly, up to the final dispatch node.
  QuantizationConfig weight_quant_config(4, kPerGroup, {group_size_val});

  const ValueRef packed_weight =
      prepack_quantized_linear_weight(graph, weight_quant_config, weight_data);
  const ValueRef packed_weight_scales = prepack_standard(
      graph, weight_scales_data, utils::kBuffer, utils::kWidthPacked);
  const ValueRef packed_weight_sums = prepack_standard(
      graph, weight_sums_data, utils::kBuffer, utils::kWidthPacked);

  TmpTensor dummy_bias(
      &graph, {}, graph.dtype_of(output), utils::kBuffer, utils::kWidthPacked);
  ValueRef packed_bias = dummy_bias.vref;
  if (graph.val_is_not_none(bias_data)) {
    packed_bias =
        prepack_standard(graph, bias_data, utils::kBuffer, utils::kWidthPacked);
  }

  ValueRef packed_input_scale = input_scale;
  ValueRef packed_input_zp = input_zp;
  if (graph.val_is_tref(input_scale)) {
    packed_input_scale = prepack_standard(
        graph, input_scale, utils::kTexture3D, utils::kWidthPacked);
    packed_input_zp = prepack_standard(
        graph, input_zp, utils::kTexture3D, utils::kWidthPacked);
  }

  TmpTensor packed_int_input(
      &graph,
      graph.sizes_of(fp_input),
      vkapi::kInt8x4,
      utils::kBuffer,
      utils::kPackedInt8_4H4W);

  const int64_t K = graph.size_at<int64_t>(-1, fp_input);
  const int64_t num_groups = K / group_size_val;
  TmpTensor int_input_sums(
      &graph,
      {num_groups, K},
      graph.dtype_of(output),
      utils::kBuffer,
      utils::kWidthPacked);

  QuantizationConfig input_quant_config(8, kPerChannel, {}, false, true);
  add_quantize_and_pack_4h4w_with_group_sums_node(
      graph,
      input_quant_config,
      fp_input,
      int_input_sums,
      packed_input_scale,
      packed_input_zp,
      packed_int_input,
      group_size);

  add_dq8ca_sweep_node(
      graph,
      fp_input,
      packed_int_input,
      int_input_sums,
      packed_input_scale,
      packed_input_zp,
      weight_data,
      packed_weight,
      packed_weight_sums,
      packed_weight_scales,
      group_size,
      bias_data,
      packed_bias,
      output,
      config_id);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(test_etvk.dq8ca_tile_sweep.default, test_dq8ca_tile_sweep_op);
}

} // namespace vkcompute
