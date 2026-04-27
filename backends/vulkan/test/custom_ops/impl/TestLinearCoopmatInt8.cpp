/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Phase 4 prototype dispatcher for the int8 cooperative-matrix linear shader.
 *
 * Inputs (all kInt buffers, but interpreted as packed int8 along the inner
 * dim — see linear_coopmat_int8.glsl for the exact layout):
 *   args[0]: A buffer  [M, K/4]   (row-major, 4 int8 per int along K)
 *   args[1]: B buffer  [K, N/4]   (row-major, 4 int8 per int along N)
 *   args[2]: output    [M, N]     (row-major int32 result)
 *
 * The dispatch is benchmark-only. The shader requires K % 32 == 0,
 * M % 64 == 0, N % 64 == 0.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

namespace {

constexpr uint32_t kTileM = 64;
constexpr uint32_t kTileN = 64;
constexpr uint32_t kInvocations = 256;

vkapi::ShaderInfo pick_int8_coopmat_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)args;
  (void)resize_args;
  return VK_KERNEL(linear_coopmat_int8);
}

utils::uvec3 int8_coopmat_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const auto out_sizes = graph->sizes_of(out);
  uint32_t M = out_sizes.at(out_sizes.size() - 2);
  uint32_t N = out_sizes.at(out_sizes.size() - 1);
  uint32_t num_tiles_n = utils::div_up(N, kTileN);
  uint32_t num_tiles_m = utils::div_up(M, kTileM);
  return {num_tiles_n * kInvocations, num_tiles_m, 1};
}

utils::uvec3 int8_coopmat_local_wg_size(
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
  return {kInvocations, 1, 1};
}

} // namespace

void test_linear_coopmat_int8(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef input_a = args.at(idx++);
  const ValueRef input_b = args.at(idx++);
  const ValueRef output = args.at(idx++);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_int8_coopmat_shader,
      int8_coopmat_global_wg_size,
      int8_coopmat_local_wg_size,
      // Inputs and Outputs
      {{output, vkapi::kWrite}, {{input_a, input_b}, vkapi::kRead}},
      // Shader params buffers
      {graph.sizes_ubo(input_a), graph.sizes_ubo(output)},
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(
      test_etvk.linear_coopmat_int8.default, test_linear_coopmat_int8);
}

} // namespace vkcompute
