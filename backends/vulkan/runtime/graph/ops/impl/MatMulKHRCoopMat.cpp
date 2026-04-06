/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

//
// KHR Cooperative Matrix GEMM operator
//
// Computes D = alpha * A * B + beta * C using GL_KHR_cooperative_matrix.
// Cross-vendor: works on Qualcomm, ARM, Intel, NVIDIA.
//
// Shader: runtime/graph/ops/glsl/addmm_khr_cm.glsl
// Variants: matmul_khr_cm_{half,float} (no bias)
//           addmm_khr_cm_{half,float}  (with bias)
//

// Tile configuration — must match shader workgroup layout.
// Workgroup: 8 subgroups (4 wide x 2 high) x 32 threads = 256 threads.
// Cooperative matrix dimensions: lM=lN=lK=16.
static constexpr uint32_t kDefaultLM = 16;
static constexpr uint32_t kDefaultLN = 16;
static constexpr uint32_t kDefaultLK = 16;
static constexpr uint32_t kDefaultTileM = 128;
static constexpr uint32_t kDefaultTileN = 128;
static constexpr uint32_t kDefaultTileK = 32;

static constexpr uint32_t kWorkgroupWidthInSubgroups = 4;
static constexpr uint32_t kWorkgroupHeightInSubgroups = 2;
static constexpr uint32_t kInvocationsPerWorkgroup =
    32 * kWorkgroupWidthInSubgroups * kWorkgroupHeightInSubgroups;

// Derived tile dimensions for shared memory layout
static constexpr uint32_t kDefaultARowLen = kDefaultTileK;
static constexpr uint32_t kDefaultANumRows = kDefaultTileM;
static constexpr uint32_t kDefaultBRowLen = kDefaultTileN;
static constexpr uint32_t kDefaultBNumRows = kDefaultTileK;

// Push constant blocks (each must be <= 16 bytes)
struct KhrCmPushBlock1 {
  uint32_t K;
  uint32_t strideA;
  uint32_t strideB;
  uint32_t strideC;
};

struct KhrCmPushBlock2 {
  uint32_t strideD;
  float alpha;
  float beta;
};

//
// Shader dispatch utilities
//

void resize_khr_cm_gemm_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;

  const ValueRef output = args.at(0).refs.at(0);
  const ValueRef mat1 = args.at(1).refs.at(0);
  const ValueRef mat2 = args.at(1).refs.at(1);

  const std::vector<int64_t> mat1_sizes = graph->sizes_of(mat1);
  const std::vector<int64_t> mat2_sizes = graph->sizes_of(mat2);

  const int64_t M = utils::val_at(-2, mat1_sizes);
  const int64_t N = utils::val_at(-1, mat2_sizes);

  std::vector<int64_t> new_out_sizes(mat1_sizes.size());
  if (mat1_sizes.size() == 2) {
    new_out_sizes.at(0) = M;
    new_out_sizes.at(1) = N;
  } else {
    new_out_sizes.at(0) = mat1_sizes.at(0);
    new_out_sizes.at(1) = M;
    new_out_sizes.at(2) = N;
  }

  graph->virtual_resize(output, new_out_sizes);
}

// Shader selection: matmul (no bias) variant
vkapi::ShaderInfo pick_matmul_khr_cm_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef mat1 = args.at(1).refs.at(0);

  std::string kernel_name = "matmul_khr_cm";
  add_dtype_suffix(kernel_name, graph->dtype_of(mat1));

  return VK_KERNEL_FROM_STR(kernel_name);
}

// Shader selection: addmm (with bias) variant
vkapi::ShaderInfo pick_addmm_khr_cm_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef mat1 = args.at(1).refs.at(0);

  std::string kernel_name = "addmm_khr_cm";
  add_dtype_suffix(kernel_name, graph->dtype_of(mat1));

  return VK_KERNEL_FROM_STR(kernel_name);
}

// Global workgroup size: 2D grid of tiles
utils::uvec3 khr_cm_gemm_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef out = args.at(0).refs.at(0);
  const auto out_sizes = graph->sizes_of(out);

  const uint32_t M = out_sizes.at(out_sizes.size() - 2);
  const uint32_t N = out_sizes.at(out_sizes.size() - 1);

  const uint32_t num_tiles_n = (N + kDefaultTileN - 1) / kDefaultTileN;
  const uint32_t num_tiles_m = (M + kDefaultTileM - 1) / kDefaultTileM;

  return {num_tiles_n * kInvocationsPerWorkgroup, num_tiles_m, 1};
}

// Local workgroup size: 256 threads (8 subgroups x 32 threads)
utils::uvec3 khr_cm_gemm_local_wg_size(
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
  return {kInvocationsPerWorkgroup, 1, 1};
}

//
// Specialization constants (shared by matmul and addmm variants)
//

vkapi::SpecVarList khr_cm_spec_vars() {
  return {
      SV(kDefaultLM), // lM (ID 3)
      SV(kDefaultLN), // lN (ID 4)
      SV(kDefaultLK), // lK (ID 5)
      SV(kDefaultTileM), // TILE_M (ID 6)
      SV(kDefaultTileN), // TILE_N (ID 7)
      SV(kDefaultTileK), // TILE_K (ID 8)
      SV(kDefaultARowLen), // A_ROW_LEN (ID 9)
      SV(kDefaultANumRows), // A_NUM_ROWS (ID 10)
      SV(kDefaultBRowLen), // B_ROW_LEN (ID 11)
      SV(kDefaultBNumRows), // B_NUM_ROWS (ID 12)
      SV(0u), // BColMajor_val (ID 13) - row major
  };
}

//
// Operator implementations
//

// D = A * B (matmul, no bias)
void khr_cm_matmul_impl(
    ComputeGraph& graph,
    const ValueRef input_A,
    const ValueRef input_B,
    const ValueRef output_D) {
  VK_CHECK_COND(
      graph.context()->adapter_ptr()->supports_cooperative_matrix(),
      "khr_cm_gemm requires VK_KHR_cooperative_matrix extension which is "
      "not available on this device.");

  const auto A_sizes = graph.sizes_of(input_A);
  const auto B_sizes = graph.sizes_of(input_B);

  const uint32_t M = A_sizes.at(A_sizes.size() - 2);
  const uint32_t K_val = A_sizes.at(A_sizes.size() - 1);
  const uint32_t N = B_sizes.at(B_sizes.size() - 1);

  const uint32_t strideA = K_val;
  const uint32_t strideB = N;
  const uint32_t strideD = N;

  KhrCmPushBlock1 pb1 = {K_val, strideA, strideB, 0};
  KhrCmPushBlock2 pb2 = {strideD, 1.0f, 0.0f};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_matmul_khr_cm_shader,
      khr_cm_gemm_global_wg_size,
      khr_cm_gemm_local_wg_size,
      {{output_D, vkapi::kWrite}, {{input_A, input_B}, vkapi::kRead}},
      {},
      {PushConstantDataInfo(&pb1, sizeof(pb1)),
       PushConstantDataInfo(&pb2, sizeof(pb2))},
      khr_cm_spec_vars(),
      {},
      resize_khr_cm_gemm_node));
}

// D = alpha * A * B + beta * C (addmm, with bias)
void khr_cm_addmm_impl(
    ComputeGraph& graph,
    const ValueRef input_A,
    const ValueRef input_B,
    const ValueRef input_C,
    const ValueRef output_D,
    const float alpha,
    const float beta) {
  VK_CHECK_COND(
      graph.context()->adapter_ptr()->supports_cooperative_matrix(),
      "khr_cm_gemm requires VK_KHR_cooperative_matrix extension which is "
      "not available on this device.");

  const auto A_sizes = graph.sizes_of(input_A);
  const auto B_sizes = graph.sizes_of(input_B);

  const uint32_t K_val = A_sizes.at(A_sizes.size() - 1);
  const uint32_t N = B_sizes.at(B_sizes.size() - 1);

  const uint32_t strideA = K_val;
  const uint32_t strideB = N;
  const uint32_t strideC = N;
  const uint32_t strideD = N;

  KhrCmPushBlock1 pb1 = {K_val, strideA, strideB, strideC};
  KhrCmPushBlock2 pb2 = {strideD, alpha, beta};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_addmm_khr_cm_shader,
      khr_cm_gemm_global_wg_size,
      khr_cm_gemm_local_wg_size,
      {{output_D, vkapi::kWrite}, {{input_A, input_B, input_C}, vkapi::kRead}},
      {},
      {PushConstantDataInfo(&pb1, sizeof(pb1)),
       PushConstantDataInfo(&pb2, sizeof(pb2))},
      khr_cm_spec_vars(),
      {},
      resize_khr_cm_gemm_node));
}

//
// Registered operator entry points
//

// etvk.khr_cm_gemm.default(A, B, C, alpha, beta) -> D
// General GEMM: D = alpha * A * B + beta * C
void khr_cm_gemm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int idx = 0;
  const ValueRef input_A = args.at(idx++); // [M, K]
  const ValueRef input_B = args.at(idx++); // [K, N]
  const ValueRef input_C = args.at(idx++); // [M, N]
  const ValueRef alpha_ref = args.at(idx++);
  const ValueRef beta_ref = args.at(idx++);
  const ValueRef output_D = args.at(idx++); // [M, N]

  float alpha_val = graph.extract_scalar<double>(alpha_ref);
  float beta_val = graph.extract_scalar<double>(beta_ref);

  if (beta_val == 0.0f) {
    khr_cm_matmul_impl(graph, input_A, input_B, output_D);
  } else {
    khr_cm_addmm_impl(
        graph, input_A, input_B, input_C, output_D, alpha_val, beta_val);
  }
}

//
// Int8 cooperative matrix GEMM
//

// Simplified push constants for int8 (no alpha/beta)
struct KhrCmInt8PushBlock {
  uint32_t K;
  uint32_t strideA;
  uint32_t strideB;
  uint32_t strideD;
};

vkapi::ShaderInfo pick_matmul_khr_cm_int8_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)args;
  (void)resize_args;
  return VK_KERNEL_FROM_STR("matmul_khr_cm_int8");
}

// D = A * B (int8 × int8 → int32)
void khr_cm_matmul_int8_impl(
    ComputeGraph& graph,
    const ValueRef input_A,
    const ValueRef input_B,
    const ValueRef output_D) {
  VK_CHECK_COND(
      graph.context()->adapter_ptr()->supports_cooperative_matrix(),
      "khr_cm_gemm_int8 requires VK_KHR_cooperative_matrix extension which is "
      "not available on this device.");

  const auto A_sizes = graph.sizes_of(input_A);
  const auto B_sizes = graph.sizes_of(input_B);

  const uint32_t K_val = A_sizes.at(A_sizes.size() - 1);
  const uint32_t N = B_sizes.at(B_sizes.size() - 1);

  const uint32_t strideA = K_val;
  const uint32_t strideB = N;
  const uint32_t strideD = N;

  KhrCmInt8PushBlock pb = {K_val, strideA, strideB, strideD};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_matmul_khr_cm_int8_shader,
      khr_cm_gemm_global_wg_size,
      khr_cm_gemm_local_wg_size,
      {{output_D, vkapi::kWrite}, {{input_A, input_B}, vkapi::kRead}},
      {},
      {PushConstantDataInfo(&pb, sizeof(pb))},
      khr_cm_spec_vars(),
      {},
      resize_khr_cm_gemm_node));
}

// etvk.khr_cm_gemm_int8.default(A, B) -> D
void khr_cm_gemm_int8(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueRef input_A = args.at(0); // [M, K] int8
  const ValueRef input_B = args.at(1); // [K, N] int8
  const ValueRef output_D = args.at(2); // [M, N] int32

  khr_cm_matmul_int8_impl(graph, input_A, input_B, output_D);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(etvk.khr_cm_gemm.default, khr_cm_gemm);
  VK_REGISTER_OP(etvk.khr_cm_gemm_int8.default, khr_cm_gemm_int8);
}

} // namespace vkcompute
