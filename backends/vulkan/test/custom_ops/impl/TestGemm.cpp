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

// Test operator that dispatches to either:
//   impl_selector=0 → aten.mm.default with buffer storage (matmul_naive_buffer)
//   impl_selector=2 → aten.mm.default with texture3d channels-packed
//                     (matmul_optimized)
//
// Storage type is controlled by ValueSpec in the test harness, not here.
// Both selectors call the same aten.mm op; the shader selected depends on the
// tensor storage type set at graph construction time.

void test_gemm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef input_A = args.at(idx++); // [M, K]
  const ValueRef input_B = args.at(idx++); // [K, N]
  const ValueRef input_C = args.at(idx++); // [M, N] placeholder (unused)
  const ValueRef alpha_ref = args.at(idx++); // scalar (unused)
  const ValueRef beta_ref = args.at(idx++); // scalar (unused)
  const ValueRef impl_selector_ref = args.at(idx++);
  const ValueRef output = args.at(idx++); // [M, N]

  (void)input_C;
  (void)alpha_ref;
  (void)beta_ref;

  int32_t impl_selector = graph.extract_scalar<int32_t>(impl_selector_ref);

  if (impl_selector == 0 || impl_selector == 2) {
    // aten.mm.default: storage type (buffer vs texture3d) set via ValueSpec
    std::vector<ValueRef> mm_args = {input_A, input_B, output};
    VK_GET_OP_FN("aten.mm.default")(graph, mm_args);
  } else if (impl_selector == 1) {
    // KHR cooperative matrix FP16 GEMM
    std::vector<ValueRef> cm_args = {
        input_A, input_B, input_C, alpha_ref, beta_ref, output};
    VK_GET_OP_FN("etvk.khr_cm_gemm.default")(graph, cm_args);
  } else if (impl_selector == 3) {
    // KHR cooperative matrix int8 GEMM
    std::vector<ValueRef> cm_int8_args = {input_A, input_B, output};
    VK_GET_OP_FN("etvk.khr_cm_gemm_int8.default")(graph, cm_int8_args);
  } else {
    VK_THROW("Invalid impl_selector value: ", impl_selector);
  }
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(test_etvk.test_gemm.default, test_gemm);
}

} // namespace vkcompute
