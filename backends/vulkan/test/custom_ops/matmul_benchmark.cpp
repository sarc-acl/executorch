/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Side-by-side benchmark of matmul implementations at the same matrix size.
// Compares:
//   0  = aten.mm naive buffer        (matmul_naive_buffer, FP16)
//   1  = KHR CoopMat FP16            (matmul_khr_cm_half)
//   2  = aten.mm optimized tex3d     (matmul_optimized, FP32)
//   2b = aten.mm optimized tex3d     (matmul_optimized, FP16)
//   3  = KHR CoopMat int8            (matmul_khr_cm_int8)
//   4  = et_vk.linear_q8ta_q8csw    (int8 × int8, buffer)
//   5  = et_vk.linear_q8csw         (FP16 input × int8 weight, buffer)

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <iostream>
#include <vector>

#include "cm_utils.h"
#include "utils.h"

using namespace executorch::vulkan::prototyping;
using namespace vkcompute;

std::vector<TestCase> generate_benchmark_test_cases() {
  std::vector<TestCase> test_cases;

  struct BenchConfig {
    int64_t M;
    int64_t N;
    int64_t K;
    std::string name;
  };

  std::vector<BenchConfig> configs = {
      {1024, 1024, 1024, "1024x1024x1024"},
      {4096, 4096, 4096, "4096x4096x4096"},
  };

  for (const auto& cfg : configs) {
    int64_t M = cfg.M, K = cfg.K, N = cfg.N;

    // --- impl 0: aten.mm naive buffer (FP16) ---
    {
      TestCase tc;
      tc.set_name("aten_mm_buffer_" + cfg.name);
      tc.set_operator_name("test_etvk.test_gemm.default");

      ValueSpec input_A(
          {M, K},
          vkapi::kHalf,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::RANDINT);
      ValueSpec input_B(
          {K, N},
          vkapi::kHalf,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::RANDINT);
      ValueSpec input_C(
          {M, N},
          vkapi::kFloat,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::ZEROS);
      ValueSpec alpha_spec(1.0f);
      ValueSpec beta_spec(0.0f);
      ValueSpec impl_selector_spec(static_cast<int32_t>(0));
      ValueSpec output_D(
          {M, N},
          vkapi::kHalf,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::ZEROS);

      tc.add_input_spec(input_A);
      tc.add_input_spec(input_B);
      tc.add_input_spec(input_C);
      tc.add_input_spec(alpha_spec);
      tc.add_input_spec(beta_spec);
      tc.add_input_spec(impl_selector_spec);
      tc.add_output_spec(output_D);
      tc.set_shader_filter(
          {"nchw_to_buffer",
           "buffer_to_nchw",
           "nchw_to_image",
           "image_to_nchw"});
      tc.set_abs_tolerance(1e10f);
      tc.set_rel_tolerance(1.0f);
      test_cases.push_back(tc);
    }

    // --- impl 1: KHR CoopMat FP16 ---
    {
      TestCase tc;
      tc.set_name("khr_cm_fp16_" + cfg.name);
      tc.set_operator_name("test_etvk.test_gemm.default");

      ValueSpec input_A(
          {M, K},
          vkapi::kHalf,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::RANDINT);
      ValueSpec input_B(
          {K, N},
          vkapi::kHalf,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::RANDINT);
      ValueSpec input_C(
          {M, N},
          vkapi::kFloat,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::ZEROS);
      ValueSpec alpha_spec(1.0f);
      ValueSpec beta_spec(0.0f);
      ValueSpec impl_selector_spec(static_cast<int32_t>(1));
      ValueSpec output_D(
          {M, N},
          vkapi::kFloat,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::ZEROS);

      tc.add_input_spec(input_A);
      tc.add_input_spec(input_B);
      tc.add_input_spec(input_C);
      tc.add_input_spec(alpha_spec);
      tc.add_input_spec(beta_spec);
      tc.add_input_spec(impl_selector_spec);
      tc.add_output_spec(output_D);
      tc.set_shader_filter(
          {"nchw_to_buffer",
           "buffer_to_nchw",
           "nchw_to_image",
           "image_to_nchw"});
      tc.set_abs_tolerance(1e10f);
      tc.set_rel_tolerance(1.0f);
      test_cases.push_back(tc);
    }

    // --- impl 2: aten.mm optimized texture3d (FP32) ---
    {
      TestCase tc;
      tc.set_name("aten_mm_optimized_fp32_" + cfg.name);
      tc.set_operator_name("test_etvk.test_gemm.default");

      ValueSpec input_A(
          {M, K},
          vkapi::kFloat,
          utils::kTexture3D,
          utils::kChannelsPacked,
          DataGenType::RANDOM);
      ValueSpec input_B(
          {K, N},
          vkapi::kFloat,
          utils::kTexture3D,
          utils::kChannelsPacked,
          DataGenType::RANDOM);
      ValueSpec input_C(
          {M, N},
          vkapi::kFloat,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::ZEROS);
      ValueSpec alpha_spec(1.0f);
      ValueSpec beta_spec(0.0f);
      ValueSpec impl_selector_spec(static_cast<int32_t>(2));
      ValueSpec output_D(
          {M, N},
          vkapi::kFloat,
          utils::kTexture3D,
          utils::kChannelsPacked,
          DataGenType::ZEROS);

      tc.add_input_spec(input_A);
      tc.add_input_spec(input_B);
      tc.add_input_spec(input_C);
      tc.add_input_spec(alpha_spec);
      tc.add_input_spec(beta_spec);
      tc.add_input_spec(impl_selector_spec);
      tc.add_output_spec(output_D);
      tc.set_shader_filter(
          {"nchw_to_buffer",
           "buffer_to_nchw",
           "nchw_to_image",
           "image_to_nchw",
           "view_"});
      tc.set_abs_tolerance(1e10f);
      tc.set_rel_tolerance(1.0f);
      test_cases.push_back(tc);
    }

    // --- impl 2b: aten.mm optimized texture3d (FP16) ---
    {
      TestCase tc;
      tc.set_name("aten_mm_optimized_fp16_" + cfg.name);
      tc.set_operator_name("test_etvk.test_gemm.default");

      ValueSpec input_A(
          {M, K},
          vkapi::kHalf,
          utils::kTexture3D,
          utils::kChannelsPacked,
          DataGenType::RANDINT);
      ValueSpec input_B(
          {K, N},
          vkapi::kHalf,
          utils::kTexture3D,
          utils::kChannelsPacked,
          DataGenType::RANDINT);
      ValueSpec input_C(
          {M, N},
          vkapi::kFloat,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::ZEROS);
      ValueSpec alpha_spec(1.0f);
      ValueSpec beta_spec(0.0f);
      ValueSpec impl_selector_spec(static_cast<int32_t>(2));
      ValueSpec output_D(
          {M, N},
          vkapi::kHalf,
          utils::kTexture3D,
          utils::kChannelsPacked,
          DataGenType::ZEROS);

      tc.add_input_spec(input_A);
      tc.add_input_spec(input_B);
      tc.add_input_spec(input_C);
      tc.add_input_spec(alpha_spec);
      tc.add_input_spec(beta_spec);
      tc.add_input_spec(impl_selector_spec);
      tc.add_output_spec(output_D);
      tc.set_shader_filter(
          {"nchw_to_buffer",
           "buffer_to_nchw",
           "nchw_to_image",
           "image_to_nchw",
           "view_"});
      tc.set_abs_tolerance(1e10f);
      tc.set_rel_tolerance(1.0f);
      test_cases.push_back(tc);
    }

    // --- impl 3: KHR CoopMat int8 ---
    {
      TestCase tc;
      tc.set_name("khr_cm_int8_" + cfg.name);
      tc.set_operator_name("test_etvk.test_gemm.default");

      ValueSpec input_A(
          {M, K},
          vkapi::kByte,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::RANDINT8);
      ValueSpec input_B(
          {K, N},
          vkapi::kByte,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::RANDINT8);
      ValueSpec input_C(
          {M, N},
          vkapi::kFloat,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::ZEROS);
      ValueSpec alpha_spec(1.0f);
      ValueSpec beta_spec(0.0f);
      ValueSpec impl_selector_spec(static_cast<int32_t>(3));
      ValueSpec output_D(
          {M, N},
          vkapi::kFloat,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::ZEROS);

      tc.add_input_spec(input_A);
      tc.add_input_spec(input_B);
      tc.add_input_spec(input_C);
      tc.add_input_spec(alpha_spec);
      tc.add_input_spec(beta_spec);
      tc.add_input_spec(impl_selector_spec);
      tc.add_output_spec(output_D);
      tc.set_shader_filter(
          {"nchw_to_buffer",
           "buffer_to_nchw",
           "nchw_to_image",
           "image_to_nchw"});
      tc.set_abs_tolerance(1e10f);
      tc.set_rel_tolerance(1.0f);
      test_cases.push_back(tc);
    }

    // --- impl 4: et_vk.linear_q8ta_q8csw (int8 × int8, buffer) ---
    {
      TestCase tc;
      tc.set_name("et_q8ta_q8csw_buf_" + cfg.name);
      tc.set_operator_name("et_vk.linear_q8ta_q8csw.default");

      // Float input [M, K]
      ValueSpec input_tensor(
          {M, K},
          vkapi::kFloat,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::RANDOM);
      ValueSpec input_scale(0.008f);
      ValueSpec input_zero_point(static_cast<int32_t>(-2));

      // Quantized weight: int8 [N, K]
      ValueSpec quantized_weight(
          {N, K},
          vkapi::kChar,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::RANDINT8);
      quantized_weight.set_constant(true);

      // Weight sums: per output channel [N]
      ValueSpec weight_sums(
          {N},
          vkapi::kInt,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::ZEROS);
      weight_sums.set_constant(true);
      compute_weight_sums(weight_sums, quantized_weight, N, K);

      // Weight scales: per output channel [N]
      ValueSpec weight_scales(
          {N},
          vkapi::kFloat,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::RANDOM_SCALES);
      weight_scales.set_constant(true);

      // No bias
      ValueSpec bias;
      bias.set_none(true);

      // Float output [M, N]
      ValueSpec output(
          {M, N},
          vkapi::kFloat,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::ZEROS);

      tc.add_input_spec(input_tensor);
      tc.add_input_spec(input_scale);
      tc.add_input_spec(input_zero_point);
      tc.add_input_spec(quantized_weight);
      tc.add_input_spec(weight_sums);
      tc.add_input_spec(weight_scales);
      tc.add_input_spec(bias);
      tc.add_output_spec(output);
      tc.set_shader_filter(
          {"nchw_to_buffer",
           "buffer_to_nchw",
           "nchw_to_image",
           "image_to_nchw",
           "pack_q8"});
      tc.set_abs_tolerance(1e10f);
      tc.set_rel_tolerance(1.0f);
      test_cases.push_back(tc);
    }

    // --- impl 5: et_vk.linear_q8csw (FP input × int8 weight, buffer) ---
    {
      TestCase tc;
      tc.set_name("et_q8csw_buf_" + cfg.name);
      tc.set_operator_name("et_vk.linear_q8csw.default");

      // Float input [M, K]
      ValueSpec input_tensor(
          {M, K},
          vkapi::kFloat,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::RANDOM);

      // Quantized weight: int8 [N, K]
      ValueSpec quantized_weight(
          {N, K},
          vkapi::kChar,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::RANDINT8);
      quantized_weight.set_constant(true);

      // Weight scales: per output channel [N]
      ValueSpec weight_scales(
          {N},
          vkapi::kFloat,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::RANDOM_SCALES);
      weight_scales.set_constant(true);

      // No bias
      ValueSpec bias;
      bias.set_none(true);

      // Float output [M, N]
      ValueSpec output(
          {M, N},
          vkapi::kFloat,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::ZEROS);

      tc.add_input_spec(input_tensor);
      tc.add_input_spec(quantized_weight);
      tc.add_input_spec(weight_scales);
      tc.add_input_spec(bias);
      tc.add_output_spec(output);
      tc.set_shader_filter(
          {"nchw_to_buffer",
           "buffer_to_nchw",
           "nchw_to_image",
           "image_to_nchw",
           "pack_q8"});
      tc.set_abs_tolerance(1e10f);
      tc.set_rel_tolerance(1.0f);
      test_cases.push_back(tc);
    }
  }

  return test_cases;
}

int64_t benchmark_flop_calculator(const TestCase& test_case) {
  if (test_case.empty() || test_case.num_inputs() < 2) {
    return 0;
  }
  const auto& A_sizes = test_case.inputs()[0].get_tensor_sizes();
  int64_t M = A_sizes.at(A_sizes.size() - 2);
  int64_t K = A_sizes.at(A_sizes.size() - 1);
  const auto& out_sizes = test_case.outputs()[0].get_tensor_sizes();
  int64_t N = out_sizes.at(out_sizes.size() - 1);
  return 2 * M * N * K;
}

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  set_print_output(false);
  set_print_latencies(true);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "Matmul Benchmark" << std::endl;
  std::cout << "Comparing: naive buffer | KHR CM FP16 | optimized tex3d "
               "FP32/FP16 | KHR CM int8 | q8ta_q8csw | q8csw"
            << std::endl;
  print_separator();

  try {
    api::context()->initialize_querypool();
  } catch (const std::exception& e) {
    std::cerr << "Failed to initialize Vulkan context: " << e.what()
              << std::endl;
    return 1;
  }

  queryCooperativeMatrixProperties();

  if (!api::context()->adapter_ptr()->supports_cooperative_matrix()) {
    std::cerr << "VK_KHR_cooperative_matrix not supported on this device. "
              << "KHR CoopMat cases will be skipped at dispatch." << std::endl;
  }

  execute_test_cases(
      generate_benchmark_test_cases,
      benchmark_flop_calculator,
      "MATMUL_BASELINE",
      3, // warmup_runs
      10, // benchmark_runs
      nullptr); // no reference compute — performance only

  return 0;
}
