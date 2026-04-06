/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// KHR Cooperative Matrix GEMM test and benchmark harness.
// Tests D = alpha * A * B + beta * C using GL_KHR_cooperative_matrix.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "cm_utils.h"
#include "utils.h"

using namespace executorch::vulkan::prototyping;

std::vector<TestCase> generate_gemm_test_cases() {
  std::vector<TestCase> test_cases;

  struct GemmConfig {
    int64_t M;
    int64_t N;
    int64_t K;
    std::string name;
  };

  // Matrix sizes must be multiples of TILE_M/TILE_N/TILE_K (128/128/32)
  std::vector<GemmConfig> configs = {
      {128, 128, 128, "128x128x128"},
      {256, 256, 256, "256x256x256"},
      {512, 512, 512, "512x512x512"},
      {1024, 1024, 1024, "1024x1024x1024"},
      {128, 256, 512, "128x256x512"},
  };

  // For each config, generate three test cases:
  // impl_selector=0 → aten.mm.default with buffer (matmul_naive_buffer)
  // impl_selector=1 → etvk.khr_cm_gemm.default (KHR cooperative matrix)
  // impl_selector=2 → aten.mm.default with texture3d channels-packed
  //                    (matmul_optimized)
  for (int impl = 0; impl <= 2; ++impl) {
    for (const auto& cfg : configs) {
      TestCase tc;
      std::string impl_label;
      if (impl == 0) {
        impl_label = "aten_mm";
      } else if (impl == 1) {
        impl_label = "khr_cm";
      } else {
        impl_label = "aten_mm_opt";
      }
      tc.set_name(impl_label + "_" + cfg.name);
      tc.set_operator_name("test_etvk.test_gemm.default");

      // Choose storage type based on implementation
      // impl 0,1 = buffer; impl 2 = texture3d channels-packed
      utils::StorageType storage =
          (impl == 2) ? utils::kTexture3D : utils::kBuffer;
      utils::GPUMemoryLayout layout =
          (impl == 2) ? utils::kChannelsPacked : utils::kWidthPacked;

      // For texture3d path, use float (ATen matmul doesn't support half)
      // For buffer paths, use half (cooperative matrix is half)
      vkapi::ScalarType a_dtype =
          (impl == 2) ? vkapi::kFloat : vkapi::kHalf;
      DataGenType data_gen =
          (impl == 2) ? DataGenType::RANDOM : DataGenType::RANDINT;

      ValueSpec input_A(
          {cfg.M, cfg.K}, a_dtype, storage, layout, data_gen);
      ValueSpec input_B(
          {cfg.K, cfg.N}, a_dtype, storage, layout, data_gen);
      ValueSpec input_C(
          {cfg.M, cfg.N},
          vkapi::kFloat,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::ZEROS);
      ValueSpec alpha_spec(1.0f);
      ValueSpec beta_spec(0.0f);
      ValueSpec impl_selector_spec(static_cast<int32_t>(impl));
      ValueSpec output_D(
          {cfg.M, cfg.N}, vkapi::kFloat, storage, layout, DataGenType::ZEROS);

      tc.add_input_spec(input_A);
      tc.add_input_spec(input_B);
      tc.add_input_spec(input_C);
      tc.add_input_spec(alpha_spec);
      tc.add_input_spec(beta_spec);
      tc.add_input_spec(impl_selector_spec);
      tc.add_output_spec(output_D);

      // Relaxed tolerance for cooperative matrix / fp16
      tc.set_abs_tolerance(1e-1f);
      tc.set_rel_tolerance(1e-1f);

      test_cases.push_back(tc);
    }
  }

  return test_cases;
}

int64_t gemm_flop_calculator(const TestCase& test_case) {
  if (test_case.empty() || test_case.num_inputs() < 2) {
    return 0;
  }
  const auto& A_sizes = test_case.inputs()[0].get_tensor_sizes();
  const auto& B_sizes = test_case.inputs()[1].get_tensor_sizes();

  int64_t M = A_sizes.at(A_sizes.size() - 2);
  int64_t K = A_sizes.at(A_sizes.size() - 1);
  int64_t N = B_sizes.at(B_sizes.size() - 1);

  // GEMM: 2*M*N*K FLOPs (multiply + accumulate)
  return 2 * M * N * K;
}

// IEEE 754 half-precision to float conversion
static float half_to_float(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exponent = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x3FF;

  uint32_t f_sign = sign << 31;
  uint32_t f_exp;
  uint32_t f_mant;

  if (exponent == 0) {
    if (mantissa == 0) {
      f_exp = 0;
      f_mant = 0;
    } else {
      // Denormalized
      uint32_t exp_adj = 1;
      uint32_t mant_temp = mantissa;
      while ((mant_temp & 0x400) == 0) {
        mant_temp <<= 1;
        exp_adj--;
      }
      mant_temp &= 0x3FF;
      f_exp = (127 - 15 + exp_adj) << 23;
      f_mant = mant_temp << 13;
    }
  } else if (exponent == 31) {
    f_exp = 0xFF << 23;
    f_mant = mantissa << 13;
  } else {
    f_exp = (exponent + 127 - 15) << 23;
    f_mant = mantissa << 13;
  }

  uint32_t bits = f_sign | f_exp | f_mant;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

// Skip CPU reference for matrices larger than this
static constexpr int64_t kRefDimSizeLimit = 2048;

void gemm_reference_compute(TestCase& test_case) {
  const ValueSpec& input_A = test_case.inputs().at(0);
  const ValueSpec& input_B = test_case.inputs().at(1);
  ValueSpec& output = test_case.outputs().at(0);

  const auto& A_sizes = input_A.get_tensor_sizes();
  const auto& B_sizes = input_B.get_tensor_sizes();

  const int64_t M = A_sizes.at(A_sizes.size() - 2);
  const int64_t K = A_sizes.at(A_sizes.size() - 1);
  const int64_t N = B_sizes.at(B_sizes.size() - 1);

  // Skip CPU reference for very large matrices
  if (M > kRefDimSizeLimit || K > kRefDimSizeLimit || N > kRefDimSizeLimit) {
    std::cerr << "Skipping reference compute for large matrix ("
              << M << "x" << K << "x" << N << ")" << std::endl;
    return;
  }

  auto& ref_data = output.get_ref_float_data();
  ref_data.resize(M * N, 0.0f);

  if (input_A.dtype == vkapi::kHalf) {
    // Half-precision: use get_half_data() + manual conversion
    const auto& A_half = input_A.get_half_data();
    const auto& B_half = input_B.get_half_data();

    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          float a_val = half_to_float(A_half[m * K + k]);
          float b_val = half_to_float(B_half[k * N + n]);
          sum += a_val * b_val;
        }
        ref_data[m * N + n] = sum;
      }
    }
  } else {
    // Float: direct access
    const auto& A_data = input_A.get_float_data();
    const auto& B_data = input_B.get_float_data();

    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          sum += A_data[m * K + k] * B_data[k * N + n];
        }
        ref_data[m * N + n] = sum;
      }
    }
  }
}

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  set_print_output(false);
  set_print_latencies(true);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "KHR Cooperative Matrix GEMM Prototyping" << std::endl;
  print_separator();

  try {
    api::context()->initialize_querypool();
  } catch (const std::exception& e) {
    std::cerr << "Failed to initialize Vulkan context: " << e.what()
              << std::endl;
    return 1;
  }

  // Query and print cooperative matrix properties
  queryCooperativeMatrixProperties();

  // Check device support
  if (!api::context()->adapter_ptr()->supports_cooperative_matrix()) {
    std::cerr
        << "VK_KHR_cooperative_matrix is not supported on this device. "
        << "Skipping tests." << std::endl;
    return 0;
  }

  auto results = execute_test_cases(
      generate_gemm_test_cases,
      gemm_flop_calculator,
      "KHR_CM_GEMM",
      3,   // warmup_runs
      10,  // benchmark_runs
      gemm_reference_compute);

  return 0;
}
