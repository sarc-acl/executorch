/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Microbenchmark: linear_coopmat vs linear_vec.
//
// Uses test_etvk.test_linear.default which routes to aten.linear.default.
//   - texture3d output  -> linear_vec (Stephen's tiled shader)
//   - buffer output + coop mat device -> linear_coopmat (KHR cooperative
//   matrix)
//
// For each matrix size and dtype, runs two variants:
//   vec_tex: input=tex3d, weight=tex3d(constant), out=tex3d -> linear_vec
//   cm:      input=buf, weight=buf(constant), out=buf -> linear_coopmat

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include "cm_utils.h"
#include "utils.h"

using namespace executorch::vulkan::prototyping;

std::vector<TestCase> generate_test_cases() {
  std::vector<TestCase> test_cases;

  struct LinearConfig {
    int64_t M, K, N;
    std::string name;
  };

  std::vector<LinearConfig> configs = {
      // BERT-like shapes
      {256, 768, 3072, "BERT_FFN_up"},
      {256, 3072, 768, "BERT_FFN_down"},
      {128, 768, 768, "BERT_QKV"},
      // LLM-like shapes (single token)
      {1, 4096, 4096, "LLM_QKV_1tok"},
      {1, 4096, 11008, "LLM_FFN_up_1tok"},
      {1, 11008, 4096, "LLM_FFN_down_1tok"},
      // LLM-like shapes (batch)
      {32, 4096, 4096, "LLM_QKV_32tok"},
      {32, 4096, 11008, "LLM_FFN_up_32tok"},
      // Square stress
      {256, 1024, 1024, "sq_1024"},
      {256, 4096, 4096, "sq_4096"},
      {4096, 4096, 4096, "sq_4096_cube"},
  };

  const std::vector<vkapi::ScalarType> dtypes = {vkapi::kHalf};

  for (vkapi::ScalarType dtype : dtypes) {
    const std::string dtype_name = dtype == vkapi::kHalf ? "fp16" : "fp32";
    for (const auto& cfg : configs) {
      TestCase tc;
      tc.set_name("vec_tex_" + dtype_name + "_" + cfg.name);
      tc.set_operator_name("test_etvk.test_linear.default");

      ValueSpec input_A(
          {cfg.M, cfg.K},
          dtype,
          utils::kTexture3D,
          utils::kWidthPacked,
          DataGenType::RANDOM);
      ValueSpec input_B(
          {cfg.N, cfg.K},
          dtype,
          utils::kTexture3D,
          utils::kWidthPacked,
          DataGenType::RANDOM);
      input_B.set_constant(true);
      ValueSpec none_bias(static_cast<int32_t>(0));
      none_bias.set_none(true);
      ValueSpec impl_selector = ValueSpec::make_string("default");
      ValueSpec output(
          {cfg.M, cfg.N},
          dtype,
          utils::kTexture3D,
          utils::kWidthPacked,
          DataGenType::ZEROS);

      tc.add_input_spec(input_A);
      tc.add_input_spec(input_B);
      tc.add_input_spec(none_bias);
      tc.add_input_spec(impl_selector);
      tc.add_output_spec(output);
      tc.set_abs_tolerance(dtype == vkapi::kHalf ? 1.0f : 1e-2f);
      tc.set_rel_tolerance(dtype == vkapi::kHalf ? 1e-1f : 1e-1f);
      test_cases.push_back(tc);
    }
  }

  for (vkapi::ScalarType dtype : dtypes) {
    const std::string dtype_name = dtype == vkapi::kHalf ? "fp16" : "fp32";
    for (const auto& cfg : configs) {
      TestCase tc;
      tc.set_name("cm_" + dtype_name + "_" + cfg.name);
      tc.set_operator_name("test_etvk.test_linear.default");

      ValueSpec input_A(
          {cfg.M, cfg.K},
          dtype,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::RANDOM);
      ValueSpec input_B(
          {cfg.N, cfg.K},
          dtype,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::RANDOM);
      input_B.set_constant(true);
      ValueSpec none_bias(static_cast<int32_t>(0));
      none_bias.set_none(true);
      ValueSpec impl_selector = ValueSpec::make_string("default");
      ValueSpec output(
          {cfg.M, cfg.N},
          dtype,
          utils::kBuffer,
          utils::kWidthPacked,
          DataGenType::ZEROS);

      tc.add_input_spec(input_A);
      tc.add_input_spec(input_B);
      tc.add_input_spec(none_bias);
      tc.add_input_spec(impl_selector);
      tc.add_output_spec(output);
      tc.set_abs_tolerance(dtype == vkapi::kHalf ? 1.0f : 5e-1f);
      tc.set_rel_tolerance(dtype == vkapi::kHalf ? 1e-1f : 5e-1f);
      test_cases.push_back(tc);
    }
  }

  return test_cases;
}

int64_t linear_flops(const TestCase& test_case) {
  if (test_case.empty() || test_case.num_inputs() < 2)
    return 0;
  const auto& A = test_case.inputs()[0].get_tensor_sizes();
  const auto& B = test_case.inputs()[1].get_tensor_sizes();
  int64_t M = A.at(A.size() - 2);
  int64_t K = A.at(A.size() - 1);
  int64_t N = B.at(B.size() - 2);
  return 2 * M * N * K;
}

static constexpr int64_t kRefLimit = 2048;
static constexpr int64_t kLargeRefSamples = 8192;

std::vector<int64_t> sampled_output_indices(int64_t M, int64_t N) {
  const int64_t total = M * N;
  const int64_t sample_count = std::min(kLargeRefSamples, total);
  std::vector<int64_t> indices;
  indices.reserve(sample_count);
  for (int64_t i = 0; i < sample_count; ++i) {
    indices.push_back((i * 104729) % total);
  }
  return indices;
}

void linear_reference(TestCase& test_case) {
  const ValueSpec& A_spec = test_case.inputs().at(0);
  const ValueSpec& B_spec = test_case.inputs().at(1);
  ValueSpec& out_spec = test_case.outputs().at(0);

  const auto& A_sizes = A_spec.get_tensor_sizes();
  const auto& B_sizes = B_spec.get_tensor_sizes();
  int64_t M = A_sizes.at(A_sizes.size() - 2);
  int64_t K = A_sizes.at(A_sizes.size() - 1);
  int64_t N = B_sizes.at(B_sizes.size() - 2);

  const auto& A_f = A_spec.get_float_data();
  const auto& B_f = B_spec.get_float_data();
  const auto& A_h = A_spec.get_half_data();
  const auto& B_h = B_spec.get_half_data();
  auto& ref = out_spec.get_ref_float_data();

  if (M > kRefLimit || K > kRefLimit || N > kRefLimit) {
    ref.assign(M * N, std::numeric_limits<float>::quiet_NaN());
    const auto sample_indices = sampled_output_indices(M, N);
    std::cerr << "Computing sampled reference for large matrix (" << M << "x"
              << K << "x" << N << "), samples=" << sample_indices.size()
              << std::endl;
    for (int64_t idx : sample_indices) {
      const int64_t m = idx / N;
      const int64_t n = idx % N;
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        if (A_spec.dtype == vkapi::kHalf) {
          sum += half_to_float(A_h[m * K + k]) * half_to_float(B_h[n * K + k]);
        } else {
          sum += A_f[m * K + k] * B_f[n * K + k];
        }
      }
      ref[idx] = sum;
    }
    return;
  }

  ref.resize(M * N, 0.0f);
  for (int64_t m = 0; m < M; ++m)
    for (int64_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        if (A_spec.dtype == vkapi::kHalf) {
          sum += half_to_float(A_h[m * K + k]) * half_to_float(B_h[n * K + k]);
        } else {
          sum += A_f[m * K + k] * B_f[n * K + k];
        }
      }
      ref[m * N + n] = sum;
    }
}

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  set_print_output(false);
  set_print_latencies(true);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "Linear Coopmat vs Vec Microbenchmark" << std::endl;
  print_separator();

  try {
    api::context()->initialize_querypool();
  } catch (const std::exception& e) {
    std::cerr << "Failed to initialize Vulkan: " << e.what() << std::endl;
    return 1;
  }

  if (api::context()->adapter_ptr()->supports_cooperative_matrix()) {
    std::cout << "Cooperative matrix: SUPPORTED" << std::endl;
    queryCooperativeMatrixProperties();
  } else {
    std::cout
        << "Cooperative matrix: NOT supported (buffer tests will use linear_vec)"
        << std::endl;
  }

  auto results = execute_test_cases(
      generate_test_cases,
      linear_flops,
      "LINEAR_COOPMAT_BENCH",
      3, // warmup
      10, // benchmark runs
      linear_reference);

  return 0;
}
