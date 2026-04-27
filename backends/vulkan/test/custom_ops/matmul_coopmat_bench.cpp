/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Microbenchmark: matmul_coopmat vs matmul_vec.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

#include "cm_utils.h"
#include "utils.h"

using namespace executorch::vulkan::prototyping;

std::vector<TestCase> generate_test_cases() {
  std::vector<TestCase> test_cases;

  struct MatmulConfig {
    int64_t M, K, N;
    std::string name;
  };

  std::vector<MatmulConfig> configs = {
      {256, 768, 3072, "BERT_FFN_up"},
      {256, 3072, 768, "BERT_FFN_down"},
      {128, 768, 768, "BERT_QKV"},
      {64, 4096, 4096, "LLM_QKV_64tok"},
      {256, 1024, 1024, "sq_1024"},
      {256, 4096, 4096, "sq_4096"},
      {4096, 4096, 4096, "sq_4096_cube"},
  };

  for (const auto& cfg : configs) {
    TestCase tc;
    tc.set_name("vec_tex_fp16_" + cfg.name);
    tc.set_operator_name("test_etvk.test_mm.default");

    ValueSpec input_A(
        {cfg.M, cfg.K},
        vkapi::kHalf,
        utils::kTexture3D,
        utils::kWidthPacked,
        DataGenType::RANDOM);
    ValueSpec input_B(
        {cfg.K, cfg.N},
        vkapi::kHalf,
        utils::kTexture3D,
        utils::kWidthPacked,
        DataGenType::RANDOM);
    ValueSpec impl_selector = ValueSpec::make_string("default");
    ValueSpec output(
        {cfg.M, cfg.N},
        vkapi::kHalf,
        utils::kTexture3D,
        utils::kWidthPacked,
        DataGenType::ZEROS);

    tc.add_input_spec(input_A);
    tc.add_input_spec(input_B);
    tc.add_input_spec(impl_selector);
    tc.add_output_spec(output);
    tc.set_abs_tolerance(1.0f);
    tc.set_rel_tolerance(1e-1f);
    test_cases.push_back(tc);
  }

  for (const auto& cfg : configs) {
    TestCase tc;
    tc.set_name("cm_fp16_" + cfg.name);
    tc.set_operator_name("test_etvk.test_mm.default");

    ValueSpec input_A(
        {cfg.M, cfg.K},
        vkapi::kHalf,
        utils::kBuffer,
        utils::kWidthPacked,
        DataGenType::RANDOM);
    ValueSpec input_B(
        {cfg.K, cfg.N},
        vkapi::kHalf,
        utils::kBuffer,
        utils::kWidthPacked,
        DataGenType::RANDOM);
    ValueSpec impl_selector = ValueSpec::make_string("default");
    ValueSpec output(
        {cfg.M, cfg.N},
        vkapi::kHalf,
        utils::kBuffer,
        utils::kWidthPacked,
        DataGenType::ZEROS);

    tc.add_input_spec(input_A);
    tc.add_input_spec(input_B);
    tc.add_input_spec(impl_selector);
    tc.add_output_spec(output);
    tc.set_abs_tolerance(1.0f);
    tc.set_rel_tolerance(1e-1f);
    test_cases.push_back(tc);
  }

  return test_cases;
}

int64_t matmul_flops(const TestCase& test_case) {
  if (test_case.empty() || test_case.num_inputs() < 2) {
    return 0;
  }
  const auto& A = test_case.inputs()[0].get_tensor_sizes();
  const auto& B = test_case.inputs()[1].get_tensor_sizes();
  int64_t M = A.at(A.size() - 2);
  int64_t K = A.at(A.size() - 1);
  int64_t N = B.at(B.size() - 1);
  return 2 * M * N * K;
}

static constexpr int64_t kRefLimit = 1024;
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

void matmul_reference(TestCase& test_case) {
  const ValueSpec& A_spec = test_case.inputs().at(0);
  const ValueSpec& B_spec = test_case.inputs().at(1);
  ValueSpec& out_spec = test_case.outputs().at(0);

  const auto& A_sizes = A_spec.get_tensor_sizes();
  const auto& B_sizes = B_spec.get_tensor_sizes();
  int64_t M = A_sizes.at(A_sizes.size() - 2);
  int64_t K = A_sizes.at(A_sizes.size() - 1);
  int64_t N = B_sizes.at(B_sizes.size() - 1);

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
        sum += half_to_float(A_h[m * K + k]) * half_to_float(B_h[k * N + n]);
      }
      ref[idx] = sum;
    }
    return;
  }

  ref.resize(M * N, 0.0f);
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        sum += half_to_float(A_h[m * K + k]) * half_to_float(B_h[k * N + n]);
      }
      ref[m * N + n] = sum;
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
  std::cout << "Matmul Coopmat vs Vec Microbenchmark" << std::endl;
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
        << "Cooperative matrix: NOT supported (buffer tests will use matmul_vec)"
        << std::endl;
  }

  auto results = execute_test_cases(
      generate_test_cases,
      matmul_flops,
      "MATMUL_COOPMAT_BENCH",
      3,
      10,
      matmul_reference);

  return 0;
}
