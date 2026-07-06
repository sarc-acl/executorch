// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// SDPA coopmat prefill microbenchmark (specs/010-sdpa-coopmat-microbench).
//
// Builds llama.custom_sdpa.default (SDPAMode::LLM, the mode SDPA.cpp's
// coopmat gate applies to) directly via ComputeGraph, mirroring the exact
// construction backends/vulkan/test/op_tests/sdpa_test.cpp's DECOMPOSED mode
// already uses (proven correct and dispatch-confirmable in this feature's
// User Story 1) -- at each target model's real prefill shape (S=2048,
// context_len=2048), Buffer+half storage. Toggles ET_VK_SDPA_COOPMAT to
// compare tiled vs. coopmat dispatch, timing via the GPU query-pool and
// isolating the two coopmat-relevant shader dispatches
// (sdpa_compute_attn_weights_*, sdpa_compute_out_*) from the
// sdpa_kv_cache_update_*/sdpa_attn_weights_softmax_* dispatches in between
// (unaccelerated, identical regardless of the coopmat toggle).
//
// research.md Decision 8: built directly on ComputeGraph rather than the
// TestCase/ValueSpec framework used elsewhere in this directory, since that
// framework has no SymInt support and this op family requires one
// (input_pos_symint) -- confirmed by reading utils.h/utils.cpp directly.
// Correctness of this exact op/shader path was already established in User
// Story 1; this harness only measures timing and confirms dispatch.

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace vkcompute;
using namespace executorch::vulkan::prototyping;

namespace {

struct ModelShape {
  const char* name;
  int64_t head_dim;
  int64_t num_heads;
  int64_t num_kv_heads;
};

// Real per-model shapes, derived directly from each checkpoint's params.json
// (dim / n_heads), matching research.md Decision 5.
const std::vector<ModelShape> kModels = {
    {"llama-3.1-8b", 128, 32, 8},
    {"llama-3.2-3b", 128, 24, 8},
    {"llama-3.2-1b", 64, 32, 8},
};

constexpr int64_t kSeqLen = 2048; // fixed prefill workload (constitution)
constexpr int kWarmupRuns = 3;
constexpr int kTimedRuns = 5;

struct RunResult {
  float mean_us;
  float stdev_us;
  std::vector<std::string> dispatched_kernels; // from the last timed run
};

float mean_of(const std::vector<float>& v) {
  return std::accumulate(v.begin(), v.end(), 0.0f) /
      static_cast<float>(v.size());
}

float stdev_of(const std::vector<float>& v, float mean) {
  if (v.size() < 2) {
    return 0.0f;
  }
  float acc = 0.0f;
  for (float x : v) {
    acc += (x - mean) * (x - mean);
  }
  return std::sqrt(acc / static_cast<float>(v.size() - 1));
}

// Fills a tensor's staging buffer with random half-precision data in
// [-1, 1]. maybe_cast_and_copy_into_staging does not support a Float->Half
// conversion (throws), so the half encoding is done host-side here and
// passed with a matching src dtype.
void fill_random(ComputeGraph& graph, const ValueRef staging, int64_t numel) {
  std::vector<uint16_t> data(numel);
  for (int64_t i = 0; i < numel; ++i) {
    const float v = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f - 1.0f;
    data[i] = float_to_half(v);
  }
  graph.maybe_cast_and_copy_into_staging(
      staging, data.data(), static_cast<size_t>(numel), vkapi::kHalf);
}

// Runs one (model, coopmat-toggle) case: builds the graph once, runs
// kWarmupRuns + kTimedRuns, and returns the SDPA-compute-only GPU time
// (sdpa_compute_attn_weights_* + sdpa_compute_out_*, excluding the
// kv-cache-update and softmax dispatches in between).
RunResult run_case(const ModelShape& m, bool enable_coopmat) {
  if (enable_coopmat) {
    setenv("ET_VK_SDPA_COOPMAT", "1", /*overwrite=*/1);
  } else {
    unsetenv("ET_VK_SDPA_COOPMAT");
  }

  GraphConfig config;
  config.enable_querypool = true;
  api::context()->initialize_querypool();
  ComputeGraph graph(config);

  const int64_t batch_size = 1;
  const std::vector<int64_t> q_sizes = {
      batch_size, kSeqLen, m.num_heads, m.head_dim};
  const std::vector<int64_t> kv_sizes = {
      batch_size, kSeqLen, m.num_kv_heads, m.head_dim};

  IOValueRef r_q =
      graph.add_input_tensor(q_sizes, vkapi::kHalf, utils::kBuffer);
  IOValueRef r_k =
      graph.add_input_tensor(kv_sizes, vkapi::kHalf, utils::kBuffer);
  IOValueRef r_v =
      graph.add_input_tensor(kv_sizes, vkapi::kHalf, utils::kBuffer);

  const ValueRef r_input_pos_symint = graph.add_symint(0);
  const ValueRef r_out =
      graph.add_tensor(q_sizes, vkapi::kHalf, utils::kBuffer);

  const ValueRef r_k_cache =
      graph.add_tensor(kv_sizes, vkapi::kHalf, utils::kBuffer);
  const ValueRef r_v_cache =
      graph.add_tensor(kv_sizes, vkapi::kHalf, utils::kBuffer);
  const ValueRef r_dummy_out =
      graph.add_tensor({1}, vkapi::kHalf, utils::kBuffer);

  VK_GET_OP_FN("update_cache.default")
  (graph, {r_k.value, r_k_cache, r_input_pos_symint, r_dummy_out});
  VK_GET_OP_FN("update_cache.default")
  (graph, {r_v.value, r_v_cache, r_input_pos_symint, r_dummy_out});
  VK_GET_OP_FN("llama.custom_sdpa.default")
  (graph,
   {
       r_q.value,
       r_k_cache,
       r_v_cache,
       r_input_pos_symint,
       kDummyValueRef, // attn_mask
       kDummyValueRef, // dropout_p
       kDummyValueRef, // is_causal
       kDummyValueRef, // scale
       r_out,
   });

  graph.set_output_tensor(r_out);
  graph.prepare();
  graph.prepack();

  fill_random(
      graph, r_q.staging, batch_size * kSeqLen * m.num_heads * m.head_dim);
  fill_random(
      graph, r_k.staging, batch_size * kSeqLen * m.num_kv_heads * m.head_dim);
  fill_random(
      graph, r_v.staging, batch_size * kSeqLen * m.num_kv_heads * m.head_dim);

  for (int i = 0; i < kWarmupRuns; ++i) {
    graph.execute();
  }

  std::vector<float> timings_us;
  std::vector<std::string> last_dispatched;
  for (int i = 0; i < kTimedRuns; ++i) {
    graph.execute();
    graph.context()->querypool().extract_results();
    const auto shader_results =
        graph.context()->querypool().get_shader_timestamp_data();

    float sdpa_time_us = 0.0f;
    last_dispatched.clear();
    for (const auto& r : shader_results) {
      last_dispatched.push_back(r.kernel_name);
      if (r.kernel_name.find("sdpa_compute_attn_weights") !=
              std::string::npos ||
          r.kernel_name.find("sdpa_compute_out") != std::string::npos) {
        const uint64_t duration_ns = r.end_time_ns - r.start_time_ns;
        sdpa_time_us += static_cast<float>(duration_ns) / 1000.0f;
      }
    }
    timings_us.push_back(sdpa_time_us);
  }

  const float mean_us = mean_of(timings_us);
  const float stdev_us = stdev_of(timings_us, mean_us);
  return {mean_us, stdev_us, last_dispatched};
}

bool has_kernel_containing(
    const std::vector<std::string>& kernels,
    const std::string& needle) {
  for (const auto& k : kernels) {
    if (k.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

} // namespace

int main() {
  std::srand(0);

  std::cout << "SDPA coopmat prefill microbenchmark (S=" << kSeqLen << ", "
            << kWarmupRuns << " warmup + " << kTimedRuns
            << " timed runs per case)\n";
  std::cout << std::left << std::setw(16) << "model" << std::right
            << std::setw(14) << "tiled(us)" << std::setw(10) << "±"
            << std::setw(14) << "coopmat(us)" << std::setw(10) << "±"
            << std::setw(10) << "speedup" << "  dispatch\n";

  bool any_failure = false;
  for (const auto& m : kModels) {
    RunResult tiled = run_case(m, /*enable_coopmat=*/false);
    RunResult coopmat = run_case(m, /*enable_coopmat=*/true);

    const bool tiled_is_tiled =
        !has_kernel_containing(tiled.dispatched_kernels, "_coopmat");
    const bool qk_coopmat = has_kernel_containing(
        coopmat.dispatched_kernels, "sdpa_compute_attn_weights_coopmat");
    const bool av_coopmat = has_kernel_containing(
        coopmat.dispatched_kernels, "sdpa_compute_out_coopmat");
    const bool dispatch_confirmed = tiled_is_tiled && qk_coopmat && av_coopmat;
    any_failure = any_failure || !dispatch_confirmed;

    const float speedup_pct =
        (tiled.mean_us - coopmat.mean_us) / tiled.mean_us * 100.0f;

    std::cout << std::left << std::setw(16) << m.name << std::right
              << std::fixed << std::setprecision(1) << std::setw(14)
              << tiled.mean_us << std::setw(10) << tiled.stdev_us
              << std::setw(14) << coopmat.mean_us << std::setw(10)
              << coopmat.stdev_us << std::setw(9) << speedup_pct << "%" << "  "
              << (dispatch_confirmed ? "confirmed" : "NOT CONFIRMED") << "\n";

    std::cout << "RESULT," << m.name << "," << m.head_dim << "," << m.num_heads
              << "," << m.num_kv_heads << "," << kSeqLen << "," << tiled.mean_us
              << "," << tiled.stdev_us << "," << coopmat.mean_us << ","
              << coopmat.stdev_us << ","
              << (dispatch_confirmed ? "confirmed" : "fallback") << "\n";
  }

  if (any_failure) {
    std::cout << "\nOne or more models did not confirm coopmat dispatch -- "
                 "see FR-006/FR-007, do not trust their speedup number.\n";
  }
  return 0;
}
