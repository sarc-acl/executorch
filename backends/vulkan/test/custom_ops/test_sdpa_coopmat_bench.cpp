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

constexpr int kWarmupRuns = 3;
constexpr int kTimedRuns = 5;

// specs/021: real e2e regimes. decode's context_len=3072 matches this
// workstream's standard ctx3072 PTEs; input_pos=3071 is the single most
// expensive real decode step (attends over the fullest cache), per
// research.md Decision 5. SDPA.cpp's is_gemv gate means the coopmat toggle
// has no effect at decode -- confirmed structurally identical to
// QuantizedLinear.cpp's is_gemv_case (linear bench's decode handling).
struct RegimeShape {
  const char* regime;
  int64_t seq_len; // this step's query / newly-written KV length
  int64_t context_len; // KV cache buffer size
  int64_t input_pos; // symint value
};
const std::vector<RegimeShape> kRegimes = {
    {"prefill", 2048, 2048, 0},
    {"decode", 1, 3072, 3071},
};

struct RunResult {
  float mean_us; // total (qk + av)
  float stdev_us;
  float qk_mean_us;
  float qk_stdev_us;
  float av_mean_us;
  float av_stdev_us;
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

// Runs one (model, coopmat-toggle, regime) case: builds the graph once, runs
// kWarmupRuns + kTimedRuns, and returns the SDPA-compute-only GPU time split
// into qk (sdpa_compute_attn_weights_*) and av (sdpa_compute_out_*), plus
// their combined total -- excluding the kv-cache-update and softmax
// dispatches in between (unaccelerated, identical regardless of the coopmat
// toggle or regime).
//
// specs/021: the KV cache buffer is always sized to `regime.context_len`
// (3072 for decode, matching this workstream's ctx3072 PTEs), separate from
// `regime.seq_len` (this step's query/newly-written-KV length -- 1 for
// decode). The cache is filled with random data at full context_len BEFORE
// update_cache writes this step's new K/V at input_pos, so the tensor shapes
// and access pattern are real even though the "history" isn't a genuine
// step-by-step prefill walk (research.md Decision 5 -- only timing is
// measured here, not output correctness).
RunResult
run_case(const ModelShape& m, bool enable_coopmat, const RegimeShape& regime) {
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
      batch_size, regime.seq_len, m.num_heads, m.head_dim};
  const std::vector<int64_t> new_kv_sizes = {
      batch_size, regime.seq_len, m.num_kv_heads, m.head_dim};
  const std::vector<int64_t> cache_sizes = {
      batch_size, regime.context_len, m.num_kv_heads, m.head_dim};

  IOValueRef r_q =
      graph.add_input_tensor(q_sizes, vkapi::kHalf, utils::kBuffer);
  IOValueRef r_k =
      graph.add_input_tensor(new_kv_sizes, vkapi::kHalf, utils::kBuffer);
  IOValueRef r_v =
      graph.add_input_tensor(new_kv_sizes, vkapi::kHalf, utils::kBuffer);

  const ValueRef r_input_pos_symint = graph.add_symint(regime.input_pos);
  const ValueRef r_out =
      graph.add_tensor(q_sizes, vkapi::kHalf, utils::kBuffer);

  const ValueRef r_k_cache =
      graph.add_tensor(cache_sizes, vkapi::kHalf, utils::kBuffer);
  const ValueRef r_v_cache =
      graph.add_tensor(cache_sizes, vkapi::kHalf, utils::kBuffer);
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
      graph,
      r_q.staging,
      batch_size * regime.seq_len * m.num_heads * m.head_dim);
  fill_random(
      graph,
      r_k.staging,
      batch_size * regime.seq_len * m.num_kv_heads * m.head_dim);
  fill_random(
      graph,
      r_v.staging,
      batch_size * regime.seq_len * m.num_kv_heads * m.head_dim);
  // Note: r_k_cache/r_v_cache are plain add_tensor() outputs (not
  // IOValueRef), so there is no staging buffer to pre-fill positions
  // 0..input_pos-1 with. That's fine per research.md Decision 5 -- only
  // timing is measured here, and the attention shaders' dispatch
  // size/access pattern depends solely on the cache's shape (context_len),
  // not its contents, so leaving the pre-update_cache portion at whatever
  // the GPU allocator returns (typically zeroed) does not affect the
  // measurement's validity.

  for (int i = 0; i < kWarmupRuns; ++i) {
    graph.execute();
  }

  std::vector<float> total_timings_us;
  std::vector<float> qk_timings_us;
  std::vector<float> av_timings_us;
  std::vector<std::string> last_dispatched;
  for (int i = 0; i < kTimedRuns; ++i) {
    graph.execute();
    graph.context()->querypool().extract_results();
    const auto shader_results =
        graph.context()->querypool().get_shader_timestamp_data();

    float qk_time_us = 0.0f;
    float av_time_us = 0.0f;
    last_dispatched.clear();
    for (const auto& r : shader_results) {
      last_dispatched.push_back(r.kernel_name);
      const uint64_t duration_ns = r.end_time_ns - r.start_time_ns;
      if (r.kernel_name.find("sdpa_compute_attn_weights") !=
          std::string::npos) {
        qk_time_us += static_cast<float>(duration_ns) / 1000.0f;
      } else if (r.kernel_name.find("sdpa_compute_out") != std::string::npos) {
        av_time_us += static_cast<float>(duration_ns) / 1000.0f;
      }
    }
    qk_timings_us.push_back(qk_time_us);
    av_timings_us.push_back(av_time_us);
    total_timings_us.push_back(qk_time_us + av_time_us);
  }

  RunResult result;
  result.mean_us = mean_of(total_timings_us);
  result.stdev_us = stdev_of(total_timings_us, result.mean_us);
  result.qk_mean_us = mean_of(qk_timings_us);
  result.qk_stdev_us = stdev_of(qk_timings_us, result.qk_mean_us);
  result.av_mean_us = mean_of(av_timings_us);
  result.av_stdev_us = stdev_of(av_timings_us, result.av_mean_us);
  result.dispatched_kernels = last_dispatched;
  return result;
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

// specs/021 (research.md Decision 1): shared unified RESULT,... line. SDPA
// has two axes the shared linear/baseline schema doesn't need to
// distinguish separately: variant (qk/av/total sub-shader, FR-007) and
// toggle (tiled/coopmat -- which ET_VK_SDPA_COOPMAT setting produced this
// row). K/N carry head_dim/num_heads (gflops has no SDPA meaning, -1
// sentinel); num_kv_heads and toggle are appended after the shared 12
// fields, since they're SDPA-specific and every other harness's row has
// exactly 12 fields.
void print_result_line(
    const ModelShape& m,
    const char* regime,
    const std::string& variant,
    const std::string& toggle,
    float avg_us,
    float stdev_us,
    const std::string& dispatch_status) {
  std::cout << "RESULT,sdpa," << m.name << ",," << regime << "," << variant
            << "," << m.head_dim << "," << m.num_heads << "," << avg_us << ","
            << stdev_us << ",-1," << dispatch_status << ",SKIPPED,"
            << m.num_kv_heads << "," << toggle << "\n";
}

int main() {
  std::srand(0);

  std::cout << "SDPA coopmat microbenchmark (real prefill(S=2048)/decode(S=1) "
               "regimes, "
            << kWarmupRuns << " warmup + " << kTimedRuns
            << " timed runs per case)\n";

  bool any_failure = false;
  for (const auto& m : kModels) {
    for (const auto& regime : kRegimes) {
      const bool is_decode = std::string(regime.regime) == "decode";
      RunResult tiled = run_case(m, /*enable_coopmat=*/false, regime);
      // Decode: SDPA.cpp's is_gemv gate dispatches the same "_coop" kernel
      // regardless of ET_VK_SDPA_COOPMAT -- running a second, "coopmat"
      // invocation would just remeasure the identical dispatch. Skip it and
      // reuse tiled's numbers (research.md Decision 2/5).
      RunResult coopmat =
          is_decode ? tiled : run_case(m, /*enable_coopmat=*/true, regime);

      std::string dispatch_status;
      if (is_decode) {
        dispatch_status = "not_applicable";
      } else {
        const bool tiled_is_tiled =
            !has_kernel_containing(tiled.dispatched_kernels, "_coopmat");
        const bool qk_coopmat = has_kernel_containing(
            coopmat.dispatched_kernels, "sdpa_compute_attn_weights_coopmat");
        const bool av_coopmat = has_kernel_containing(
            coopmat.dispatched_kernels, "sdpa_compute_out_coopmat");
        dispatch_status = (tiled_is_tiled && qk_coopmat && av_coopmat)
            ? "confirmed"
            : "fallback_tiled";
        any_failure = any_failure || dispatch_status != "confirmed";
      }

      print_result_line(
          m,
          regime.regime,
          "qk",
          "tiled",
          tiled.qk_mean_us,
          tiled.qk_stdev_us,
          dispatch_status);
      print_result_line(
          m,
          regime.regime,
          "av",
          "tiled",
          tiled.av_mean_us,
          tiled.av_stdev_us,
          dispatch_status);
      print_result_line(
          m,
          regime.regime,
          "total",
          "tiled",
          tiled.mean_us,
          tiled.stdev_us,
          dispatch_status);
      if (!is_decode) {
        print_result_line(
            m,
            regime.regime,
            "qk",
            "coopmat",
            coopmat.qk_mean_us,
            coopmat.qk_stdev_us,
            dispatch_status);
        print_result_line(
            m,
            regime.regime,
            "av",
            "coopmat",
            coopmat.av_mean_us,
            coopmat.av_stdev_us,
            dispatch_status);
        print_result_line(
            m,
            regime.regime,
            "total",
            "coopmat",
            coopmat.mean_us,
            coopmat.stdev_us,
            dispatch_status);
      }
    }
  }

  if (any_failure) {
    std::cout << "\nOne or more prefill cases did not confirm coopmat "
                 "dispatch -- see FR-006/FR-007, do not trust their speedup "
                 "number.\n";
  }
  return 0;
}
