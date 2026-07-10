// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// MiniPC no-WMMA baseline microbenchmark (specs/001-minipc-baseline-benchmarks,
// extended by specs/004-linear-storage-comparison): real per-model linear
// shapes for Llama 3.1 8B / 3.2 3B / 3.2 1B, at the 4w and 8da4w int4
// schemes, at the fixed prefill (M=2048) and decode (M=1) regimes, at both
// Texture3D and Buffer output storage. Shapes are read from
// specs/001-minipc-baseline-benchmarks/results/shapes.json (derived from each
// checkpoint's real params.json, per that feature's research.md Decision 5)
// and duplicated here as static data since this binary has no JSON reader.
//
// Dispatch path: every case is run with ET_VK_FORCE_TILED_LINEAR=1 set for
// the whole process -- this is REQUIRED, not a formality. Unlike the real
// exported model (whose linear output is rank-3 and already blocked by that
// alone, per specs/003-wmma-shader-candidates), this harness's tensors are
// plain 2D ({M, K}), and every case's M/N/K already satisfies coopmat's tile
// alignment -- so a Buffer-storage prefill case would otherwise silently
// dispatch the real coopmat shader instead of the intended tiled comparison
// (specs/004-linear-storage-comparison/research.md Decision 2). At the
// decode regime (M=1, a GEMV shape) QuantizedLinear.cpp's dispatch never
// considers the coopmat shader in the first place (is_gemv_case short-
// circuits before the coopmat eligibility check) -- it always dispatches the
// "_coop" software-cooperative shader regardless of storage type or the
// toggle; do not expect a "_tiled" decode kernel name at either storage type.
//
// Output: one "RESULT,<model>,<scheme>,<regime>,<storage>,<op>,M,K,N,
// mean_us,stddev_us,iterations,kernel" line per case, meant to be parsed by a
// small script that merges them into results/raw/<model>_<scheme>.json's
// `microbench` array (001) / storage-comparison-report.md (004).

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "utils.h"

using namespace executorch::vulkan::prototyping;
using namespace vkcompute;

namespace {

struct LinearConfig {
  int64_t M;
  int64_t K;
  int64_t N;
  int64_t group_size;
  std::string op_name; // "linear_q4gsw" | "linear_dq8ca_q4gsw"
  std::string model;
  std::string scheme; // "4w" | "8da4w"
  std::string op; // "wq", "wk", ...
  std::string regime; // "prefill" | "decode"
  std::string storage_name; // "texture3d" | "buffer"
  utils::StorageType storage;
};

bool is_dq8ca(const std::string& op) {
  return op.find("dq8ca") != std::string::npos;
}

// execute_test_cases() internally groups cases by a ReferenceKey that
// explicitly excludes storage_type (it exists to cache/reuse reference-output
// computation across cases that only differ in storage/layout), and returns
// `results` in group-processing order, NOT in generate_cases()'s original
// sequential order. Since texture3d/buffer variants of the same op share a
// ReferenceKey, this reorders them relative to a naive parallel nested loop.
// BenchmarkResult's kernel_name is seeded from TestCase::name() before any
// per-shader override (see execute_test_case in utils.cpp) and survives
// unchanged through to the caller, so it is used here as the one reliable
// way to recover which (model, scheme, regime, storage, op) a given result
// actually corresponds to -- looked up by name rather than assumed by index.
std::unordered_map<std::string, LinearConfig> g_case_configs;

// Same TestCase construction as test_coopmat_linear_bench.cpp's make_case,
// parameterized over storage type (see file header -- ET_VK_FORCE_TILED_LINEAR
// keeps every case, both storage types, on the tiled/coop dispatch).
TestCase make_case(const LinearConfig& cfg) {
  const utils::StorageType storage = cfg.storage;
  const vkapi::ScalarType dt = vkapi::kHalf;
  TestCase tc;
  const std::string case_name = cfg.model + "_" + cfg.scheme + "_" +
      cfg.regime + "_" + cfg.storage_name + "_" + cfg.op + "_M" +
      std::to_string(cfg.M);
  tc.set_name(case_name);
  tc.set_operator_name("et_vk." + cfg.op_name + ".default");
  g_case_configs.emplace(case_name, cfg);

  ValueSpec input(
      {cfg.M, cfg.K}, dt, storage, utils::kWidthPacked, DataGenType::RANDINT);

  ValueSpec input_scale(
      {1, cfg.M}, dt, storage, utils::kWidthPacked, DataGenType::RANDOM_SCALES);
  input_scale.set_constant(true);
  ValueSpec input_zp(
      {1, cfg.M},
      vkapi::kChar,
      storage,
      utils::kWidthPacked,
      DataGenType::RANDINT);
  input_zp.set_constant(true);

  ValueSpec qweight(
      {cfg.N, cfg.K / 2},
      vkapi::kByte,
      storage,
      utils::kWidthPacked,
      DataGenType::RANDINT4);
  qweight.set_constant(true);
  qweight.set_int4(true);

  std::vector<int64_t> scales_size = {cfg.K / cfg.group_size, cfg.N};
  ValueSpec weight_scales(
      scales_size,
      dt,
      storage,
      utils::kWidthPacked,
      DataGenType::RANDOM_SCALES);
  weight_scales.set_constant(true);

  ValueSpec weight_sums(
      scales_size,
      vkapi::kInt,
      storage,
      utils::kWidthPacked,
      DataGenType::ZEROS);
  weight_sums.set_constant(true);
  compute_weight_sums_4bit_grouped(
      weight_sums, qweight, cfg.K / cfg.group_size, cfg.N, cfg.group_size);

  ValueSpec group_size_spec(static_cast<int32_t>(cfg.group_size));

  ValueSpec bias({cfg.N}, dt, storage, utils::kWidthPacked, DataGenType::ZEROS);
  bias.set_constant(true);
  bias.set_none(true);

  ValueSpec output(
      {cfg.M, cfg.N}, dt, storage, utils::kWidthPacked, DataGenType::ZEROS);

  tc.add_input_spec(input);
  if (is_dq8ca(cfg.op_name)) {
    tc.add_input_spec(input_scale);
    tc.add_input_spec(input_zp);
  }
  tc.add_input_spec(qweight);
  if (is_dq8ca(cfg.op_name)) {
    tc.add_input_spec(weight_sums);
  }
  tc.add_input_spec(weight_scales);
  tc.add_input_spec(group_size_spec);
  tc.add_input_spec(bias);
  tc.add_output_spec(output);
  return tc;
}

int64_t flop_calc(const TestCase& tc) {
  const auto& in = tc.inputs()[0].get_tensor_sizes();
  const auto& out = tc.outputs()[0].get_tensor_sizes();
  const int64_t M = in[0], K = in[1], N = out[1];
  return 2 * M * N * K;
}

struct OpShape {
  const char* op;
  int64_t k;
  int64_t n;
};

struct ModelShapes {
  const char* model;
  int64_t group_size;
  std::vector<OpShape> ops;
};

// Mirrors specs/001-minipc-baseline-benchmarks/results/shapes.json.
// specs/021 (research.md Decision 9 follow-up): `lm_head` (K,128256) is
// EXCLUDED per explicit user decision -- it's this suite's single largest
// dispatch (~270us-2.4ms observed, wildly variable) and the direct trigger
// for the QueryPool race (Decision 9) and a suspected GPU/driver reset that
// disconnected the device entirely during a later run. Removing it both
// speeds up the full sweep (no more ~2.4s-per-case outlier dominating wall
// time) and sidesteps the race at its source, rather than continuing to
// paper over it with try/catch on every run.
const std::vector<ModelShapes> kModels = {
    {"llama-3.1-8b",
     32,
     {{"wq", 4096, 4096},
      {"wk", 4096, 1024},
      {"wv", 4096, 1024},
      {"wo", 4096, 4096},
      {"w1_gate", 4096, 14336},
      {"w3_up", 4096, 14336},
      {"w2_down", 14336, 4096}}},
    {"llama-3.2-3b",
     32,
     {{"wq", 3072, 3072},
      {"wk", 3072, 1024},
      {"wv", 3072, 1024},
      {"wo", 3072, 3072},
      {"w1_gate", 3072, 8192},
      {"w3_up", 3072, 8192},
      {"w2_down", 8192, 3072}}},
    {"llama-3.2-1b",
     32,
     {{"wq", 2048, 2048},
      {"wk", 2048, 512},
      {"wv", 2048, 512},
      {"wo", 2048, 2048},
      {"w1_gate", 2048, 8192},
      {"w3_up", 2048, 8192},
      {"w2_down", 8192, 2048}}},
};

const std::vector<std::pair<std::string, std::string>> kSchemes = {
    {"4w", "linear_q4gsw"},
    {"8da4w", "linear_dq8ca_q4gsw"},
};

const std::vector<std::pair<std::string, int64_t>> kRegimes = {
    {"prefill", 2048},
    {"decode", 1},
};

// specs/004-linear-storage-comparison: the storage-type axis this feature
// adds. Every case is generated at both, always under
// ET_VK_FORCE_TILED_LINEAR=1 (see file header) so the comparison isolates
// storage type from any dispatch-algorithm change.
const std::vector<std::pair<std::string, utils::StorageType>> kStorageTypes = {
    {"texture3d", utils::kTexture3D},
    {"buffer", utils::kBuffer},
};

// specs/021 (research.md Decision 3/8): generates one model's 64 cases only
// (2 schemes x 2 regimes x 2 storage x 8 ops), not all 192 -- main() calls
// this once per model as the outer grouping loop, then calls
// execute_test_cases() once per individual case within that model (Decision
// 8), which is what actually bounds peak memory to a single case's own
// tensors (~525MB worst case, a lm_head prefill case) instead of the
// previous ~6.3GB from all 12 lm_head cases materialized at once.
std::vector<TestCase> generate_cases_for_model(const ModelShapes& model) {
  std::vector<TestCase> cases;
  for (const auto& scheme : kSchemes) {
    for (const auto& regime : kRegimes) {
      for (const auto& storage : kStorageTypes) {
        for (const auto& op : model.ops) {
          LinearConfig cfg{
              regime.second,
              op.k,
              op.n,
              model.group_size,
              scheme.second,
              model.model,
              scheme.first,
              op.op,
              regime.first,
              storage.first,
              storage.second};
          cases.push_back(make_case(cfg));
        }
      }
    }
  }
  return cases;
}

// specs/021 (research.md Decision 1/2): shared unified RESULT,... line,
// printed immediately after each case's single execute_test_cases() call
// returns. dispatch_status is always not_applicable -- this harness forces
// ET_VK_FORCE_TILED_LINEAR=1 and has no coopmat toggle at all, so its
// storage-type (texture3d vs buffer) comparison is a structurally different
// question from linear bench's tiled-vs-coopmat one; storage_name is
// appended after the shared 12 fields, since it's this harness's own
// comparison axis, not a fallback/dispatch signal. `variant` instead reports
// which real kernel suffix actually dispatched ("tiled" for prefill,
// "coop" for decode's is_gemv_case short-circuit -- structurally identical
// to linear bench's decode handling; "coopmat" would indicate
// ET_VK_FORCE_TILED_LINEAR somehow didn't take effect, a real anomaly this
// harness has never observed but checks for rather than assumes away).
void print_result_line(const BenchmarkResult& r) {
  const std::string& case_name = r.get_kernel_name();
  auto it = g_case_configs.find(case_name);
  if (it == g_case_configs.end()) {
    std::cerr << "WARNING: no LinearConfig found for result named '"
              << case_name << "', skipping" << std::endl;
    return;
  }
  const LinearConfig& cfg = it->second;
  std::string dispatched_kernel = case_name;
  for (const auto& st : r.get_shader_timings()) {
    if (st.shader_name.find("linear_") != std::string::npos) {
      dispatched_kernel = st.shader_name;
    }
  }
  std::string variant;
  if (dispatched_kernel.find("_coopmat") != std::string::npos) {
    variant = "coopmat"; // should never happen under ET_VK_FORCE_TILED_LINEAR=1
  } else if (dispatched_kernel.find("_coop") != std::string::npos) {
    variant = "coop";
  } else {
    variant = "tiled";
  }
  const std::string correctness_status =
      r.get_correctness_status() == CorrectnessStatus::PASSED   ? "PASSED"
      : r.get_correctness_status() == CorrectnessStatus::FAILED ? "FAILED"
                                                                : "SKIPPED";
  const float avg_us = r.get_avg_time_us();
  const float gflops =
      avg_us > 0 ? (2.0f * cfg.M * cfg.N * cfg.K) / (avg_us * 1e3f) : 0.0f;

  std::cout << "RESULT,baseline," << cfg.model << "," << cfg.scheme << ","
            << cfg.regime << "," << variant << "," << cfg.K << "," << cfg.N
            << "," << avg_us << "," << r.get_std_dev_us() << "," << gflops
            << ",not_applicable," << correctness_status << ","
            << cfg.storage_name << "\n";
}

} // namespace

int main() {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout
      << "MiniPC no-WMMA baseline: Llama 3.1 8B / 3.2 3B / 3.2 1B, 4w/8da4w, "
      << "prefill(M=2048)/decode(M=1), tiled/coop dispatch only" << std::endl;
  print_separator();

  // specs/021: outer loop over models (output grouping only, per
  // research.md Decision 3), inner loop calls execute_test_cases() once per
  // individual case (Decision 8) -- this per-case granularity is what
  // actually fixes the OOM, not the per-model grouping itself.
  for (const auto& model : kModels) {
    std::vector<TestCase> model_cases = generate_cases_for_model(model);
    for (auto& tc : model_cases) {
      try {
        auto single_case_result = execute_test_cases(
            [&tc]() { return std::vector<TestCase>{tc}; },
            flop_calc,
            "LlamaBaselineBench",
            /*warmup=*/3,
            /*runs=*/5,
            /*reference=*/nullptr);
        if (single_case_result.empty()) {
          continue; // defensive; execute_test_cases always returns 1 result
                    // here
        }
        print_result_line(single_case_result[0]);
      } catch (const std::exception& e) {
        // specs/021: a case-local exception (observed: QueryPool.cpp's
        // non-blocking vkGetQueryPoolResults occasionally returning
        // VK_NOT_READY on lm_head's ~270us/dispatch, the largest single
        // dispatch in this suite -- a pre-existing shared-runtime race this
        // feature's per-case execution was the first thing to ever actually
        // reach, since every prior run OOM'd before getting this far; not a
        // regression introduced by this feature, and fixing the race itself
        // is out of scope per FR-011) must not take down the other 191
        // cases -- this extends Decision 8's "partial data survives a
        // failure" principle to a case-local exception, not just a
        // process-level OOM. Recovered from via g_case_configs (populated by
        // make_case() before the exception, so the case's identity is known
        // even though its measurement isn't) and surfaced as its own
        // RESULT,... row (correctness_status=CRASHED) rather than silently
        // skipped, per FR-010.
        std::cerr << "WARNING: case '" << tc.name() << "' threw: " << e.what()
                  << " -- recorded as CRASHED, continuing to next case\n";
        auto cfg_it = g_case_configs.find(tc.name());
        if (cfg_it != g_case_configs.end()) {
          const LinearConfig& cfg = cfg_it->second;
          std::cout << "RESULT,baseline," << cfg.model << "," << cfg.scheme
                    << "," << cfg.regime << ",crashed," << cfg.K << "," << cfg.N
                    << ",-1,-1,-1,not_applicable,CRASHED," << cfg.storage_name
                    << "\n";
        } else {
          std::cout << "RESULT,baseline,unknown,unknown,unknown,crashed,-1,-1,"
                       "-1,-1,-1,not_applicable,CRASHED,unknown\n";
        }
      }
    }
  }
  return 0;
}
