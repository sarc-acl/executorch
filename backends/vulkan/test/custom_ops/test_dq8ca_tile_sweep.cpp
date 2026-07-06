// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// specs/008-8da4w-parameter-sweep driver. Forces one of 12 tile-shape/
// subgroup-size variants of the 8da4w (linear_dq8ca_q4gsw) coopmat shader
// via the test-only op `test_etvk.dq8ca_tile_sweep.default`
// (impl/TestDq8caTileSweep.cpp), bypassing the production eligibility
// gate entirely. Config 0 (the shipped baseline) is not run here -- its
// numbers are reused from 007's already-captured data.
//
// One config_id per process invocation (env var DQ8CA_SWEEP_CONFIG_ID,
// required, 1-12) -- NOT all 12 in one run. The harness framework
// (utils.cpp's execute_test_cases) only catches
// vkapi::ShaderNotSupportedError; a genuine pipeline-creation crash (the
// exact Xclipse PAL failure mode dq8ca_q4gsw_coopmat_sweep.glsl's header
// documents) cannot be caught by any in-process try/catch. Process-level
// isolation is what actually guarantees one bad config can't erase every
// other config's rows (research.md Decision 2's second correction). The
// shell loop that drives this across all 12 configs, synthesizing a
// pipeline_crash row for any invocation that doesn't exit 0, lives in
// quickstart.md step 3.
//
// DQ8CA_SWEEP_FULL_CATALOG=1 switches from the 6-shape sweep-phase set
// (wq + w1_gate per model, research.md Decision 3) to the full 3-model x
// 7-op catalog (matching 007's exact shapes) -- used only for the
// winning config(s) in US3 (T019).

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "utils.h"

using namespace executorch::vulkan::prototyping;
using namespace vkcompute;

namespace {

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

// Mirrors specs/001-minipc-baseline-benchmarks/results/shapes.json and
// test_llama_baseline_bench.cpp's kModels (lm_head excluded, matching
// 007's Excluded/Out-of-Scope note: lm_head's M=2048 case has no
// production analogue).
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

constexpr int64_t kM = 2048; // prefill regime, matching 004/007.

struct SweepCase {
  int32_t config_id;
  std::string model;
  std::string op;
  int64_t m, k, n;
  int64_t group_size;
  bool is_correctness; // true: small M, exact-data, reference computed.
                       // false: full M, perf-only, reference skipped.
  int32_t ref_key_salt; // unique per case; see encode_config_id_arg below.
};

// execute_test_cases() groups cases by ReferenceKey, which serializes
// every input's (sizes, dtype, data_gen_type, is_constant, is_none) plus
// scalar values -- it does NOT know about op identity. Full-catalog ops
// that happen to share an (M,K,N) shape (wq/wo, wk/wv, w1_gate/w3_up --
// every model has these pairs) therefore land in the SAME reference-cache
// group even though they have different weight data; whichever case is
// first in that group becomes the "prototype" whose reference is computed
// and copied to the rest, silently skipping the others' own correctness
// check (discovered while running T019 -- the framework's grouping is
// itself correct, generic caching behavior, just not one this per-op
// harness can rely on). Fix: fold a per-case salt into the scalar
// `config_id` argument so no two cases ever share a ReferenceKey; decoded
// back out in TestDq8caTileSweep.cpp's op function.
int32_t encode_config_id_arg(int32_t config_id, int32_t salt) {
  return config_id * 10000 + salt;
}

// M for the correctness-check case. Must be a multiple of every config's
// WG_TILE_M (max 256, config 8/9) so M % WG_TILE_M == 0 holds for all 12
// (the shader's own documented hard precondition) -- 256 satisfies every
// config in research.md Decision 4's table.
constexpr int64_t kCorrectnessM = 256;

std::vector<SweepCase> build_case_list(int32_t config_id, bool full_catalog) {
  std::vector<SweepCase> out;
  int32_t salt = 0;
  for (const auto& model : kModels) {
    for (const auto& op : model.ops) {
      const bool is_representative =
          std::string(op.op) == "wq" || std::string(op.op) == "w1_gate";
      if (!full_catalog && !is_representative) {
        continue;
      }
      out.push_back(
          {config_id,
           model.model,
           op.op,
           kCorrectnessM,
           op.k,
           op.n,
           model.group_size,
           /*is_correctness=*/true,
           salt++});
      // Config 12 is a deliberate negative test (research.md Decision 4):
      // only the correctness-scale case is meaningful -- a timing number
      // for a known-broken kernel is not (research.md Decision 4).
      if (config_id != 12) {
        out.push_back(
            {config_id,
             model.model,
             op.op,
             kM,
             op.k,
             op.n,
             model.group_size,
             /*is_correctness=*/false,
             salt++});
      }
      if (config_id == 12 && !full_catalog) {
        return out;
      }
    }
  }
  return out;
}

TestCase make_case(const SweepCase& c) {
  const vkapi::ScalarType dt = vkapi::kHalf;
  const utils::StorageType storage = utils::kBuffer;
  TestCase tc;
  tc.set_name(
      "cfg" + std::to_string(c.config_id) + "_" + c.model + "_" + c.op + "_M" +
      std::to_string(c.m) + "_K" + std::to_string(c.k) + "_N" +
      std::to_string(c.n));
  tc.set_operator_name("test_etvk.dq8ca_tile_sweep.default");

  ValueSpec input(
      {c.m, c.k}, dt, storage, utils::kWidthPacked, DataGenType::RANDINT);

  // Forced scale=1/16, zp=0 (research.md Decision 5): the fp16 dynamic
  // int8 quantization round-trip is exact, so the fp32 CPU reference is
  // valid at any K (int32 GEMM accumulation, no fp rounding until the
  // single final dequantize -- research.md Decision 5's U2 addendum).
  ValueSpec input_scale(
      {1, c.m}, dt, storage, utils::kWidthPacked, DataGenType::RANDOM_SCALES);
  input_scale.set_constant(true);
  ValueSpec input_zp(
      {1, c.m},
      vkapi::kChar,
      storage,
      utils::kWidthPacked,
      DataGenType::RANDINT);
  input_zp.set_constant(true);

  ValueSpec qweight(
      {c.n, c.k / 2},
      vkapi::kByte,
      storage,
      utils::kWidthPacked,
      DataGenType::RANDINT4);
  qweight.set_constant(true);
  qweight.set_int4(true);

  const int64_t num_groups = c.k / c.group_size;
  const std::vector<int64_t> scales_size = {num_groups, c.n};
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
      weight_sums, qweight, num_groups, c.n, c.group_size);

  ValueSpec group_size_spec(static_cast<int32_t>(c.group_size));

  ValueSpec bias({c.n}, dt, storage, utils::kWidthPacked, DataGenType::ZEROS);
  bias.set_constant(true);
  bias.set_none(true);

  ValueSpec config_id_spec(encode_config_id_arg(c.config_id, c.ref_key_salt));

  ValueSpec output(
      {c.m, c.n}, dt, storage, utils::kWidthPacked, DataGenType::ZEROS);

  tc.add_input_spec(input);
  tc.add_input_spec(input_scale);
  tc.add_input_spec(input_zp);
  tc.add_input_spec(qweight);
  tc.add_input_spec(weight_sums);
  tc.add_input_spec(weight_scales);
  tc.add_input_spec(group_size_spec);
  tc.add_input_spec(bias);
  tc.add_input_spec(config_id_spec);
  tc.add_output_spec(output);

  // Structural-bug tolerance, matching test_coopmat_linear_bench.cpp's
  // dq8ca correctness cases: forced-exact quantization data (scale=1/16,
  // zp=0) makes fp32-vs-fp16 divergence the only source of error, bounded
  // by K (more terms summed before the single final dequantize rounding).
  const float k_scaled_abs = 0.1f * std::sqrt(static_cast<float>(c.k));
  tc.set_abs_tolerance(std::max(1.0f, k_scaled_abs));
  tc.set_rel_tolerance(0.05f);
  return tc;
}

std::vector<float> as_f(const ValueSpec& s) {
  if (s.dtype == vkapi::kFloat) {
    return s.get_float_data();
  }
  const auto& h = s.get_half_data();
  std::vector<float> o(h.size());
  for (size_t i = 0; i < h.size(); ++i) {
    o[i] = half_to_float(h[i]);
  }
  return o;
}

// Exact fp32 reference for linear_dq8ca_q4gsw, matching
// test_coopmat_linear_bench.cpp's bench_reference dq8ca branch, but WITHOUT
// its K/N size cap: this feature's correctness cases run at the model's
// real K/N (a config's tile-shape correctness genuinely depends on K/group
// divisibility and N-tile stride math at production width, not just at toy
// sizes) with only M reduced (kCorrectnessM=256). Skips (via exception,
// caught by the framework as ref-not-computed -> SKIPPED) only the
// full-M=2048 performance-only companion case, whose M*K*N would make an
// O(M*K*N) CPU reference impractically slow -- verified separately by its
// paired correctness case at the same K/N (research.md Decision 5's U2
// addendum: int32 GEMM accumulation makes exactness M-invariant, so a
// smaller M validates the same K/group_size/N-tile logic without needing
// the full M).
void sweep_reference(TestCase& tc) {
  const ValueSpec& in = tc.inputs()[0];
  ValueSpec& out = tc.outputs()[0];
  const auto is = in.get_tensor_sizes();
  const int64_t M = is[0], K = is[1];
  const int64_t N = out.get_tensor_sizes()[1];
  if (M > kCorrectnessM) {
    throw std::invalid_argument("ref: perf-only case, skip");
  }

  const ValueSpec& w = tc.inputs()[3];
  const ValueSpec& sc = tc.inputs()[5];
  const int64_t group = tc.inputs()[6].get_int_value();
  const ValueSpec& bias = tc.inputs()[7];
  const bool has_bias = !bias.is_none();

  const std::vector<float> inf = as_f(in);
  const std::vector<float> scf = as_f(sc);
  const std::vector<float> bf = has_bias ? as_f(bias) : std::vector<float>();
  const std::vector<float> in_scale = as_f(tc.inputs()[1]);
  const std::vector<int8_t>& in_zp = tc.inputs()[2].get_int8_data();
  const std::vector<uint8_t>& w4 = w.get_uint8_data();

  auto& ref = out.get_ref_float_data();
  ref.resize(M * N);
  for (int64_t m = 0; m < M; ++m) {
    const float s_in = in_scale[m];
    const int zp = int(in_zp[m]);
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        float a = inf[m * K + k];
        float q = std::round(a / s_in) + float(zp);
        q = std::min(std::max(q, -128.0f), 127.0f);
        a = q - float(zp);
        const uint8_t byte = w4[n * (K / 2) + k / 2];
        const int nib = (k & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
        const int wv = nib - 8;
        const float w_scale = scf[(k / group) * N + n];
        acc += a * float(wv) * w_scale;
      }
      float r = acc * s_in;
      if (has_bias) {
        r += bf[n];
      }
      ref[m * N + n] = r;
    }
  }
}

// Force well-conditioned, EXACTLY-round-trippable quantization data
// (research.md Decision 5): activations multiples of 1/16 in [0.5,1.375],
// int4 nibbles in {9..14} (weight +1..+6), scale forced to 1/16, zp=0.
void force_exact_data(TestCase& tc) {
  auto& hin = tc.inputs()[0].get_half_data();
  for (size_t i = 0; i < hin.size(); ++i) {
    hin[i] = float_to_half(0.5f + 0.125f * float(i % 8));
  }
  auto& hs = tc.inputs()[1].get_half_data();
  std::fill(hs.begin(), hs.end(), float_to_half(0.0625f));
  auto& zp = tc.inputs()[2].get_int8_data();
  std::fill(zp.begin(), zp.end(), int8_t(0));
  auto& wq = tc.inputs()[3].get_uint8_data();
  const uint8_t kPos[6] = {0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE};
  for (size_t i = 0; i < wq.size(); ++i) {
    wq[i] = kPos[i % 6];
  }
}

std::vector<TestCase> generate_cases(int32_t config_id, bool full_catalog) {
  std::vector<TestCase> tcs;
  for (const auto& c : build_case_list(config_id, full_catalog)) {
    TestCase tc = make_case(c);
    if (c.is_correctness) {
      force_exact_data(tc);
      // Recompute weight sums against the now-overwritten weight bytes.
      const int64_t num_groups = c.k / c.group_size;
      compute_weight_sums_4bit_grouped(
          tc.inputs()[4], tc.inputs()[3], num_groups, c.n, c.group_size);
    }
    tcs.push_back(tc);
  }
  return tcs;
}

int64_t flop_calc(const TestCase& tc) {
  const auto& in = tc.inputs()[0].get_tensor_sizes();
  const auto& out = tc.outputs()[0].get_tensor_sizes();
  return 2 * in[0] * out[1] * in[1];
}

std::string outcome_for(const BenchmarkResult& r) {
  const bool vulkan_ok =
      r.get_num_iterations() > 0 && r.get_avg_time_us() > 0.0f;
  if (!vulkan_ok) {
    return "pipeline_crash";
  }
  if (r.get_correctness_status() == CorrectnessStatus::FAILED) {
    return "correctness_failure";
  }
  return "measured";
}

// The dispatched kernel name, picked out of per-shader timings the same
// way test_coopmat_linear_bench.cpp does (the op also dispatches a
// quantize_and_pack shader; we want the dq8ca_q4gsw_coopmat_sweep one).
std::string dispatched_kernel(const BenchmarkResult& r) {
  std::string name = r.get_kernel_name();
  for (const auto& st : r.get_shader_timings()) {
    if (st.shader_name.find("dq8ca_q4gsw_coopmat_sweep") != std::string::npos) {
      name = st.shader_name;
    }
  }
  return name;
}

} // namespace

int main() {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  const char* config_id_env = std::getenv("DQ8CA_SWEEP_CONFIG_ID");
  if (config_id_env == nullptr) {
    std::cerr << "DQ8CA_SWEEP_CONFIG_ID env var is required (1-12)."
              << std::endl;
    return 1;
  }
  const int32_t config_id = std::atoi(config_id_env);
  if (config_id < 1 || config_id > 12) {
    std::cerr << "DQ8CA_SWEEP_CONFIG_ID must be 1-12, got " << config_id
              << std::endl;
    return 1;
  }
  const bool full_catalog = std::getenv("DQ8CA_SWEEP_FULL_CATALOG") != nullptr;

  print_performance_header();
  std::cout << "8da4w tile/subgroup sweep: config_id=" << config_id
            << (full_catalog ? " (full catalog)" : " (sweep phase)")
            << std::endl;
  print_separator();

  std::vector<SweepCase> case_meta = build_case_list(config_id, full_catalog);
  auto gen = [&]() { return generate_cases(config_id, full_catalog); };

  TestResult results = execute_test_cases(
      gen,
      flop_calc,
      "Dq8caTileSweep",
      /*warmup=*/3,
      /*runs=*/5,
      /*reference=*/sweep_reference);

  // Each logical (config, model, op) produced 1 case_meta/results entry
  // (config 12: correctness-only) or 2 (correctness at kCorrectnessM,
  // immediately followed by the full-M performance companion) -- combine
  // each pair into one SWEEP_RESULT row: outcome/correctness from the
  // small-M case (the only one with a reference), timing from the large-M
  // case (the only one that's actually a valid perf number).
  size_t i = 0;
  while (i < results.size() && i < case_meta.size()) {
    const BenchmarkResult& corr_r = results[i];
    const SweepCase& corr_c = case_meta[i];
    const bool has_perf_companion = corr_c.is_correctness &&
        (i + 1) < results.size() && (i + 1) < case_meta.size() &&
        !case_meta[i + 1].is_correctness;

    const bool corr_vulkan_ok =
        corr_r.get_num_iterations() > 0 && corr_r.get_avg_time_us() > 0.0f;
    std::string outcome;
    std::string detail;
    if (!corr_vulkan_ok) {
      outcome = "pipeline_crash";
      detail = "correctness-scale dispatch did not produce a valid timing";
    } else if (corr_r.get_correctness_status() == CorrectnessStatus::FAILED) {
      outcome = "correctness_failure";
      detail = "exact-reference mismatch (see stdout above for values)";
    } else if (corr_r.get_correctness_status() != CorrectnessStatus::PASSED) {
      outcome = "pipeline_crash";
      detail = "correctness reference was not computed (see stdout)";
    }

    const BenchmarkResult* perf_r =
        has_perf_companion ? &results[i + 1] : nullptr;
    const SweepCase& row_c = corr_c; // model/op/k/n identical across pair

    if (outcome.empty()) {
      if (perf_r == nullptr) {
        // Config 12: correctness-only, no timing to report.
        outcome = "measured";
      } else {
        const bool perf_vulkan_ok = perf_r->get_num_iterations() > 0 &&
            perf_r->get_avg_time_us() > 0.0f;
        if (!perf_vulkan_ok) {
          outcome = "pipeline_crash";
          detail = "performance-scale dispatch did not produce a valid timing";
        } else {
          outcome = "measured";
        }
      }
    }

    const std::string kernel = dispatched_kernel(corr_r);
    std::cout << "SWEEP_RESULT," << row_c.config_id << "," << row_c.model << ","
              << row_c.op << "," << kM << "," << row_c.k << "," << row_c.n
              << "," << outcome << ",";
    if (outcome == "measured" && perf_r != nullptr) {
      std::cout << perf_r->get_avg_time_us() << "," << perf_r->get_std_dev_us()
                << "," << perf_r->get_num_iterations();
    } else if (outcome == "measured") {
      // Config 12: no perf companion; nothing to report but the pass.
      std::cout << ",,";
    } else {
      std::cout << ",,";
    }
    std::cout << "," << kernel << "," << detail << std::endl;

    i += has_perf_companion ? 2 : 1;
  }

  return 0;
}
