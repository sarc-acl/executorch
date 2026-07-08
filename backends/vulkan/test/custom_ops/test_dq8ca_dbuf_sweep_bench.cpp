// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// specs/023-8da4w-int8-dbuf-sweep driver (Foundational T008): times the int8
// `dq8ca_q4gsw` coopmat linear op (`et_vk.linear_dq8ca_q4gsw.default`)
// across the 6-shape curated catalog (`wq` + `w1_gate` for each of 1B/3B/8B,
// per spec Clarifications), at the prefill regime M=2048. One process per
// `ET_VK_DQ8CA_COOPMAT_VARIANT` value drives the dbuf variant under test
// (research.md Decision 2/3) -- this binary itself is variant-agnostic, the
// env var is read inside QuantizedLinear.cpp's existing dispatch selection.
//
// A sibling of the shared test_coopmat_linear_bench.cpp (8B-only,
// group_size=128, kM=1024) rather than a mutation of it: this workstream's
// specs/007/016 own historical numbers are tied to that file's exact
// shapes/kM/group, so its constants stay untouched. This file duplicates
// the minimal subset of its TestCase-building / correctness-reference logic
// needed for the dq8ca_q4gsw op only, mirroring specs/008's own precedent
// of a dedicated sweep-driver file rather than parameterizing a shared one.
//
// Modes (env vars):
//   ET_VK_DQ8CA_COOPMAT_VARIANT=dbuf{1,2,3,4}  -- which loop-structure
//       variant to dispatch (read by QuantizedLinear.cpp, not this file).
//   DQ8CA_DBUF_SWEEP_CORRECTNESS_ONLY=1  -- run only the small aligned
//       correctness case (skips the 6-shape perf sweep), mirroring
//       COOPMAT_BENCH_CORRECTNESS_ONLY in test_coopmat_linear_bench.cpp.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "utils.h"

using namespace executorch::vulkan::prototyping;
using namespace vkcompute;

namespace {

struct ShapeConfig {
  const char* label; // "1b_wq", "1b_w1_gate", ...
  int64_t k;
  int64_t n;
  int64_t group_size;
  int64_t m;
};

constexpr int64_t kM = 2048; // prefill regime, matches this workstream's
                             // other coopmat linear microbenches.

// wq + w1_gate for each of 1B/3B/8B (spec Clarifications' curated 6-shape
// set), mirroring the model catalog already used by
// test_dq8ca_tile_sweep.cpp / specs/001's shapes.json. All three models use
// group_size=32, at the prefill M above.
const std::vector<ShapeConfig> kShapes = {
    {"1b_wq", 2048, 2048, 32, kM},
    {"1b_w1_gate", 2048, 8192, 32, kM},
    {"3b_wq", 3072, 3072, 32, kM},
    {"3b_w1_gate", 3072, 8192, 32, kM},
    {"8b_wq", 4096, 4096, 32, kM},
    {"8b_w1_gate", 4096, 14336, 32, kM},
};

constexpr const char* kOpName = "linear_dq8ca_q4gsw";

TestCase make_case(const ShapeConfig& cfg, utils::StorageType storage) {
  const vkapi::ScalarType dt = vkapi::kHalf;
  TestCase tc;
  const std::string storage_str =
      (storage == utils::kTexture3D) ? "Texture3D" : "Buffer";
  tc.set_name(
      std::string(kOpName) + "_" + cfg.label + "_M" + std::to_string(cfg.m) +
      "_K" + std::to_string(cfg.k) + "_N" + std::to_string(cfg.n) + "_" +
      storage_str);
  tc.set_operator_name(std::string("et_vk.") + kOpName + ".default");

  ValueSpec input(
      {cfg.m, cfg.k}, dt, storage, utils::kWidthPacked, DataGenType::RANDINT);

  ValueSpec input_scale(
      {1, cfg.m}, dt, storage, utils::kWidthPacked, DataGenType::RANDOM_SCALES);
  input_scale.set_constant(true);
  ValueSpec input_zp(
      {1, cfg.m},
      vkapi::kChar,
      storage,
      utils::kWidthPacked,
      DataGenType::RANDINT);
  input_zp.set_constant(true);

  ValueSpec qweight(
      {cfg.n, cfg.k / 2},
      vkapi::kByte,
      storage,
      utils::kWidthPacked,
      DataGenType::RANDINT4);
  qweight.set_constant(true);
  qweight.set_int4(true);

  const std::vector<int64_t> scales_size = {cfg.k / cfg.group_size, cfg.n};
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
      weight_sums, qweight, cfg.k / cfg.group_size, cfg.n, cfg.group_size);

  ValueSpec group_size_spec(static_cast<int32_t>(cfg.group_size));

  ValueSpec bias({cfg.n}, dt, storage, utils::kWidthPacked, DataGenType::ZEROS);
  bias.set_constant(true);
  bias.set_none(true);

  ValueSpec output(
      {cfg.m, cfg.n}, dt, storage, utils::kWidthPacked, DataGenType::ZEROS);

  tc.add_input_spec(input);
  tc.add_input_spec(input_scale);
  tc.add_input_spec(input_zp);
  tc.add_input_spec(qweight);
  tc.add_input_spec(weight_sums);
  tc.add_input_spec(weight_scales);
  tc.add_input_spec(group_size_spec);
  tc.add_input_spec(bias);
  tc.add_output_spec(output);
  return tc;
}

// Small, tile-aligned (M/N%64==0, K%32==0, group_size%32==0) correctness
// case -- coopmat-eligible, cheap enough for an exact fp32 CPU reference.
// Well-conditioned data (no fp16 cancellation), mirroring
// test_coopmat_linear_bench.cpp's own correctness-case recipe: activations
// are multiples of 1/16 in [0.5, 1.375], int4 nibbles in {9..14} (weight
// +1..+6), per-row activation scale fixed at 1/16 with zp=0 so the dynamic
// int8 quant round-trip is exact in both fp16 and fp32.
TestCase make_correctness_case(utils::StorageType storage) {
  const ShapeConfig cfg{"correctness", 128, 128, 32, /*m=*/128};
  TestCase t = make_case(cfg, storage);

  auto& hin = t.inputs()[0].get_half_data();
  for (size_t i = 0; i < hin.size(); ++i) {
    hin[i] = float_to_half(0.5f + float(i % 14) / 16.0f);
  }
  auto& hsc = t.inputs()[1].get_half_data();
  for (size_t i = 0; i < hsc.size(); ++i) {
    hsc[i] = float_to_half(1.0f / 16.0f);
  }
  auto& izp = t.inputs()[2].get_int8_data();
  for (size_t i = 0; i < izp.size(); ++i) {
    izp[i] = 0;
  }
  auto& wq = t.inputs()[3].get_uint8_data();
  static const uint8_t kPos[6] = {9, 10, 11, 12, 13, 14};
  for (size_t i = 0; i < wq.size(); ++i) {
    wq[i] = uint8_t(kPos[i % 6] | (kPos[(i + 1) % 6] << 4));
  }
  compute_weight_sums_4bit_grouped(
      t.inputs()[4],
      t.inputs()[3],
      cfg.k / cfg.group_size,
      cfg.n,
      cfg.group_size);
  t.set_abs_tolerance(0.5f);
  t.set_rel_tolerance(0.05f);
  return t;
}

std::vector<float> as_f(const ValueSpec& s) {
  const auto& h = s.get_half_data();
  std::vector<float> o(h.size());
  for (size_t i = 0; i < h.size(); ++i) {
    o[i] = half_to_float(h[i]);
  }
  return o;
}

// Exact fp32 CPU reference; throws (-> SKIPPED) for the full-size perf
// shapes, matching test_coopmat_linear_bench.cpp's own cap rationale (an
// O(M*N*K) CPU reference at production K/N would take far too long and
// isn't the point of a perf case).
void bench_reference(TestCase& tc) {
  const ValueSpec& in = tc.inputs()[0];
  ValueSpec& out = tc.outputs()[0];
  const auto is = in.get_tensor_sizes();
  const int64_t M = is[is.size() - 2], K = is[is.size() - 1];
  const int64_t N = out.get_tensor_sizes().back();
  if (M > 256 || N > 256 || K > 4096) {
    throw std::invalid_argument("ref: too big");
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
        float q = std::round(inf[m * K + k] / s_in) + float(zp);
        q = std::min(std::max(q, -128.0f), 127.0f);
        const float a = q - float(zp);
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

std::vector<TestCase> generate_cases() {
  std::vector<TestCase> cases;
  const bool correctness_only =
      std::getenv("DQ8CA_DBUF_SWEEP_CORRECTNESS_ONLY") != nullptr;
  cases.push_back(make_correctness_case(utils::kBuffer));
  if (!correctness_only) {
    for (const auto& cfg : kShapes) {
      cases.push_back(make_case(cfg, utils::kBuffer));
    }
  }
  return cases;
}

FlopCalculatorFunc flop_calc = [](const TestCase& tc) -> int64_t {
  const auto& is = tc.inputs()[0].get_tensor_sizes();
  const int64_t M = is[is.size() - 2], K = is[is.size() - 1];
  const int64_t N = tc.outputs()[0].get_tensor_sizes().back();
  return 2 * M * N * K;
};

} // namespace

int main() {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "specs/023 dq8ca_q4gsw dbuf-variant sweep bench (M=" << kM
            << ", variant="
            << (std::getenv("ET_VK_DQ8CA_COOPMAT_VARIANT")
                    ? std::getenv("ET_VK_DQ8CA_COOPMAT_VARIANT")
                    : "<default/dbuf4>")
            << ")" << std::endl;
  print_separator();

  auto results = execute_test_cases(
      generate_cases,
      flop_calc,
      "Dq8caDbufSweepBench",
      /*warmup=*/3,
      /*runs=*/3,
      /*reference=*/bench_reference);

  // First case is always the correctness case (see generate_cases above).
  const BenchmarkResult& correctness = results[0];
  std::string dispatched = correctness.get_kernel_name();
  for (const auto& st : correctness.get_shader_timings()) {
    if (st.shader_name.find("linear_") != std::string::npos) {
      dispatched = st.shader_name;
    }
  }
  const bool fired = dispatched.find("coopmat") != std::string::npos;
  const bool correctness_ok = fired &&
      correctness.get_correctness_status() == CorrectnessStatus::PASSED;
  std::cout << "[correctness] " << correctness.get_kernel_name() << " -> "
            << dispatched
            << (fired ? " (coopmat dispatched)" : " (NOT coopmat -- fallback)")
            << ", correctness="
            << (correctness.get_correctness_status() ==
                        CorrectnessStatus::PASSED
                    ? "PASSED"
                    : (correctness.get_correctness_status() ==
                               CorrectnessStatus::FAILED
                           ? "FAILED"
                           : "SKIPPED"))
            << "\n";
  if (!correctness_ok) {
    std::cout << "[correctness] FAILED -- did not dispatch coopmat and/or "
                 "failed correctness\n";
    return 1;
  }
  if (results.size() < 1 + kShapes.size()) {
    return 0; // correctness-only run: no perf cases to summarize
  }

  std::cout << "\n================ SUMMARY: dq8ca_q4gsw dbuf sweep "
               "(GFLOP/s) ================\n";
  std::cout << std::left << std::setw(16) << "shape" << std::setw(15) << "(K,N)"
            << std::right << std::setw(12) << "avg_us" << std::setw(10)
            << "cov_pct" << std::setw(12) << "GFLOP/s"
            << "  dispatched kernel\n";
  for (size_t i = 0; i < kShapes.size(); ++i) {
    const BenchmarkResult& r = results[1 + i];
    std::string kernel = r.get_kernel_name();
    for (const auto& st : r.get_shader_timings()) {
      if (st.shader_name.find("linear_") != std::string::npos) {
        kernel = st.shader_name;
      }
    }
    const float us = r.get_avg_time_us();
    const float std_us = r.get_std_dev_us();
    const float cov_pct = us > 0 ? 100.0f * std_us / us : 0.0f;
    const float gflops =
        us > 0 ? (2.0f * kM * kShapes[i].n * kShapes[i].k) / (us * 1e3f) : 0.0f;
    const bool this_fired = kernel.find("coopmat") != std::string::npos;
    std::cout << std::left << std::setw(16) << kShapes[i].label << std::setw(15)
              << ("(" + std::to_string(kShapes[i].k) + "," +
                  std::to_string(kShapes[i].n) + ")")
              << std::right << std::setw(12) << std::fixed
              << std::setprecision(1) << us << std::setw(9)
              << std::setprecision(2) << cov_pct << "%" << std::setw(12)
              << std::setprecision(1) << gflops << "  " << kernel
              << (this_fired ? "" : " (NOT coopmat!)") << "\n";
  }
  return 0;
}
