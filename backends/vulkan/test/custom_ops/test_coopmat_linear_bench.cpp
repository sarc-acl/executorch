// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Consolidated coopmat-vs-tiled microbenchmark for the two int4-quantized
// linear types at Llama 3.1 8B prefill shapes:
//   4w    = linear_q4gsw          (weight-only int4)
//   8da4w = linear_dq8ca_q4gsw    (dyn-act int8 x int4 weight)
//
// Baseline (tiled) is selected by Texture3D+Half output storage; coopmat is
// selected by Buffer+Half (the runtime gate in QuantizedLinear.cpp picks the
// _coopmat shader when M%64==0, N%64==0, K%32==0, subgroup==64). Perf-only:
// no CPU reference is run (correctness is covered by the per-op test_*_linear
// benches at small shapes).

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "utils.h"

using namespace executorch::vulkan::prototyping;
using namespace vkcompute;

struct LinearConfig {
  int64_t M;
  int64_t K;
  int64_t N;
  int64_t group_size; // only meaningful for 4-bit
  std::string op_name;
  // 0 = rank-2 input/output ({M,K}/{M,N}, every pre-existing config below).
  // >=1 = rank-3 ({batch,M,K}/{batch,M,N}) -- used to cover the real
  // exported model's rank-3, batch=1 activations (specs/003, never
  // squeezed), which can_use_q4gsw_coopmat() only started admitting to the
  // coopmat path after specs/009's guard relaxation (research.md Decision
  // 1/2). See kRank3CorrectnessShapes below.
  int64_t batch = 0;
};

static bool is_dq8ca(const std::string& op) {
  return op.find("dq8ca") != std::string::npos;
}
static bool is_4bit(const std::string& op) {
  return op.find("q4gsw") != std::string::npos;
}

// Build one test case for the given op at (storage, half dtype), no bias.
static TestCase make_case(const LinearConfig& cfg, utils::StorageType storage) {
  const vkapi::ScalarType dt = vkapi::kHalf;
  TestCase tc;
  const std::string storage_str =
      (storage == utils::kTexture3D) ? "Texture3D" : "Buffer";
  tc.set_name(
      cfg.op_name + "_M" + std::to_string(cfg.M) + "_K" +
      std::to_string(cfg.K) + "_N" + std::to_string(cfg.N) +
      (cfg.batch > 0 ? "_rank3batch" + std::to_string(cfg.batch) : "") + "_" +
      storage_str);
  tc.set_operator_name("et_vk." + cfg.op_name + ".default");

  const std::vector<int64_t> input_sizes = cfg.batch > 0
      ? std::vector<int64_t>{cfg.batch, cfg.M, cfg.K}
      : std::vector<int64_t>{cfg.M, cfg.K};
  ValueSpec input(
      input_sizes, dt, storage, utils::kWidthPacked, DataGenType::RANDINT);

  // dynamic per-row activation scale/zp (dq8ca only)
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

  // weight + scales + sums depend on 4-bit vs 8-bit
  const bool four = is_4bit(cfg.op_name);
  ValueSpec qweight(
      four ? std::vector<int64_t>{cfg.N, cfg.K / 2}
           : std::vector<int64_t>{cfg.N, cfg.K},
      four ? vkapi::kByte : vkapi::kChar,
      storage,
      utils::kWidthPacked,
      four ? DataGenType::RANDINT4 : DataGenType::RANDINT8);
  qweight.set_constant(true);
  if (four) {
    qweight.set_int4(true);
  }

  std::vector<int64_t> scales_size = four
      ? std::vector<int64_t>{cfg.K / cfg.group_size, cfg.N}
      : std::vector<int64_t>{cfg.N};
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
  if (four) {
    compute_weight_sums_4bit_grouped(
        weight_sums, qweight, cfg.K / cfg.group_size, cfg.N, cfg.group_size);
  } else {
    compute_weight_sums(weight_sums, qweight, cfg.N, cfg.K);
  }

  ValueSpec group_size_spec(static_cast<int32_t>(cfg.group_size));

  ValueSpec bias({cfg.N}, dt, storage, utils::kWidthPacked, DataGenType::ZEROS);
  bias.set_constant(true);
  bias.set_none(true);

  const std::vector<int64_t> output_sizes = cfg.batch > 0
      ? std::vector<int64_t>{cfg.batch, cfg.M, cfg.N}
      : std::vector<int64_t>{cfg.M, cfg.N};
  ValueSpec output(
      output_sizes, dt, storage, utils::kWidthPacked, DataGenType::ZEROS);

  // assemble per op signature
  if (cfg.op_name == "linear_q4gsw") {
    tc.add_input_spec(input);
    tc.add_input_spec(qweight);
    tc.add_input_spec(weight_scales);
    tc.add_input_spec(group_size_spec);
    tc.add_input_spec(bias);
  } else if (cfg.op_name == "linear_dq8ca_q4gsw") {
    tc.add_input_spec(input);
    tc.add_input_spec(input_scale);
    tc.add_input_spec(input_zp);
    tc.add_input_spec(qweight);
    tc.add_input_spec(weight_sums);
    tc.add_input_spec(weight_scales);
    tc.add_input_spec(group_size_spec);
    tc.add_input_spec(bias);
  }
  tc.add_output_spec(output);
  return tc;
}

// ---- correctness reference for all four ops; oversized shapes (the perf
// cases) throw -> framework marks them SKIPPED. For dq8ca the activation
// quant round-trip (round(x/scale)+zp) is mirrored in fp32; this is exact
// (not just close) for the correctness data below, which uses scale=1/16,
// zp=0 and activations that are multiples of 1/16, so fp16-vs-fp32
// divergence cannot occur. ----
static std::vector<float> as_f(const ValueSpec& s) {
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
static void bench_reference(TestCase& tc) {
  const std::string op = tc.operator_name();
  const bool dq8ca = op.find("dq8ca") != std::string::npos;
  const bool four = op.find("q4gsw") != std::string::npos;
  const ValueSpec& in = tc.inputs()[0];
  ValueSpec& out = tc.outputs()[0];
  // Rank-agnostic: reads the trailing two dims, so a rank-3 [batch, M, K]
  // input (batch=1, specs/009) is handled identically to plain [M, K] --
  // the reference matmul below only ever needs (M, K, N), never the batch.
  const auto is = in.get_tensor_sizes();
  const int64_t M = is[is.size() - 2], K = is[is.size() - 1];
  const int64_t N = out.get_tensor_sizes().back();
  // M/N stay capped at 256 (the perf-sweep shapes reuse this same function
  // and go up to M=1024/N=14336 -- an O(M*N*K) CPU reference at that size
  // would take far too long and isn't the point of a perf case anyway).
  // K's cap is raised for specs/014-m5-linear-coopmat-retune's FR-008
  // production-K correctness cases (K=2048/4096, M/N still <=256): without
  // this, those cases silently throw here and get marked SKIPPED, giving a
  // false impression of "validated" when no reference was ever computed.
  if (M > 256 || N > 256 || K > 4096) {
    throw std::invalid_argument("ref: too big");
  }
  // input layouts: weight-only = {in, w, w_scales, [group], bias};
  // dq8ca = {in, in_scale, in_zp, w, w_sums, w_scales, [group], bias}
  const ValueSpec& w = tc.inputs()[dq8ca ? 3 : 1];
  const ValueSpec& sc = tc.inputs()[dq8ca ? 5 : 2];
  const int64_t group = four ? tc.inputs()[dq8ca ? 6 : 3].get_int_value() : K;
  const ValueSpec& bias = tc.inputs()[dq8ca ? (four ? 7 : 6) : (four ? 4 : 3)];
  const bool has_bias = !bias.is_none();

  const std::vector<float> inf = as_f(in);
  const std::vector<float> scf = as_f(sc);
  const std::vector<float> bf = has_bias ? as_f(bias) : std::vector<float>();
  const std::vector<float> in_scale =
      dq8ca ? as_f(tc.inputs()[1]) : std::vector<float>();
  const std::vector<int8_t>& in_zp =
      dq8ca ? tc.inputs()[2].get_int8_data() : std::vector<int8_t>();
  const std::vector<uint8_t>& w4 =
      four ? w.get_uint8_data() : std::vector<uint8_t>(); // [N, K/2] nibbles
  const std::vector<int8_t>& w8 =
      four ? std::vector<int8_t>() : w.get_int8_data(); // [N, K]

  auto& ref = out.get_ref_float_data();
  ref.resize(M * N);
  for (int64_t m = 0; m < M; ++m) {
    const float s_in = dq8ca ? in_scale[m] : 1.0f;
    const int zp = dq8ca ? int(in_zp[m]) : 0;
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        float a = inf[m * K + k];
        if (dq8ca) {
          float q = std::round(a / s_in) + float(zp);
          q = std::min(std::max(q, -128.0f), 127.0f);
          a = q - float(zp);
        }
        int wv;
        if (four) {
          const uint8_t byte = w4[n * (K / 2) + k / 2];
          const int nib = (k & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
          wv = nib - 8;
        } else {
          wv = w8[n * K + k];
        }
        const float w_scale = four ? scf[(k / group) * N + n] : scf[n];
        acc += a * float(wv) * w_scale;
      }
      float r = dq8ca ? acc * s_in : acc;
      if (has_bias) {
        r += bf[n];
      }
      ref[m * N + n] = r;
    }
  }
}

// Llama 3.1 8B linear weight shapes (K,N) at prefill M (multiple of 64 so
// coopmat fires).
static const std::vector<std::pair<int64_t, int64_t>> kShapes = {
    {4096, 4096}, // q_proj / o_proj
    {4096, 1024}, // k_proj / v_proj (GQA)
    {4096, 14336}, // gate_proj / up_proj
    {14336, 4096}, // down_proj
};
static const std::vector<std::string> kOps = {
    "linear_q4gsw",
    "linear_dq8ca_q4gsw"};
static constexpr int64_t kM = 1024;
static constexpr int64_t kGroup = 128;

// Builds one deterministic, well-conditioned correctness case (POSITIVE
// data, no fp16 cancellation -- see the comment block above
// kCorrectnessShapes) for the given op/shape/storage. Factored out of the
// kCorrectnessShapes loop so kRank3CorrectnessShapes (below) can reuse the
// exact same data-generation recipe instead of duplicating it.
static TestCase make_deterministic_correctness_case(
    const LinearConfig& cfg,
    const std::string& op,
    utils::StorageType st) {
  const bool dq = is_dq8ca(op);
  const bool four = is_4bit(op);
  TestCase t = make_case(cfg, st);
  auto& hin = t.inputs()[0].get_half_data();
  for (size_t i = 0; i < hin.size(); ++i) {
    hin[i] = float_to_half(0.5f + 0.125f * float(i % 8));
  }
  const size_t w_idx = dq ? 3 : 1;
  if (four) {
    auto& wq = t.inputs()[w_idx].get_uint8_data();
    const uint8_t kPos[6] = {0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE};
    for (size_t i = 0; i < wq.size(); ++i) {
      wq[i] = kPos[i % 6];
    }
  } else {
    auto& wq = t.inputs()[w_idx].get_int8_data();
    for (size_t i = 0; i < wq.size(); ++i) {
      wq[i] = int8_t(1 + (i % 6));
    }
  }
  if (dq) {
    auto& hs = t.inputs()[1].get_half_data();
    std::fill(hs.begin(), hs.end(), float_to_half(0.0625f));
    auto& zp = t.inputs()[2].get_int8_data();
    std::fill(zp.begin(), zp.end(), int8_t(0));
    // weights were overwritten above -> recompute the sums
    if (four) {
      compute_weight_sums_4bit_grouped(
          t.inputs()[4],
          t.inputs()[w_idx],
          cfg.K / cfg.group_size,
          cfg.N,
          cfg.group_size);
    } else {
      compute_weight_sums(t.inputs()[4], t.inputs()[w_idx], cfg.N, cfg.K);
    }
  }
  t.set_abs_tolerance(0.5f);
  t.set_rel_tolerance(0.05f);
  return t;
}

// Generation order: for each op, for each shape -> {Texture3D, Buffer}.
// Summary pairs results [2i]=tiled, [2i+1]=coopmat.
std::vector<TestCase> generate_cases() {
  std::vector<TestCase> cases;
  // COOPMAT_BENCH_CORRECTNESS_ONLY=1 skips the (slow) M=1024 perf cases and
  // runs just the small correctness matrix below.
  const bool correctness_only =
      std::getenv("COOPMAT_BENCH_CORRECTNESS_ONLY") != nullptr;
  if (!correctness_only) {
    for (const auto& op : kOps) {
      for (const auto& kn : kShapes) {
        LinearConfig cfg{kM, kn.first, kn.second, kGroup, op};
        cases.push_back(make_case(cfg, utils::kTexture3D)); // tiled baseline
        cases.push_back(
            make_case(cfg, utils::kBuffer)); // coopmat (gate-permitting)
      }
    }
  }
  // Correctness: small aligned {64,128,64} cases for ALL FOUR ops; the buffer
  // case fires the coopmat shader, validated by bench_reference (the perf
  // cases above are skipped by it). POSITIVE well-conditioned data (no fp16
  // cancellation): activations are multiples of 1/16 in [0.5,1.375]; int4
  // nibbles in {9..14} (-> weight +1..+6) / int8 weights in {1..6}. For dq8ca
  // the per-row activation scale is forced to 1/16 with zp=0 so the dynamic
  // int8 quant round-trip is EXACT in both fp16 and fp32 (quantized values
  // 8..22) and the fp32 reference is valid. fp16~=fp32 throughout, so a tight
  // tolerance validates shader structure (catches zero-subtile bugs) while
  // ignoring benign fp16 noise. Texture3D = tiled, Buffer = coopmat.
  // Shapes align to BOTH coopmat geometries (64x64x32 legacy, 128x128x16
  // double-buffered); the second shape dispatches a multi-workgroup grid for
  // both, covering the gl_WorkGroupID-derived tile offsets in the store
  // address math.
  // Production-K cases (specs/014-m5-linear-coopmat-retune, FR-008): every
  // shape above tops out at K=256, well short of the real Llama K-dimensions
  // (see kShapes above, K up to 14336). fp16-accumulation drift grows with
  // the K-length of the reduction, so a shader change to the accumulator
  // path (e.g. fp32->fp16 accumulate) can pass at small K and still diverge
  // at production K -- these two entries close that gap using the exact
  // same well-conditioned-data + tolerance strategy as the rest of this
  // table, at the same group_size (128) the real production shapes above
  // use, not a new methodology.
  static const std::vector<LinearConfig> kCorrectnessShapes = {
      {64, 128, 64, 64, ""},
      {128, 256, 128, 64, ""},
      {128, 128, 128, 64, ""},
      {256, 256, 256, 64, ""},
      // Discriminators for the tiled-texture cube-shape failure:
      {128, 128, 256, 64, ""}, // M == K only
      {256, 128, 128, 64, ""}, // K == N only
      {64, 128, 256, 64, ""}, // K > M, K < N
      {256, 128, 64, 64, ""}, // K < M, K > N
      // Production-K (FR-008):
      {128, 2048, 128, 128, ""},
      {128, 4096, 128, 128, ""}};
  for (const auto& op : kOps) {
    for (const auto& shape : kCorrectnessShapes) {
      LinearConfig cfg{shape.M, shape.K, shape.N, shape.group_size, op};
      for (auto st : {utils::kTexture3D, utils::kBuffer}) {
        cases.push_back(make_deterministic_correctness_case(cfg, op, st));
      }
    }
  }
  // Rank-3, batch=1 correctness cases (specs/009-e2e-tokrate-report,
  // research.md Decision 2): the real exported model's linear activations
  // are rank-3 [1, M, K] (batch dim always 1, never squeezed --
  // specs/003-wmma-shader-candidates/research.md). No pre-existing test
  // here (or anywhere in this test directory) exercised that shape through
  // either coopmat kernel before specs/009's guard relaxation made it
  // reachable -- this is genuinely new coverage, not a citation of the
  // rank-2 cases above. Reuses one of kCorrectnessShapes' own
  // already-validated (shape, group_size) pairs, at Buffer storage only
  // (the storage the coopmat path actually requires); Texture3D+rank-3
  // continues to exercise the pre-existing tiled path, unaffected by this
  // change, so is not repeated here.
  // Production-K rank-3 case (FR-008): reuses one of kCorrectnessShapes'
  // own new production-K entries (K=4096, the larger/higher-risk of the
  // two) rather than inventing a separate shape, same rationale as the
  // pre-existing rank-2 reuse above.
  static const std::vector<LinearConfig> kRank3CorrectnessShapes = {
      {128, 128, 128, 64, "", /*batch=*/1},
      {128, 4096, 128, 128, "", /*batch=*/1}};
  for (const auto& op : kOps) {
    for (const auto& shape : kRank3CorrectnessShapes) {
      LinearConfig cfg{
          shape.M, shape.K, shape.N, shape.group_size, op, shape.batch};
      cases.push_back(
          make_deterministic_correctness_case(cfg, op, utils::kBuffer));
    }
  }
  return cases;
}

int64_t flop_calc(const TestCase& tc) {
  const auto& in = tc.inputs()[0].get_tensor_sizes();
  const auto& out = tc.outputs()[0].get_tensor_sizes();
  const int64_t M = in[in.size() - 2], K = in[in.size() - 1], N = out.back();
  return 2 * M * N * K; // MAC = 2 flops
}

int main() {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout
      << "Coopmat vs Tiled quantized-linear microbench (Llama 3.1 8B shapes, M="
      << kM << ")" << std::endl;
  print_separator();

  auto results = execute_test_cases(
      generate_cases,
      flop_calc,
      "CoopmatLinearBench",
      /*warmup=*/3,
      /*runs=*/5,
      /*reference=*/bench_reference);

  // Rank-3, batch=1 dispatch + correctness verdict (specs/009, FR-003 /
  // research.md Decision 1/2): numeric PASS alone doesn't prove the coopmat
  // path actually ran -- the tiled fallback would numerically pass too.
  // Explicitly confirm the dispatched kernel name contains "coopmat".
  bool rank3_all_ok = true;
  for (const auto& r : results) {
    if (r.get_kernel_name().find("_rank3batch") == std::string::npos) {
      continue;
    }
    std::string shader_name = r.get_kernel_name();
    for (const auto& st : r.get_shader_timings()) {
      if (st.shader_name.find("linear_") != std::string::npos) {
        shader_name = st.shader_name;
      }
    }
    const bool fired = shader_name.find("coopmat") != std::string::npos;
    const bool ok =
        fired && r.get_correctness_status() == CorrectnessStatus::PASSED;
    rank3_all_ok = rank3_all_ok && ok;
    std::cout << "[rank3 batch=1] " << r.get_kernel_name() << " -> "
              << shader_name
              << (fired ? " (coopmat dispatched)"
                        : " (NOT coopmat -- fallback)")
              << ", correctness="
              << (r.get_correctness_status() == CorrectnessStatus::PASSED
                      ? "PASSED"
                      : (r.get_correctness_status() == CorrectnessStatus::FAILED
                             ? "FAILED"
                             : "SKIPPED"))
              << "\n";
  }
  if (!rank3_all_ok) {
    std::cout << "[rank3 batch=1] FAILED -- at least one rank-3 case did not "
                 "dispatch coopmat and/or failed correctness\n";
    return 1;
  }

  // Summary table: pair tiled (even idx) vs coopmat (odd idx) per (op, shape).
  // GFLOP/s computed from avg GPU time and 2*M*N*K flops.
  if (results.size() < kOps.size() * kShapes.size() * 2) {
    return 0; // correctness-only run: no perf cases to summarize
  }
  auto gflops = [](float time_us, int64_t M, int64_t K, int64_t N) -> float {
    return time_us > 0 ? (2.0f * M * N * K) / (time_us * 1e3f) : 0.0f;
  };
  // The result's kernel_name is the test-case name; the dispatched shader
  // names are in the per-shader timings (dq8ca cases also run a
  // quantize_and_pack shader, so pick the linear_* one).
  auto linear_kernel = [](const BenchmarkResult& r) -> std::string {
    std::string name = r.get_kernel_name();
    for (const auto& st : r.get_shader_timings()) {
      if (st.shader_name.find("linear_") != std::string::npos) {
        name = st.shader_name;
      }
    }
    return name;
  };
  std::cout
      << "\n================ SUMMARY: tiled vs coopmat (GFLOP/s) ================\n";
  std::cout << std::left << std::setw(22) << "op" << std::setw(13)
            << "shape(K,N)" << std::right << std::setw(10) << "tiled"
            << std::setw(10) << "coopmat" << std::setw(9) << "speedup"
            << "  coopmat kernel\n";
  size_t idx = 0;
  for (const auto& op : kOps) {
    for (const auto& kn : kShapes) {
      const float t_us = results[idx].get_avg_time_us();
      const float c_us = results[idx + 1].get_avg_time_us();
      const std::string coop_kernel = linear_kernel(results[idx + 1]);
      const float tiled = gflops(t_us, kM, kn.first, kn.second);
      const float coop = gflops(c_us, kM, kn.first, kn.second);
      idx += 2;
      // If the "coopmat" (buffer) case did not actually pick a _coopmat shader,
      // flag it (e.g. shape not gate-eligible).
      const bool fired = coop_kernel.find("coopmat") != std::string::npos;
      std::cout << std::left << std::setw(22) << op << std::setw(13)
                << ("(" + std::to_string(kn.first) + "," +
                    std::to_string(kn.second) + ")")
                << std::right << std::setw(10) << std::fixed
                << std::setprecision(1) << tiled << std::setw(10) << coop
                << std::setw(8) << std::setprecision(2)
                << (tiled > 0 ? coop / tiled : 0.0f) << "x"
                << (fired ? "  " : " !") << coop_kernel << "\n";
    }
  }
  std::cout << "(! = buffer case did NOT dispatch a coopmat shader)\n";
  return 0;
}
