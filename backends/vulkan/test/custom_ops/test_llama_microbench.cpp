// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Unified Llama microbenchmark: the merge of the three previous harnesses
// (test_coopmat_linear_bench, test_llama_baseline_bench,
// test_sdpa_coopmat_bench) into one binary, at the real e2e dispatch shapes
// of Llama 3.1 8B / 3.2 3B / 3.2 1B (2048-token prefill, single-token
// decode; BENCHMARKING.md's ctx3072 PTEs).
//
// Suites (all run when no suite flag is given):
//   --linear    coopmat-vs-tiled for the two int4 linear types:
//                 4w    = linear_q4gsw          (weight-only int4)
//                 8da4w = linear_dq8ca_q4gsw    (dyn-act int8 x int4 weight)
//               Texture3D+Half output selects the tiled baseline; Buffer+Half
//               lets QuantizedLinear.cpp's gate pick the _coopmat (WMMA)
//               shader at prefill. At decode (M=1) is_gemv_case
//               short-circuits to the "_coop" gemv shader for BOTH storages.
//   --baseline  the same linear cases run with ET_VK_FORCE_TILED_LINEAR=1
//               (specs/001's no-WMMA baseline): the buffer rows give the
//               forced-tiled reference on the SAME storage the coopmat
//               shader uses, isolating the algorithm from the storage type
//               (specs/004).
//   --sdpa      llama.custom_sdpa.default tiled-vs-coopmat at each model's
//               real attention shape (specs/010/021): prefill S=2048/ctx=3072
//               and decode S=1/ctx=3072/input_pos=3071 (the single most
//               expensive real decode step). SDPA coopmat is default-on in
//               this tree; ET_VK_DISABLE_COOPMAT is the kill switch, toggled
//               per-case here to measure both variants. Decode never
//               considers coopmat (is_gemv), so only tiled is measured there.
//
// Other flags:
//   --model=<substr>     only run models whose name contains <substr>
//   --correctness-only   run just the linear correctness matrix, skip perf
//   --skip-correctness   skip the linear correctness gate before perf
//   --list               print every case that would run (with sizes), no GPU
//   --help
//
// Matching the real exported model:
//   - per-model linear (K,N) from each checkpoint's params.json. lm_head
//     (K,128256) is excluded per specs/021's explicit decision (largest and
//     wildly-variable dispatch; QueryPool-race / GPU-reset trigger).
//   - linear prefill M=2048, decode M=1; group_size 32 (`--group_size 32` /
//     8da4w default), coopmat-eligible for both ops' tile geometries.
//   - rank-3 [1, M, K] activations, never squeezed (specs/003) -- admitted
//     to the coopmat path by specs/009's leading-dims==1 relaxation.
//
// Output: one specs/021-schema "RESULT,..." line per case streamed during
// the run (shared 12 fields, then suite-specific extras -- linear/baseline:
// storage, M; sdpa: num_kv_heads, toggle), then a report: raw-results
// table, per-site WMMA speedups (coopmat vs tiled -- and vs the forced-tiled
// buffer baseline when --baseline ran), and geomeans per scheme/model/suite
// plus an overall geomean.
//
// Perf cases run one execute_test_cases() call each (specs/021 Decision 8's
// pattern) so peak host memory stays bounded by a single case's tensors
// (the 8B M=2048 FFN cases are ~115MB each). Perf cases are perf-only:
// bench_reference rejects their sizes -> SKIPPED (correctness is covered by
// the small deterministic matrix run as a gate before the sweep).

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "utils.h"

using namespace executorch::vulkan::prototyping;
using namespace vkcompute;

namespace {

// ======================= shared: records + output =======================

struct Record {
  std::string suite; // "linear" | "baseline" | "sdpa"
  std::string model;
  std::string scheme; // "4w" | "8da4w" | "" (sdpa)
  std::string regime; // "prefill" | "decode"
  // linear/baseline: op label (wq_wo, ...); sdpa: sub-shader (qk/av/total)
  std::string op;
  std::string storage; // linear/baseline: texture3d | buffer
  // linear/baseline: dispatched kernel class (tiled/coop/coopmat/crashed);
  // sdpa: which ET_VK_DISABLE_COOPMAT setting produced this row
  std::string variant;
  std::string kernel; // full dispatched shader name (informational)
  int64_t M = 0; // sdpa: seq_len
  int64_t K = 0; // sdpa: head_dim
  int64_t N = 0; // sdpa: num_heads
  int64_t kv = 0; // sdpa only: num_kv_heads
  float mean_us = -1.0f;
  float stdev_us = -1.0f;
  // linear/baseline only: the linear_* shader's own per-invocation time.
  // mean_us is OP-level (all unfiltered dispatches -- for 8da4w that
  // includes the activation quantize_and_pack shader, real per-op e2e
  // overhead); kernel_us isolates the linear kernel itself so the two
  // schemes' shader-level numbers stay comparable. -1 for sdpa rows.
  float kernel_us = -1.0f;
  float gflops = -1.0f; // no SDPA meaning (-1 sentinel, per specs/021)
  std::string dispatch = "not_applicable";
  std::string correctness = "SKIPPED";
  bool ok = false;
};

std::vector<Record> g_records;

// specs/021 (research.md Decision 1): shared unified RESULT,... line --
// 12 shared fields, then suite-specific extras.
void emit(const Record& r) {
  g_records.push_back(r);
  std::cout << "RESULT," << r.suite << "," << r.model << "," << r.scheme << ","
            << r.regime << ",";
  if (r.suite == "sdpa") {
    std::cout << r.op << "," << r.K << "," << r.N << "," << r.mean_us << ","
              << r.stdev_us << ",-1," << r.dispatch << ",SKIPPED," << r.kv
              << "," << r.variant << "\n";
  } else {
    std::cout << r.variant << "," << r.K << "," << r.N << "," << r.mean_us
              << "," << r.stdev_us << "," << r.gflops << "," << r.dispatch
              << "," << r.correctness << "," << r.storage << "," << r.M << ","
              << r.kernel_us << "\n";
  }
}

float geomean(const std::vector<float>& v) {
  if (v.empty()) {
    return 0.0f;
  }
  double acc = 0.0;
  for (float x : v) {
    acc += std::log(static_cast<double>(x));
  }
  return static_cast<float>(std::exp(acc / static_cast<double>(v.size())));
}

// ===================== linear / baseline suites =====================

struct LinearConfig {
  int64_t M;
  int64_t K;
  int64_t N;
  int64_t group_size; // only meaningful for 4-bit
  std::string op_name;
  // 0 = rank-2 input/output ({M,K}/{M,N}, the correctness matrix below).
  // >=1 = rank-3 ({batch,M,K}/{batch,M,N}) -- the real exported model's
  // rank-3, batch=1 activations (specs/003, never squeezed), admitted to
  // the coopmat path by specs/009's guard relaxation. All perf cases run
  // this way; kRank3CorrectnessShapes carries the correctness coverage.
  int64_t batch = 0;
  // Perf-case labeling; empty for correctness cases.
  std::string model;
  std::string regime; // "prefill" | "decode"
  std::string op_label; // "wq_wo", "wk_wv", ...
};

bool is_dq8ca(const std::string& op) {
  return op.find("dq8ca") != std::string::npos;
}
bool is_4bit(const std::string& op) {
  return op.find("q4gsw") != std::string::npos;
}

// Build one test case for the given op at (storage, half dtype), no bias.
TestCase make_linear_case(const LinearConfig& cfg, utils::StorageType storage) {
  const vkapi::ScalarType dt = vkapi::kHalf;
  TestCase tc;
  const std::string storage_str =
      (storage == utils::kTexture3D) ? "Texture3D" : "Buffer";
  const std::string prefix = cfg.model.empty()
      ? ""
      : cfg.model + "_" + cfg.regime + "_" + cfg.op_label + "_";
  tc.set_name(
      prefix + cfg.op_name + "_M" + std::to_string(cfg.M) + "_K" +
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

// ---- correctness reference for both ops; oversized shapes (the perf
// cases) throw -> framework marks them SKIPPED. For dq8ca the activation
// quant round-trip (round(x/scale)+zp) is mirrored in fp32; this is exact
// (not just close) for the correctness data below, which uses scale=1/16,
// zp=0 and activations that are multiples of 1/16, so fp16-vs-fp32
// divergence cannot occur. ----
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
void bench_reference(TestCase& tc) {
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
  // and go up to M=2048/N=14336 -- an O(M*N*K) CPU reference at that size
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

// Real per-model linear weight shapes (K,N), from each checkpoint's
// params.json -- the same table specs/001's shapes.json carries. wq/wo,
// wk/wv, and w1/w3 share a (K,N) within each model, so each unique dispatch
// shape is measured once and labeled with both ops. lm_head is excluded
// (see file header).
struct OpShape {
  const char* op_label;
  int64_t K;
  int64_t N;
};
struct LinearModel {
  const char* model;
  std::vector<OpShape> ops;
};
const std::vector<LinearModel> kLinearModels = {
    {"llama-3.1-8b",
     {{"wq_wo", 4096, 4096},
      {"wk_wv", 4096, 1024},
      {"w1_w3", 4096, 14336},
      {"w2", 14336, 4096}}},
    {"llama-3.2-3b",
     {{"wq_wo", 3072, 3072},
      {"wk_wv", 3072, 1024},
      {"w1_w3", 3072, 8192},
      {"w2", 8192, 3072}}},
    {"llama-3.2-1b",
     {{"wq_wo", 2048, 2048},
      {"wk_wv", 2048, 512},
      {"w1_w3", 2048, 8192},
      {"w2", 8192, 2048}}},
};
const std::vector<std::pair<const char*, const char*>> kSchemes = {
    {"4w", "linear_q4gsw"},
    {"8da4w", "linear_dq8ca_q4gsw"}};
// Real regimes: prefill dispatches every linear at M=2048 (the full prompt);
// each of the 1024 decode steps dispatches at M=1, independent of position.
const std::vector<std::pair<const char*, int64_t>> kLinearRegimes = {
    {"prefill", 2048},
    {"decode", 1}};
constexpr int64_t kGroup = 32;
constexpr int kWarmupRuns = 3;
constexpr int kTimedRuns = 5;

// Builds one deterministic, well-conditioned correctness case (POSITIVE
// data, no fp16 cancellation -- see generate_correctness_cases) for the
// given op/shape/storage.
TestCase make_deterministic_correctness_case(
    const LinearConfig& cfg,
    const std::string& op,
    utils::StorageType st) {
  const bool dq = is_dq8ca(op);
  const bool four = is_4bit(op);
  TestCase t = make_linear_case(cfg, st);
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

// Correctness: small aligned cases for both ops; the buffer case fires the
// coopmat shader, validated against the fp32 reference (the perf cases are
// rejected by it). POSITIVE well-conditioned data (no fp16 cancellation):
// activations are multiples of 1/16 in [0.5,1.375]; int4 nibbles in {9..14}
// (-> weight +1..+6). For dq8ca the per-row activation scale is forced to
// 1/16 with zp=0 so the dynamic int8 quant round-trip is EXACT in both fp16
// and fp32 and the fp32 reference is valid. fp16~=fp32 throughout, so a
// tight tolerance validates shader structure (catches zero-subtile bugs)
// while ignoring benign fp16 noise. Texture3D = tiled, Buffer = coopmat.
// Shapes align to BOTH coopmat geometries (64x64x32 legacy, 128x128x16
// double-buffered); the second shape dispatches a multi-workgroup grid for
// both, covering the gl_WorkGroupID-derived tile offsets in the store
// address math.
// Production-K cases (specs/014, FR-008): fp16-accumulation drift grows
// with the K-length of the reduction, so a shader change to the accumulator
// path can pass at small K and still diverge at production K -- the
// K=2048/4096 entries close that gap. (These correctness rows keep their
// original group sizes; the perf sweep's real-export group_size is kGroup.)
std::vector<TestCase> generate_correctness_cases() {
  std::vector<TestCase> cases;
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
  for (const auto& scheme : kSchemes) {
    for (const auto& shape : kCorrectnessShapes) {
      LinearConfig cfg{
          shape.M, shape.K, shape.N, shape.group_size, scheme.second};
      for (auto st : {utils::kTexture3D, utils::kBuffer}) {
        cases.push_back(
            make_deterministic_correctness_case(cfg, scheme.second, st));
      }
    }
  }
  // Rank-3, batch=1 correctness cases (specs/009): the real exported
  // model's linear activations are rank-3 [1, M, K]. The perf sweep runs
  // rank-3 too, but perf-only -- these are the cases that actually validate
  // the shape against the fp32 reference, at Buffer storage (the storage
  // the coopmat path requires); Texture3D+rank-3 exercises the pre-existing
  // tiled path, so is not repeated here. The K=4096 entry is the
  // production-K rank-3 case (FR-008).
  static const std::vector<LinearConfig> kRank3CorrectnessShapes = {
      {128, 128, 128, 64, "", /*batch=*/1},
      {128, 4096, 128, 128, "", /*batch=*/1}};
  for (const auto& scheme : kSchemes) {
    for (const auto& shape : kRank3CorrectnessShapes) {
      LinearConfig cfg{
          shape.M,
          shape.K,
          shape.N,
          shape.group_size,
          scheme.second,
          shape.batch};
      cases.push_back(make_deterministic_correctness_case(
          cfg, scheme.second, utils::kBuffer));
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

// The result's kernel_name is the test-case name; the dispatched shader
// names are in the per-shader timings (dq8ca cases also run a
// quantize_and_pack shader, so pick the linear_* one).
std::string linear_kernel(const BenchmarkResult& r) {
  std::string name = r.get_kernel_name();
  for (const auto& st : r.get_shader_timings()) {
    if (st.shader_name.find("linear_") != std::string::npos) {
      name = st.shader_name;
    }
  }
  return name;
}

// Per-invocation time of the linear_* shader alone. ShaderTiming holds one
// iter_timings_us entry per dispatch (chained dispatches included), so its
// get_avg_time_us() is already per-invocation -- unlike the case-level
// mean_us, which sums every unfiltered dispatch (for dq8ca that adds the
// activation quantize_and_pack shader).
float linear_kernel_us(const BenchmarkResult& r) {
  float us = -1.0f;
  for (const auto& st : r.get_shader_timings()) {
    if (st.shader_name.find("linear_") != std::string::npos) {
      us = st.get_avg_time_us();
    }
  }
  return us;
}

std::string kernel_class(const std::string& kernel) {
  // _coopmat must be checked before _coop (substring).
  if (kernel.find("_coopmat") != std::string::npos) {
    return "coopmat";
  }
  if (kernel.find("_coop") != std::string::npos) {
    return "coop";
  }
  return "tiled";
}

// Runs the linear correctness matrix and the specs/009 rank-3
// dispatch+correctness verdict. Returns false on any failure. Must run
// WITHOUT ET_VK_FORCE_TILED_LINEAR set -- the buffer cases exist to fire
// and validate the coopmat shader.
//
// One execute_test_cases() call per case, each wrapped in try/catch: the
// framework throws on a numeric validation failure (utils.cpp's
// execute_test_cases, outside its own try/catch), so a batched call would
// die at the FIRST failing case and never enumerate the rest. Per-case
// execution turns that throw into one recorded failure and keeps going --
// the whole point of the gate is to list everything that broke. (Costs the
// cross-case reference cache, but every shape here is small.)
bool run_linear_correctness() {
  unsetenv("ET_VK_FORCE_TILED_LINEAR");
  std::vector<BenchmarkResult> results;
  std::vector<std::string> failed_names;
  for (auto& tc : generate_correctness_cases()) {
    try {
      auto res = execute_test_cases(
          [&tc]() { return std::vector<TestCase>{tc}; },
          flop_calc,
          "LlamaMicrobenchCorrectness",
          kWarmupRuns,
          kTimedRuns,
          bench_reference);
      if (!res.empty()) {
        if (res[0].get_correctness_status() == CorrectnessStatus::FAILED) {
          failed_names.push_back(tc.name());
        }
        results.push_back(res[0]);
      }
    } catch (const std::exception& e) {
      failed_names.push_back(tc.name());
      std::cout << "[correctness] " << tc.name() << " FAILED: " << e.what()
                << "\n";
    }
  }
  bool all_ok = failed_names.empty();
  // Rank-3, batch=1 dispatch + correctness verdict: numeric PASS alone
  // doesn't prove the coopmat path actually ran -- the tiled fallback would
  // numerically pass too. Explicitly confirm the dispatched kernel name.
  for (const auto& r : results) {
    if (r.get_kernel_name().find("_rank3batch") == std::string::npos) {
      continue;
    }
    const std::string shader_name = linear_kernel(r);
    const bool fired = shader_name.find("coopmat") != std::string::npos;
    const bool ok =
        fired && r.get_correctness_status() == CorrectnessStatus::PASSED;
    all_ok = all_ok && ok;
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
  if (!failed_names.empty()) {
    std::cout << "[correctness] " << failed_names.size()
              << " case(s) FAILED:\n";
    for (const auto& n : failed_names) {
      std::cout << "  " << n << "\n";
    }
  }
  if (!all_ok) {
    std::cout << "[correctness] FAILED -- numeric failure(s) and/or a rank-3 "
                 "case did not dispatch coopmat\n";
  }
  return all_ok;
}

struct PerfCase {
  LinearConfig cfg;
  utils::StorageType storage;
};
std::vector<PerfCase> generate_linear_perf_cases(
    const std::string& model_filter) {
  std::vector<PerfCase> cases;
  for (const auto& scheme : kSchemes) {
    for (const auto& model : kLinearModels) {
      if (std::string(model.model).find(model_filter) == std::string::npos) {
        continue;
      }
      for (const auto& regime : kLinearRegimes) {
        for (const auto& shape : model.ops) {
          LinearConfig cfg{
              regime.second,
              shape.K,
              shape.N,
              kGroup,
              scheme.second,
              /*batch=*/1,
              model.model,
              regime.first,
              shape.op_label};
          cases.push_back({cfg, utils::kTexture3D}); // tiled/gemv baseline
          cases.push_back({cfg, utils::kBuffer}); // coopmat (gate-permitting)
        }
      }
    }
  }
  return cases;
}

// Runs the linear perf sweep as either the "linear" suite (coopmat enabled)
// or the "baseline" suite (ET_VK_FORCE_TILED_LINEAR=1 for every case --
// specs/001's no-WMMA baseline; its buffer rows are the forced-tiled
// reference on the same storage the coopmat shader uses). One
// execute_test_cases() call per case (see file header); a case-local
// failure is recorded as a crashed row and must not take down the sweep.
void run_linear_suite(
    const std::string& suite,
    const std::string& model_filter) {
  const bool force_tiled = suite == "baseline";
  if (force_tiled) {
    setenv("ET_VK_FORCE_TILED_LINEAR", "1", /*overwrite=*/1);
  } else {
    unsetenv("ET_VK_FORCE_TILED_LINEAR");
  }
  for (const auto& pc : generate_linear_perf_cases(model_filter)) {
    const LinearConfig& cfg = pc.cfg;
    Record rec;
    rec.suite = suite;
    rec.model = cfg.model;
    rec.scheme = is_dq8ca(cfg.op_name) ? "8da4w" : "4w";
    rec.regime = cfg.regime;
    rec.op = cfg.op_label;
    rec.storage = pc.storage == utils::kTexture3D ? "texture3d" : "buffer";
    rec.M = cfg.M;
    rec.K = cfg.K;
    rec.N = cfg.N;
    rec.variant = "crashed";
    rec.kernel = "CRASHED";
    rec.correctness = "SKIPPED"; // perf-only; see bench_reference
    try {
      TestCase tc = make_linear_case(cfg, pc.storage);
      auto res = execute_test_cases(
          [&tc]() { return std::vector<TestCase>{tc}; },
          flop_calc,
          "LlamaMicrobench",
          kWarmupRuns,
          kTimedRuns,
          bench_reference);
      if (!res.empty()) {
        rec.mean_us = res[0].get_avg_time_us();
        rec.stdev_us = res[0].get_std_dev_us();
        rec.kernel = linear_kernel(res[0]);
        rec.kernel_us = linear_kernel_us(res[0]);
        rec.variant = kernel_class(rec.kernel);
        rec.gflops = rec.mean_us > 0
            ? (2.0f * cfg.M * cfg.N * cfg.K) / (rec.mean_us * 1e3f)
            : -1.0f;
        rec.ok = true;
      }
    } catch (const std::exception& e) {
      std::cerr << "WARNING: case '" << cfg.model << " " << cfg.regime << " "
                << cfg.op_label << " " << cfg.op_name << " " << rec.storage
                << "' threw: " << e.what()
                << " -- recorded as CRASHED, continuing\n";
    }
    // Dispatch expectation: the linear suite's prefill buffer rows must be
    // coopmat ("confirmed"/"fallback_tiled" -- specs/021 FR-006 semantics);
    // everywhere else no coopmat is possible (texture, decode's is_gemv
    // short-circuit, or the baseline suite's forced-tiled), so the status
    // is not_applicable -- except a coopmat kernel showing up there, which
    // is a real anomaly.
    if (rec.ok) {
      const bool expects_coopmat = suite == "linear" &&
          rec.regime == "prefill" && rec.storage == "buffer";
      if (expects_coopmat) {
        rec.dispatch =
            rec.variant == "coopmat" ? "confirmed" : "fallback_tiled";
      } else {
        rec.dispatch =
            rec.variant == "coopmat" ? "unexpected_coopmat" : "not_applicable";
      }
    } else {
      rec.dispatch = "crashed";
    }
    emit(rec);
  }
  if (force_tiled) {
    unsetenv("ET_VK_FORCE_TILED_LINEAR");
  }
}

// ============================ sdpa suite ============================

struct SdpaModel {
  const char* name;
  int64_t head_dim;
  int64_t num_heads;
  int64_t num_kv_heads;
};
// Real per-model shapes, derived directly from each checkpoint's params.json
// (dim / n_heads), matching specs/010 research.md Decision 5.
const std::vector<SdpaModel> kSdpaModels = {
    {"llama-3.1-8b", 128, 32, 8},
    {"llama-3.2-3b", 128, 24, 8},
    {"llama-3.2-1b", 64, 32, 8},
};

// specs/021: real e2e regimes. context_len=3072 for BOTH regimes: the real
// ctx3072 PTEs (2048 prefill + 1024 decode) allocate the KV cache at
// max_context_length up front, so even the prefill step's cache tensor --
// and therefore the attention shaders' strides/access pattern -- is sized
// 3072, not 2048. input_pos=3071 is the single most expensive real decode
// step (attends over the fullest cache). SDPA.cpp's is_gemv gate means the
// coopmat toggle has no effect at decode.
struct SdpaRegime {
  const char* regime;
  int64_t seq_len; // this step's query / newly-written KV length
  int64_t context_len; // KV cache buffer size
  int64_t input_pos; // symint value
};
const std::vector<SdpaRegime> kSdpaRegimes = {
    {"prefill", 2048, 3072, 0},
    {"decode", 1, 3072, 3071},
};

struct SdpaRunResult {
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

// Runs one (model, coopmat-toggle, regime) case: builds
// llama.custom_sdpa.default directly via ComputeGraph (specs/010 research.md
// Decision 8: the TestCase framework has no SymInt support and this op
// family requires one). Returns the SDPA-compute-only GPU time split into
// qk (sdpa_compute_attn_weights_*) and av (sdpa_compute_out_*), plus their
// combined total -- excluding the kv-cache-update and softmax dispatches in
// between (unaccelerated, identical regardless of the coopmat toggle).
//
// SDPA coopmat is default-on in this tree; ET_VK_DISABLE_COOPMAT is the
// kill switch (read at shader-pick / graph-build time), so "tiled" here
// means running with it set. The KV cache buffer is always sized to
// `regime.context_len`, filled with random data BEFORE update_cache writes
// this step's new K/V at input_pos -- the shapes and access pattern are
// real even though the "history" isn't a genuine step-by-step prefill walk
// (only timing is measured here, not output correctness).
SdpaRunResult sdpa_run_case(
    const SdpaModel& m,
    bool enable_coopmat,
    const SdpaRegime& regime) {
  if (enable_coopmat) {
    unsetenv("ET_VK_DISABLE_COOPMAT");
  } else {
    setenv("ET_VK_DISABLE_COOPMAT", "1", /*overwrite=*/1);
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
  // 0..input_pos-1 with. That's fine -- only timing is measured here, and
  // the attention shaders' dispatch size/access pattern depends solely on
  // the cache's shape (context_len), not its contents.

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

  SdpaRunResult result;
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

void emit_sdpa_records(
    const SdpaModel& m,
    const SdpaRegime& regime,
    const std::string& toggle,
    const SdpaRunResult& r,
    const std::string& dispatch) {
  const struct {
    const char* sub;
    float mean;
    float stdev;
  } subs[] = {
      {"qk", r.qk_mean_us, r.qk_stdev_us},
      {"av", r.av_mean_us, r.av_stdev_us},
      {"total", r.mean_us, r.stdev_us},
  };
  for (const auto& s : subs) {
    Record rec;
    rec.suite = "sdpa";
    rec.model = m.name;
    rec.regime = regime.regime;
    rec.op = s.sub;
    rec.variant = toggle;
    rec.M = regime.seq_len;
    rec.K = m.head_dim;
    rec.N = m.num_heads;
    rec.kv = m.num_kv_heads;
    rec.mean_us = s.mean;
    rec.stdev_us = s.stdev;
    rec.dispatch = dispatch;
    rec.ok = true;
    emit(rec);
  }
}

// Returns false if any prefill case failed to confirm coopmat dispatch.
bool run_sdpa_suite(const std::string& model_filter) {
  bool all_confirmed = true;
  for (const auto& m : kSdpaModels) {
    if (std::string(m.name).find(model_filter) == std::string::npos) {
      continue;
    }
    for (const auto& regime : kSdpaRegimes) {
      const bool is_decode = std::string(regime.regime) == "decode";
      SdpaRunResult tiled = sdpa_run_case(m, /*enable_coopmat=*/false, regime);
      // Decode: SDPA.cpp's is_gemv gate never considers coopmat -- a second
      // invocation would just remeasure the identical dispatch. Skip it.
      SdpaRunResult coopmat =
          is_decode ? tiled : sdpa_run_case(m, /*enable_coopmat=*/true, regime);

      std::string dispatch;
      if (is_decode) {
        dispatch = "not_applicable";
      } else {
        const bool tiled_is_tiled =
            !has_kernel_containing(tiled.dispatched_kernels, "_coopmat");
        const bool qk_coopmat = has_kernel_containing(
            coopmat.dispatched_kernels, "sdpa_compute_attn_weights_coopmat");
        const bool av_coopmat = has_kernel_containing(
            coopmat.dispatched_kernels, "sdpa_compute_out_coopmat");
        dispatch = (tiled_is_tiled && qk_coopmat && av_coopmat)
            ? "confirmed"
            : "fallback_tiled";
        all_confirmed = all_confirmed && dispatch == "confirmed";
      }

      emit_sdpa_records(m, regime, "tiled", tiled, dispatch);
      if (!is_decode) {
        emit_sdpa_records(m, regime, "coopmat", coopmat, dispatch);
      }
    }
  }
  unsetenv("ET_VK_DISABLE_COOPMAT"); // restore the tree's default-on state
  return all_confirmed;
}

// ============================== report ==============================

const Record* find_record(
    const std::string& suite,
    const std::string& model,
    const std::string& scheme,
    const std::string& regime,
    const std::string& op,
    const std::string& storage,
    const std::string& variant = "") {
  for (const auto& r : g_records) {
    if (r.suite == suite && r.model == model && r.scheme == scheme &&
        r.regime == regime && r.op == op && r.storage == storage &&
        (variant.empty() || r.variant == variant) && r.ok) {
      return &r;
    }
  }
  return nullptr;
}

std::string fmt_us(float us) {
  if (us < 0) {
    return "-";
  }
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(1) << us;
  return ss.str();
}
std::string fmt_x(float x) {
  if (x <= 0) {
    return "-";
  }
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(2) << x << "x";
  return ss.str();
}

// Prints the raw-results table, the per-site WMMA speedups, and the
// geomeans. Returns false if any expected coopmat site failed to speed up
// AND failed to dispatch -- dispatch anomalies, not slowness, fail the run.
void print_report(bool baseline_ran) {
  print_separator();
  std::cout << "==================== RAW RESULTS ====================\n";
  std::cout << std::left << std::setw(10) << "suite" << std::setw(14) << "model"
            << std::setw(7) << "scheme" << std::setw(9) << "regime"
            << std::setw(7) << "op" << std::setw(11) << "storage"
            << std::setw(9) << "variant" << std::setw(21) << "(M,K,N)"
            << std::right << std::setw(12) << "mean_us" << std::setw(10)
            << "stdev" << std::setw(10) << "kern_us" << std::setw(10)
            << "GFLOP/s" << "  dispatch\n";
  for (const auto& r : g_records) {
    std::ostringstream shape;
    shape << "(" << r.M << "," << r.K << "," << r.N << ")";
    std::cout << std::left << std::setw(10) << r.suite << std::setw(14)
              << r.model << std::setw(7) << (r.scheme.empty() ? "-" : r.scheme)
              << std::setw(9) << r.regime << std::setw(7) << r.op
              << std::setw(11) << (r.storage.empty() ? "-" : r.storage)
              << std::setw(9) << r.variant << std::setw(21) << shape.str()
              << std::right << std::setw(12) << fmt_us(r.mean_us)
              << std::setw(10) << fmt_us(r.stdev_us) << std::setw(10)
              << fmt_us(r.kernel_us) << std::setw(10)
              << (r.gflops >= 0 ? fmt_us(r.gflops) : "-") << "  " << r.dispatch
              << "\n";
  }

  // ---- linear WMMA speedups (prefill only; decode has no coopmat) ----
  std::vector<float> all_wmma_speedups;
  bool have_linear = false;
  for (const auto& r : g_records) {
    have_linear = have_linear || r.suite == "linear";
  }
  if (have_linear) {
    // op_x = op-level speedup (all of the op's dispatches -- for 8da4w that
    // includes the activation quantize_and_pack shader, which the real
    // model pays on every linear, so this is the per-op e2e gain). kern_x =
    // the linear shader alone, the number that judges the WMMA kernel
    // itself. For 4w the two coincide (no quantize dispatch). Geomeans use
    // op_x -- the e2e-relevant quantity -- with kern_x geomeans printed
    // alongside per scheme.
    std::cout << "\n========== LINEAR: coopmat (WMMA) vs tiled, prefill "
                 "M=2048 ==========\n";
    std::cout << std::left << std::setw(7) << "scheme" << std::setw(14)
              << "model" << std::setw(7) << "op" << std::setw(15) << "(K,N)"
              << std::right << std::setw(12) << "tiled_tex" << std::setw(12)
              << "coopmat" << std::setw(9) << "op_x" << std::setw(9)
              << "kern_x";
    if (baseline_ran) {
      std::cout << std::setw(14) << "tiled_buf" << std::setw(9) << "vs_buf";
    }
    std::cout << "  (us; op_x = whole op incl. 8da4w act-quant, kern_x = "
                 "linear shader only)\n";
    std::vector<std::pair<std::string, std::vector<float>>> scheme_geo;
    std::vector<std::pair<std::string, std::vector<float>>> scheme_kern_geo;
    for (const auto& scheme : kSchemes) {
      std::vector<float> scheme_speedups;
      std::vector<float> scheme_kern_speedups;
      for (const auto& model : kLinearModels) {
        std::vector<float> model_speedups;
        for (const auto& shape : model.ops) {
          const Record* tex = find_record(
              "linear",
              model.model,
              scheme.first,
              "prefill",
              shape.op_label,
              "texture3d");
          // Unfiltered buffer row for display (shows the actually-dispatched
          // kernel even on a fallback); coopmat-filtered row for speedups
          // and geomeans, so a fallback can never contribute a bogus ratio.
          const Record* buf = find_record(
              "linear",
              model.model,
              scheme.first,
              "prefill",
              shape.op_label,
              "buffer");
          const Record* cm = (buf && buf->variant == "coopmat") ? buf : nullptr;
          const Record* base_buf = baseline_ran ? find_record(
                                                      "baseline",
                                                      model.model,
                                                      scheme.first,
                                                      "prefill",
                                                      shape.op_label,
                                                      "buffer")
                                                : nullptr;
          if (tex == nullptr && buf == nullptr) {
            continue; // model filtered out
          }
          const float speedup = (tex && cm && cm->mean_us > 0)
              ? tex->mean_us / cm->mean_us
              : 0.0f;
          const float kern_speedup =
              (tex && cm && tex->kernel_us > 0 && cm->kernel_us > 0)
              ? tex->kernel_us / cm->kernel_us
              : 0.0f;
          const float vs_buf = (base_buf && cm && cm->mean_us > 0)
              ? base_buf->mean_us / cm->mean_us
              : 0.0f;
          if (speedup > 0) {
            model_speedups.push_back(speedup);
            all_wmma_speedups.push_back(speedup);
          }
          if (kern_speedup > 0) {
            scheme_kern_speedups.push_back(kern_speedup);
          }
          std::cout << std::left << std::setw(7) << scheme.first
                    << std::setw(14) << model.model << std::setw(7)
                    << shape.op_label << std::setw(15)
                    << ("(" + std::to_string(shape.K) + "," +
                        std::to_string(shape.N) + ")")
                    << std::right << std::setw(12)
                    << fmt_us(tex ? tex->mean_us : -1.0f) << std::setw(12)
                    << fmt_us(buf ? buf->mean_us : -1.0f) << std::setw(9)
                    << fmt_x(speedup) << std::setw(9) << fmt_x(kern_speedup);
          if (baseline_ran) {
            std::cout << std::setw(14)
                      << fmt_us(base_buf ? base_buf->mean_us : -1.0f)
                      << std::setw(9) << fmt_x(vs_buf);
          }
          if (buf && buf->variant != "coopmat") {
            std::cout << "  ! " << buf->kernel;
          }
          std::cout << "\n";
        }
        if (!model_speedups.empty()) {
          std::cout << std::left << std::setw(7) << scheme.first
                    << std::setw(14) << model.model << std::setw(7) << "geo"
                    << std::setw(15) << "" << std::right << std::setw(12) << ""
                    << std::setw(12) << "" << std::setw(9)
                    << fmt_x(geomean(model_speedups)) << "\n";
          scheme_speedups.insert(
              scheme_speedups.end(),
              model_speedups.begin(),
              model_speedups.end());
        }
      }
      scheme_geo.emplace_back(scheme.first, scheme_speedups);
      scheme_kern_geo.emplace_back(scheme.first, scheme_kern_speedups);
    }
    for (size_t i = 0; i < scheme_geo.size(); ++i) {
      if (!scheme_geo[i].second.empty()) {
        std::cout << "linear " << scheme_geo[i].first
                  << " geomean (all models): op "
                  << fmt_x(geomean(scheme_geo[i].second)) << ", kernel "
                  << fmt_x(geomean(scheme_kern_geo[i].second)) << "\n";
      }
    }
    std::cout << "(! = buffer case did NOT dispatch a coopmat shader; shown "
                 "for reference, excluded from speedups/geomeans)\n";
  }

  // ---- sdpa WMMA speedups (prefill only) ----
  bool have_sdpa = false;
  for (const auto& r : g_records) {
    have_sdpa = have_sdpa || r.suite == "sdpa";
  }
  if (have_sdpa) {
    std::cout << "\n========== SDPA: coopmat (WMMA) vs tiled, prefill S=2048 "
                 "==========\n";
    std::cout << std::left << std::setw(14) << "model" << std::setw(7) << "sub"
              << std::right << std::setw(12) << "tiled_us" << std::setw(12)
              << "coopmat_us" << std::setw(9) << "speedup" << "  dispatch\n";
    std::vector<float> sdpa_totals;
    for (const auto& m : kSdpaModels) {
      for (const char* sub : {"qk", "av", "total"}) {
        const Record* t =
            find_record("sdpa", m.name, "", "prefill", sub, "", "tiled");
        const Record* c =
            find_record("sdpa", m.name, "", "prefill", sub, "", "coopmat");
        if (t == nullptr || c == nullptr) {
          continue; // model filtered out
        }
        const float speedup = c->mean_us > 0 ? t->mean_us / c->mean_us : 0.0f;
        // Only "total" (qk+av combined) feeds the geomeans -- counting qk
        // and av separately alongside it would double-weight each model.
        // Only confirmed-dispatch rows count.
        if (std::string(sub) == "total" && speedup > 0 &&
            c->dispatch == "confirmed") {
          sdpa_totals.push_back(speedup);
          all_wmma_speedups.push_back(speedup);
        }
        std::cout << std::left << std::setw(14) << m.name << std::setw(7) << sub
                  << std::right << std::setw(12) << fmt_us(t->mean_us)
                  << std::setw(12) << fmt_us(c->mean_us) << std::setw(9)
                  << fmt_x(speedup) << "  " << c->dispatch << "\n";
      }
    }
    if (!sdpa_totals.empty()) {
      std::cout << "sdpa geomean (total, all models): "
                << fmt_x(geomean(sdpa_totals)) << "\n";
    }
  }

  if (!all_wmma_speedups.empty()) {
    std::cout << "\nOVERALL WMMA geomean (" << all_wmma_speedups.size()
              << " sites: linear prefill shapes + sdpa prefill totals): "
              << fmt_x(geomean(all_wmma_speedups)) << "\n";
  }
}

void print_usage() {
  std::cout
      << "test_llama_microbench: unified Llama linear/SDPA microbenchmark\n"
         "  --linear             run the coopmat-vs-tiled linear suite\n"
         "  --baseline           run the forced-tiled (no-WMMA) linear suite\n"
         "  --sdpa               run the SDPA coopmat-vs-tiled suite\n"
         "                       (no suite flag = all three)\n"
         "  --model=<substr>     only models whose name contains <substr>\n"
         "  --correctness-only   run just the linear correctness matrix\n"
         "  --skip-correctness   skip the correctness gate before perf\n"
         "  --list               print every case with its sizes, no GPU\n"
         "  --help               this message\n";
}

void list_cases(
    bool linear,
    bool baseline,
    bool sdpa,
    const std::string& model_filter) {
  int n = 0;
  for (const char* suite : {"linear", "baseline"}) {
    if ((std::string(suite) == "linear" && !linear) ||
        (std::string(suite) == "baseline" && !baseline)) {
      continue;
    }
    for (const auto& pc : generate_linear_perf_cases(model_filter)) {
      std::cout << suite << "," << pc.cfg.model << ","
                << (is_dq8ca(pc.cfg.op_name) ? "8da4w" : "4w") << ","
                << pc.cfg.regime << "," << pc.cfg.op_label << ","
                << (pc.storage == utils::kTexture3D ? "texture3d" : "buffer")
                << ",[1," << pc.cfg.M << "," << pc.cfg.K << "]x[" << pc.cfg.K
                << "," << pc.cfg.N << "],group" << pc.cfg.group_size << "\n";
      ++n;
    }
  }
  if (sdpa) {
    for (const auto& m : kSdpaModels) {
      if (std::string(m.name).find(model_filter) == std::string::npos) {
        continue;
      }
      for (const auto& regime : kSdpaRegimes) {
        const bool is_decode = std::string(regime.regime) == "decode";
        for (const char* toggle : {"tiled", "coopmat"}) {
          if (is_decode && std::string(toggle) == "coopmat") {
            continue;
          }
          std::cout << "sdpa," << m.name << ",," << regime.regime << ",qk+av,"
                    << toggle << ",S" << regime.seq_len << "_ctx"
                    << regime.context_len << "_pos" << regime.input_pos
                    << ",head" << m.head_dim << "_h" << m.num_heads << "_kv"
                    << m.num_kv_heads << "\n";
          ++n;
        }
      }
    }
  }
  std::cout << n << " cases\n";
}

} // namespace

int main(int argc, char** argv) {
  bool linear = false, baseline = false, sdpa = false;
  bool correctness_only = false, skip_correctness = false, list_only = false;
  std::string model_filter;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--linear") {
      linear = true;
    } else if (arg == "--baseline") {
      baseline = true;
    } else if (arg == "--sdpa") {
      sdpa = true;
    } else if (arg == "--correctness-only") {
      correctness_only = true;
    } else if (arg == "--skip-correctness") {
      skip_correctness = true;
    } else if (arg == "--list") {
      list_only = true;
    } else if (arg.rfind("--model=", 0) == 0) {
      model_filter = arg.substr(8);
    } else if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    } else {
      std::cerr << "unknown flag: " << arg << "\n";
      print_usage();
      return 2;
    }
  }
  if (!linear && !baseline && !sdpa) {
    linear = baseline = sdpa = true; // default: everything
  }

  if (list_only) {
    list_cases(linear, baseline, sdpa, model_filter);
    return 0;
  }

  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "Llama microbench (3.1 8B / 3.2 3B / 3.2 1B real e2e shapes; "
               "prefill 2048 / decode 1 @ ctx3072; linear group_size="
            << kGroup << "; " << kWarmupRuns << " warmup + " << kTimedRuns
            << " timed runs per case)\n";
  // Device provenance: without this, thermal/DVFS drift between runs (or
  // between the linear and baseline suites within one run) cannot even be
  // diagnosed post hoc from a saved log.
  {
    const auto* adapter = api::context()->adapter_ptr();
    std::cout << "DEVICE," << adapter->device_name()
              << ",timestamp_period_ns=" << adapter->timestamp_period()
              << ",subgroup_size=" << adapter->subgroup_size() << ",coopmat="
              << (adapter->supports_cooperative_matrix() ? "yes" : "no")
              << "\n";
  }
  print_separator();

  std::srand(0);
  bool ok = true;

  // Correctness gate: validates the tiled and coopmat linear kernels
  // (including the rank-3 dispatch check) before any perf time is spent.
  if (correctness_only) {
    return run_linear_correctness() ? 0 : 1;
  }
  if ((linear || baseline) && !skip_correctness) {
    if (!run_linear_correctness()) {
      std::cout << "correctness gate FAILED -- not running the perf sweep\n";
      return 1;
    }
  }

  if (linear) {
    run_linear_suite("linear", model_filter);
  }
  if (baseline) {
    run_linear_suite("baseline", model_filter);
  }
  bool sdpa_confirmed = true;
  if (sdpa) {
    sdpa_confirmed = run_sdpa_suite(model_filter);
  }

  print_report(baseline);

  // Exit code reflects dispatch sanity, not speed: every linear-suite
  // prefill buffer row must have dispatched coopmat, no coopmat may appear
  // where it can't (decode/forced-tiled/texture), nothing crashed, and
  // every sdpa prefill case must have confirmed coopmat dispatch.
  for (const auto& r : g_records) {
    if (r.dispatch == "fallback_tiled" || r.dispatch == "unexpected_coopmat" ||
        r.dispatch == "crashed") {
      ok = false;
    }
  }
  ok = ok && sdpa_confirmed;
  if (!ok) {
    std::cout << "\nOne or more cases crashed or did not dispatch the "
                 "expected kernel (see dispatch column) -- do not trust "
                 "their speedup numbers.\n";
  }
  return ok ? 0 : 1;
}
