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
//   --sdpa-regions-only  enumerate the QK^T causal-mask tile grid for every
//                       SDPA correctness case and print each tile's region
//                       classification (all_masked / all_visible / diagonal),
//                       then report which of the three shader paths the gate
//                       actually reaches. Host-side only, no GPU. A path no
//                       case produces a tile for is UNCOVERED, however many
//                       times --sdpa-correctness-only passes.
//   --sdpa-tier=<fast|regions|all>
//                       which SDPA correctness tier to run; default all.
//                       "fast" = the original S=128 cases, the cheap
//                       post-edit pre-check. "regions" = the S=256 cases
//                       that put a tile in every QK^T mask region (4x the
//                       reference cost). "all" = the extended gate, which is
//                       what an accept decision requires.
//   --sdpa-correctness-only  run just the SDPA coopmat correctness cases
//                       (sdpa_compute_attn_weights_coopmat /
//                       sdpa_compute_out_coopmat vs. a CPU causal-attention
//                       reference, at small tile-aligned shapes -- see
//                       run_sdpa_correctness). A single pass is NOT
//                       sufficient evidence for a coopmat shader (the linear
//                       tile sweep found a tile that passed once and then
//                       failed 1-in-10 identical repeats, silently) -- rerun
//                       this flag repeatedly before trusting a pass.
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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
  // Median and coefficient of variation of the linear kernel's own
  // per-iteration timings. The text report has always shown mean +/- stdev;
  // these are for --json, where a consumer ranking candidates needs a robust
  // centre and an explicit noise figure rather than a mean that one outlier
  // can move. -1 where the suite does not produce per-iteration samples.
  float kernel_median_us = -1.0f;
  float kernel_cov = -1.0f;
  float gflops = -1.0f; // no SDPA meaning (-1 sentinel, per specs/021)
  std::string dispatch = "not_applicable";
  std::string correctness = "SKIPPED";
  // Failure text, for the cases where the framework threw. Empty otherwise.
  std::string detail_note;
  bool ok = false;
};

std::vector<Record> g_records;

// specs/021 (research.md Decision 1): shared unified RESULT,... line --
// 12 shared fields, then suite-specific extras.
// Append to the record set WITHOUT printing. emit() below both records and
// prints a RESULT line; the correctness matrix must not gain one, because the
// text output has to stay byte-identical -- but --json still needs its
// verdicts, which are the whole point of a correctness gate.
void record_only(const Record& r) {
  g_records.push_back(r);
}

void emit(const Record& r) {
  g_records.push_back(r);
  std::cout << "RESULT," << r.suite << "," << r.model << "," << r.scheme << ","
            << r.regime << ",";
  if (r.suite == "sdpa") {
    std::cout << r.op << "," << r.K << "," << r.N << "," << r.mean_us << ","
              << r.stdev_us << ",-1," << r.dispatch << ",SKIPPED," << r.kv
              << "," << r.variant << "\n";
  } else {
    // r.kernel (the full dispatched shader name) is appended last, after the
    // specs/021 fields, so existing parsers keep working. It is the ONLY
    // field that identifies WHICH tile variant ran: r.variant is
    // kernel_class(), which collapses every shader to coopmat/coop/tiled,
    // and r.dispatch is derived from r.variant. Without this a tile-sweep
    // driver cannot tell that an unrecognized ET_VK_*_COOPMAT_VARIANT token
    // silently fell back to the default kernel.
    std::cout << r.variant << "," << r.K << "," << r.N << "," << r.mean_us
              << "," << r.stdev_us << "," << r.gflops << "," << r.dispatch
              << "," << r.correctness << "," << r.storage << "," << r.M << ","
              << r.kernel_us << "," << r.kernel << "\n";
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
// Opt-in escape hatch for run_production_diff() below: false everywhere else,
// so the perf sweep's SKIPPED-via-throw behavior at production M/N/K is
// unchanged (et-microbench-correctness-gate-blind-above-256 still applies to
// every existing call site). --production-diff sets this for the duration of
// its own run only, accepting the O(M*N*K) CPU cost as a one-shot diagnostic,
// not something to pay on every correctness/perf invocation.
bool g_allow_large_reference = false;

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
  if (!g_allow_large_reference && (M > 256 || N > 256 || K > 4096)) {
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
// Linear-weight quantization group size. MUST match the pte being modelled:
// the export recipe uses quantization.group_size=128 (setup/README.md:247,429;
// specs/036 protocol.md "this box's buffer ptes: 128"). The previous
// hardcoded 32 was the EMBEDDING group (embedding_quantize=4,32) and is a
// different tensor; at 32 the runtime's group_size %% tile_k == 0 check
// (QuantizedLinear.cpp) rejects every tile_k>32 variant, silently falling
// back to the tiled kernel -- 71 of 160 dbuf4 tokens, i.e. the whole
// tile_k in {64,128} subspace. Overridable so a sweep can state it.
int64_t g_group = 128;
constexpr int kWarmupRuns = 3;
constexpr int kTimedRuns = 5;

// Case selection filters (specs/041 tile sweep). A tile-variant token only
// affects ONE (scheme, storage) cell: an ET_VK_Q4GSW_COOPMAT_VARIANT token
// changes 4w+buffer, an ET_VK_DQ8CA_COOPMAT_VARIANT token changes
// 8da4w+buffer. The texture3d rows are the tiled baseline and the other
// scheme is untouched, so running them per token is pure waste -- across a
// 160-token sweep restricting to the affected cell cuts wall clock ~4x.
// Empty string = no filter (the pre-existing all-cases behaviour).
// --regime is the other half of the saving: at M=1 the linear op takes the
// is_gemv short-circuit and dispatches linear_*_coop, NOT the tsweep coopmat
// variant (verified on device: decode rows report dispatch=not_applicable and
// kernel=linear_q4gsw_coop_...). A tile token therefore cannot change a decode
// row at all, so sweeping it over decode measures the same kernel 160 times.
struct CaseFilter {
  std::string model; // substring match on model name
  std::string scheme; // "4w" | "8da4w"
  std::string storage; // "buffer" | "texture3d"
  std::string regime; // "prefill" | "decode"
};

bool regime_selected(const CaseFilter& f, const char* regime) {
  return f.regime.empty() || f.regime == regime;
}

bool model_selected(const CaseFilter& f, const char* model) {
  return f.model.empty() ||
      std::string(model).find(f.model) != std::string::npos;
}

bool scheme_selected(const CaseFilter& f, const char* scheme_label) {
  return f.scheme.empty() || f.scheme == scheme_label;
}

bool storage_selected(const CaseFilter& f, utils::StorageType st) {
  if (f.storage.empty()) {
    return true;
  }
  return f.storage == (st == utils::kTexture3D ? "texture3d" : "buffer");
}

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
// original group sizes; the perf sweep's group_size is g_group.)
std::vector<TestCase> generate_correctness_cases(const CaseFilter& filter) {
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
    if (!scheme_selected(filter, scheme.first)) {
      continue;
    }
    for (const auto& shape : kCorrectnessShapes) {
      LinearConfig cfg{
          shape.M, shape.K, shape.N, shape.group_size, scheme.second};
      for (auto st : {utils::kTexture3D, utils::kBuffer}) {
        if (!storage_selected(filter, st)) {
          continue;
        }
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
  // Rank-3 was buffer-only because coopmat was buffer-only. With texture IO
  // (ET_VK_TEXTURE_COOPMAT) it must also run at texture3d: the rank-2 cases
  // above fall back to tiled at large tiles, so rank-3 is the ONLY correctness
  // case that exercises the texture coopmat epilogue at MMAS_PER_SG_M > 1 --
  // i.e. the dynamically-indexed result[i][j] drain that specs/040 flags as
  // the Xclipse/PAL risk. Validating only small tiles would miss it entirely.
  const utils::StorageType rank3_storage =
      filter.storage == "texture3d" ? utils::kTexture3D : utils::kBuffer;
  for (const auto& scheme : kSchemes) {
    if (!scheme_selected(filter, scheme.first) ||
        !storage_selected(filter, rank3_storage)) {
      continue;
    }
    for (const auto& shape : kRank3CorrectnessShapes) {
      LinearConfig cfg{
          shape.M,
          shape.K,
          shape.N,
          shape.group_size,
          scheme.second,
          shape.batch};
      cases.push_back(make_deterministic_correctness_case(
          cfg, scheme.second, rank3_storage));
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

// Median and CoV of the linear kernel's per-iteration timings, from the same
// ShaderTiming entries linear_kernel_us() averages.
void linear_kernel_dist(
    const BenchmarkResult& r,
    float* median_us,
    float* cov) {
  *median_us = -1.0f;
  *cov = -1.0f;
  std::vector<float> samples;
  for (const auto& st : r.get_shader_timings()) {
    if (st.shader_name.find("linear_") != std::string::npos) {
      samples = st.iter_timings_us;
    }
  }
  if (samples.empty()) {
    return;
  }
  std::vector<float> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const size_t n = sorted.size();
  *median_us =
      (n % 2) ? sorted[n / 2] : 0.5f * (sorted[n / 2 - 1] + sorted[n / 2]);
  double sum = 0.0;
  for (float v : samples) {
    sum += v;
  }
  const double mean = sum / n;
  double var = 0.0;
  for (float v : samples) {
    var += (v - mean) * (v - mean);
  }
  var /= n;
  *cov = mean > 0.0 ? static_cast<float>(std::sqrt(var) / mean) : -1.0f;
}

// Recover (M, K, N) from a case name of the form "..._M<m>_K<k>_N<n>_...".
void parse_mkn_from_case_name(
    const std::string& name,
    int64_t* M,
    int64_t* K,
    int64_t* N) {
  auto grab = [&name](char tag) -> int64_t {
    const std::string needle = std::string("_") + tag;
    for (size_t i = 0; i + needle.size() < name.size(); ++i) {
      if (name.compare(i, needle.size(), needle) != 0) {
        continue;
      }
      size_t j = i + needle.size();
      if (j >= name.size() || !std::isdigit((unsigned char)name[j])) {
        continue;
      }
      int64_t v = 0;
      while (j < name.size() && std::isdigit((unsigned char)name[j])) {
        v = v * 10 + (name[j++] - '0');
      }
      return v;
    }
    return 0;
  };
  *M = grab('M');
  *K = grab('K');
  *N = grab('N');
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
bool run_linear_correctness(const CaseFilter& filter) {
  unsetenv("ET_VK_FORCE_TILED_LINEAR");
  std::vector<BenchmarkResult> results;
  std::vector<std::string> failed_names;
  for (auto& tc : generate_correctness_cases(filter)) {
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
        Record rec;
        rec.suite = "correctness";
        rec.op = tc.name();
        // make_linear_case() builds the name from M/K/N, so recovering them
        // from it keeps the JSON's shape fields populated for correctness
        // rows too -- a consumer cannot tell whether a fallback was legal
        // without knowing the shape the case ran at.
        parse_mkn_from_case_name(tc.name(), &rec.M, &rec.K, &rec.N);
        rec.kernel = linear_kernel(res[0]);
        rec.variant = kernel_class(rec.kernel);
        rec.kernel_us = linear_kernel_us(res[0]);
        linear_kernel_dist(res[0], &rec.kernel_median_us, &rec.kernel_cov);
        const auto st = res[0].get_correctness_status();
        rec.correctness = st == CorrectnessStatus::PASSED ? "PASSED"
            : st == CorrectnessStatus::FAILED             ? "FAILED"
                                                          : "SKIPPED";
        rec.ok = st != CorrectnessStatus::FAILED;
        record_only(rec);
      }
    } catch (const std::exception& e) {
      failed_names.push_back(tc.name());
      Record rec;
      rec.suite = "correctness";
      rec.op = tc.name();
      rec.correctness = "FAILED";
      rec.detail_note = e.what();
      rec.ok = false;
      record_only(rec);
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

// --production-diff: a real element-wise numeric check at the actual 8B
// prefill M=2048 GEMM shapes, which run_linear_correctness's gate never
// covers (et-microbench-correctness-gate-blind-above-256 -- bench_reference
// throws above M/N=256 and those cases are marked SKIPPED, not validated).
// Every shader change promoted to the shipped dq8ca default so far
// (dq8ca-dequant-unpack-ablation Addenda 7-9) has only ever cleared 10+
// consecutive PASSED runs at M/N<=256; this is the check that was missing
// before trusting any of it at production size.
//
// Deliberately dq8ca (8da4w) only, buffer storage only (the shipped coopmat
// dispatch's storage mode) -- matches this investigation's current scope.
// Uses the same well-conditioned deterministic data generator as the
// existing correctness matrix (positive values, no fp16-cancellation noise)
// so a mismatch here means the shader's addressing/arithmetic is wrong at
// this shape, not that the reference itself is numerically unstable.
//
// Cost: O(M*N*K) CPU reference per shape, unthrottled via
// g_allow_large_reference -- on the order of minutes total for all four 8B
// shapes. That is acceptable for a one-shot diagnostic; it must never run as
// part of the default correctness gate or perf sweep.
bool run_production_diff() {
  const LinearModel* model = nullptr;
  for (const auto& m : kLinearModels) {
    if (std::string(m.model) == "llama-3.1-8b") {
      model = &m;
    }
  }
  if (model == nullptr) {
    std::cout << "[production-diff] llama-3.1-8b not found in kLinearModels\n";
    return false;
  }
  g_allow_large_reference = true;
  bool all_ok = true;
  for (const auto& op_shape : model->ops) {
    LinearConfig cfg{
        /*M=*/2048,
        op_shape.K,
        op_shape.N,
        /*group_size=*/g_group,
        /*op_name=*/"linear_dq8ca_q4gsw",
        /*batch=*/0,
        /*model=*/model->model,
        /*regime=*/"prefill",
        /*op_label=*/op_shape.op_label};
    TestCase tc = make_deterministic_correctness_case(
        cfg, "linear_dq8ca_q4gsw", utils::kBuffer);
    std::cout << "[production-diff] " << tc.name()
              << " (M=2048, K=" << op_shape.K << ", N=" << op_shape.N
              << ", group_size=" << g_group << ")\n";
    try {
      auto res = execute_test_cases(
          [&tc]() { return std::vector<TestCase>{tc}; },
          flop_calc,
          "LlamaMicrobenchProductionDiff",
          kWarmupRuns,
          kTimedRuns,
          bench_reference);
      if (res.empty()) {
        std::cout << "[production-diff] " << tc.name()
                  << ": no result produced\n";
        all_ok = false;
        continue;
      }
      const std::string shader_name = linear_kernel(res[0]);
      const bool coopmat_fired =
          shader_name.find("coopmat") != std::string::npos;
      const bool passed =
          res[0].get_correctness_status() == CorrectnessStatus::PASSED;
      all_ok = all_ok && coopmat_fired && passed;
      std::cout << "[production-diff] " << tc.name() << " -> " << shader_name
                << (coopmat_fired ? " (coopmat dispatched)"
                                  : " (NOT coopmat -- fallback, cannot "
                                    "validate the shader under test)")
                << ", correctness="
                << (passed ? "PASSED"
                           : (res[0].get_correctness_status() ==
                                      CorrectnessStatus::FAILED
                                  ? "FAILED (see mismatch detail above)"
                                  : "SKIPPED"))
                << "\n";
    } catch (const std::exception& e) {
      std::cout << "[production-diff] " << tc.name() << " threw: " << e.what()
                << "\n";
      all_ok = false;
    }
  }
  g_allow_large_reference = false;
  std::cout << "[production-diff] " << (all_ok ? "ALL PASSED" : "FAILED")
            << " (4 shapes, M=2048, 8da4w, buffer)\n";
  return all_ok;
}

struct PerfCase {
  LinearConfig cfg;
  utils::StorageType storage;
};
std::vector<PerfCase> generate_linear_perf_cases(const CaseFilter& filter) {
  std::vector<PerfCase> cases;
  for (const auto& scheme : kSchemes) {
    if (!scheme_selected(filter, scheme.first)) {
      continue;
    }
    for (const auto& model : kLinearModels) {
      if (!model_selected(filter, model.model)) {
        continue;
      }
      for (const auto& regime : kLinearRegimes) {
        if (!regime_selected(filter, regime.first)) {
          continue;
        }
        for (const auto& shape : model.ops) {
          LinearConfig cfg{
              regime.second,
              shape.K,
              shape.N,
              g_group,
              scheme.second,
              /*batch=*/1,
              model.model,
              regime.first,
              shape.op_label};
          if (storage_selected(filter, utils::kTexture3D)) {
            cases.push_back({cfg, utils::kTexture3D}); // tiled/gemv baseline
          }
          if (storage_selected(filter, utils::kBuffer)) {
            cases.push_back({cfg, utils::kBuffer}); // coopmat (gate-permitting)
          }
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
void run_linear_suite(const std::string& suite, const CaseFilter& filter) {
  const bool force_tiled = suite == "baseline";
  if (force_tiled) {
    setenv("ET_VK_FORCE_TILED_LINEAR", "1", /*overwrite=*/1);
  } else {
    unsetenv("ET_VK_FORCE_TILED_LINEAR");
  }
  for (const auto& pc : generate_linear_perf_cases(filter)) {
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
        linear_kernel_dist(res[0], &rec.kernel_median_us, &rec.kernel_cov);
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
      // texture3d+coopmat is expected too once ET_VK_TEXTURE_COOPMAT is set;
      // without this the texture runs report unexpected_coopmat and the binary
      // exits nonzero despite dispatching exactly what was asked for.
      static const bool tex_coopmat =
          std::getenv("ET_VK_TEXTURE_COOPMAT") != nullptr;
      const bool expects_coopmat = suite == "linear" &&
          rec.regime == "prefill" &&
          (rec.storage == "buffer" ||
           (tex_coopmat && rec.storage == "texture3d"));
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

// ===================== sdpa correctness =====================
// Coopmat SDPA correctness gate for sdpa_compute_attn_weights_coopmat
// (QK^T) and sdpa_compute_out_coopmat (attn*V). run_sdpa_suite above is
// perf-only -- it never reads output data back or checks it against a
// reference. Like the SDPA perf path (specs/010 Decision 8), this builds
// the graph directly via ComputeGraph rather than the TestCase framework,
// because llama.custom_sdpa.default needs SymInt support the framework
// doesn't have.
//
// Shapes are small and coopmat-tile-aligned (S%128==0, D%64==0 -- SDPA.cpp's
// kSdpaCmQkTileM/kSdpaCmTileN/kSdpaCmTileK/kSdpaCmTileM eligibility check)
// with input_pos=0, so context_len == seq_len and the freshly-written KV
// cache covers exactly the self-attention window under test -- no
// uninitialized cache history to account for. A naive O(S^2*D) fp32
// reference at production S=2048 would be far too slow to run the repeated
// (10+) back-to-back passes a coopmat correctness check needs (today's
// linear-tile-sweep incident: a tile that passed --correctness-only ONCE
// was later found to fail 1-in-10 identical repeat runs, silent wrong
// output, no crash) -- these shapes keep one pass under a second.
struct SdpaCorrectnessCase {
  const char* name;
  int64_t seq_len; // S; also context_len (input_pos=0, fresh prefill)
  int64_t head_dim; // D
  int64_t num_heads; // Q_H
  int64_t num_kv_heads; // KV_H
  // "fast"    -- the cheap pre-check tier, run after any shader edit.
  // "regions" -- the region-coverage tier: the smallest shapes that put a
  //              tile in EVERY QK^T mask region (see sdpa_report_qk_regions).
  //              4x the reference cost of "fast", so it is kept separate to
  //              protect the 10+ repeat discipline (--sdpa-tier).
  const char* tier;
};
const std::vector<SdpaCorrectnessCase> kSdpaCorrectnessCases = {
    // Minimal GQA case: 128 is the smallest legal QK^T M-tile multiple, 64
    // the smallest legal head_dim (both QK^T K-tile and attn*V N-tile).
    {"tiny_gqa", 128, 64, 2, 1, "fast"},
    // 1B's real head configuration (head_dim=64, 32 Q heads, 8 KV heads --
    // kSdpaModels), S truncated from the real 2048 to the same aligned 128
    // so the CPU reference stays fast; this is the shape most likely to
    // exercise a head-indexing (GQA) bug the tiny case's group size of 2
    // could hide.
    {"1b_head_config", 128, 64, 32, 8, "fast"},
    // Region-coverage tier. S=256 is the SMALLEST size that reaches all three
    // QK^T mask regions at the shipped 128x64 tile: it gives num_tiles_m=2,
    // num_tiles_n=4, so
    //   m=0: n=0,1 diagonal      n=2,3 fully masked
    //   m=1: n=0,1 fully visible n=2,3 diagonal
    // which populates every class AND both boundary transitions
    // (visible->diagonal and diagonal->masked). S=192 is not usable: it is
    // not a multiple of WG_TILE_M=128, so SDPA.cpp's alignment gate would
    // refuse to dispatch the coopmat shader at all.
    //
    // Also gives sdpa_out 8 K-chunks (context_len/WG_TILE_K = 256/32) instead
    // of 4, so a double-buffer ping-pong parity error cannot hide in a
    // 2-iteration loop.
    //
    // Cost: the reference is O(S^2*D*Q_H), so these are 4x the "fast" cases
    // (~537 M MACs for the 32-head one against ~134 M) -- seconds per pass,
    // not sub-second. That is why they are a separate tier.
    {"tiny_gqa_s256", 256, 64, 2, 1, "regions"},
    {"1b_head_config_s256", 256, 64, 32, 8, "regions"},
};

// ---------------- QK^T mask-region enumeration (host-side) ----------------
// sdpa_compute_attn_weights_coopmat.glsl classifies each WG tile by where it
// sits relative to the causal diagonal, and runs different code per class.
// A correctness gate only covers a class if some case actually produces a
// tile in it, so enumerate the grid rather than assume (specs
// sdpa-coopmat-causal-mask-paths: "A correctness gate is not treated as
// covering a path its shapes cannot reach").
//
//   fully masked : c_tile_base > s_tile_base + WG_TILE_M - 1 + input_pos
//                  (lowest context index in the tile already exceeds the
//                  highest s + input_pos) -- the shader's tile_all_masked
//   fully visible: c_tile_base + WG_TILE_N - 1 <= s_tile_base + input_pos
//                  (highest context index is within the lowest row's window)
//   diagonal     : neither, so the per-element mask is required
//
// Both fast-path conditions holding at once would require
// WG_TILE_M + WG_TILE_N < 2, so the classification is exhaustive and
// non-overlapping by construction -- `both` below asserts that per tile
// instead of trusting the algebra.
//
// Tile dims mirror SDPA.cpp's kSdpaAttnDefaultDims / the shipped default in
// sdpa_compute_attn_weights_coopmat.yaml. An ET_VK_SDPA_ATTN_COOPMAT_VARIANT
// override would need these updated to match.
constexpr int64_t kSdpaAttnWgTileM = 128;
constexpr int64_t kSdpaAttnWgTileN = 64;

struct SdpaRegionCounts {
  int64_t tiles = 0;
  int64_t all_masked = 0;
  int64_t all_visible = 0;
  int64_t diagonal = 0;
  int64_t both = 0; // overlap; must stay 0
};

// Enumerates the QK^T tile grid for one shape. `verbose` prints the class of
// every tile (task 2.1 wants the per-tile classification, not just totals).
SdpaRegionCounts sdpa_enumerate_qk_regions(
    int64_t S,
    int64_t context_len,
    int64_t input_pos,
    bool verbose,
    int64_t wg_tile_m = kSdpaAttnWgTileM,
    int64_t wg_tile_n = kSdpaAttnWgTileN) {
  SdpaRegionCounts c;
  const int64_t num_tiles_m = (S + wg_tile_m - 1) / wg_tile_m;
  const int64_t num_tiles_n = (context_len + wg_tile_n - 1) / wg_tile_n;
  for (int64_t i = 0; i < num_tiles_m; ++i) {
    for (int64_t j = 0; j < num_tiles_n; ++j) {
      const int64_t s_base = wg_tile_m * i;
      const int64_t c_base = wg_tile_n * j;
      const bool masked = c_base > s_base + wg_tile_m - 1 + input_pos;
      const bool visible = c_base + wg_tile_n - 1 <= s_base + input_pos;
      ++c.tiles;
      const char* label;
      if (masked && visible) {
        ++c.both;
        label = "BOTH(BUG)";
      } else if (masked) {
        ++c.all_masked;
        label = "all_masked";
      } else if (visible) {
        ++c.all_visible;
        label = "all_visible";
      } else {
        ++c.diagonal;
        label = "diagonal";
      }
      if (verbose) {
        std::cout << "[sdpa-regions]   tile m=" << i << " n=" << j
                  << " s_base=" << s_base << " c_base=" << c_base << " -> "
                  << label << "\n";
      }
    }
  }
  return c;
}

// Checks the classification is exhaustive and non-overlapping AT ITS
// BOUNDARIES, which is where an off-by-one actually hides. For a fixed M-tile
// row, increasing the N-tile column must walk the classes in exactly the order
//   [all_visible...] [diagonal...] [all_masked...]
// with no class recurring once left. That single property catches both failure
// modes the spec names: a boundary tile claimed by two classes (`overlap`,
// which the shader's two conditions would have to both accept) and a boundary
// tile claimed by none (which would show up as a class reappearing after the
// run it belongs to). The transition columns themselves are printed, so the
// "last fully-visible / first diagonal" and "last diagonal / first
// fully-masked" pairs are on the record rather than inferred.
bool sdpa_check_region_boundaries(
    int64_t S,
    int64_t context_len,
    int64_t input_pos,
    const char* label) {
  const int64_t num_tiles_m = (S + kSdpaAttnWgTileM - 1) / kSdpaAttnWgTileM;
  const int64_t num_tiles_n =
      (context_len + kSdpaAttnWgTileN - 1) / kSdpaAttnWgTileN;
  bool ok = true;
  for (int64_t i = 0; i < num_tiles_m; ++i) {
    // phase 0 = expecting all_visible, 1 = diagonal, 2 = all_masked
    int phase = 0;
    int64_t last_visible = -1, first_diagonal = -1;
    int64_t last_diagonal = -1, first_masked = -1;
    for (int64_t j = 0; j < num_tiles_n; ++j) {
      const int64_t s_base = kSdpaAttnWgTileM * i;
      const int64_t c_base = kSdpaAttnWgTileN * j;
      const bool masked = c_base > s_base + kSdpaAttnWgTileM - 1 + input_pos;
      const bool visible = c_base + kSdpaAttnWgTileN - 1 <= s_base + input_pos;
      if (masked && visible) {
        std::cout << "[sdpa-boundary] " << label << " m=" << i << " n=" << j
                  << " OVERLAP: satisfies BOTH all_masked and all_visible\n";
        ok = false;
        continue;
      }
      const int cls = masked ? 2 : (visible ? 0 : 1);
      if (cls < phase) {
        std::cout << "[sdpa-boundary] " << label << " m=" << i << " n=" << j
                  << " OUT OF ORDER: class " << cls
                  << " reappeared after phase " << phase
                  << " -- classification is not a clean visible/diagonal/"
                     "masked partition\n";
        ok = false;
      }
      phase = cls > phase ? cls : phase;
      if (cls == 0) {
        last_visible = j;
      } else if (cls == 1) {
        if (first_diagonal < 0) {
          first_diagonal = j;
        }
        last_diagonal = j;
      } else if (first_masked < 0) {
        first_masked = j;
      }
    }
    std::cout << "[sdpa-boundary] " << label << " m=" << i
              << " last_visible=" << last_visible
              << " first_diagonal=" << first_diagonal
              << " last_diagonal=" << last_diagonal
              << " first_masked=" << first_masked;
    // Adjacency: a present transition must be between consecutive columns,
    // i.e. no column is skipped between one class's end and the next's start.
    bool adjacent = true;
    if (last_visible >= 0 && first_diagonal >= 0 &&
        first_diagonal != last_visible + 1) {
      adjacent = false;
    }
    if (last_diagonal >= 0 && first_masked >= 0 &&
        first_masked != last_diagonal + 1) {
      adjacent = false;
    }
    if (!adjacent) {
      std::cout << " NON-ADJACENT TRANSITION (a column is unclassified)";
      ok = false;
    }
    std::cout << "\n";
  }
  return ok;
}

// Prints the region distribution for every correctness case and reports
// whether the gate as a whole reaches all three classes. Returns true iff
// no tile is doubly-classified AND all three classes are covered.
bool sdpa_report_qk_regions(bool verbose, const char* tier = "all") {
  SdpaRegionCounts total;
  bool boundaries_ok = true;
  for (const auto& c : kSdpaCorrectnessCases) {
    if (std::string(tier) != "all" && std::string(c.tier) != tier) {
      continue;
    }
    // input_pos == 0 for every case, so context_len == seq_len.
    const SdpaRegionCounts r =
        sdpa_enumerate_qk_regions(c.seq_len, c.seq_len, 0, verbose);
    std::cout << "[sdpa-regions] " << c.name << " S=" << c.seq_len
              << " context_len=" << c.seq_len << " tile=" << kSdpaAttnWgTileM
              << "x" << kSdpaAttnWgTileN << " num_tiles_m="
              << (c.seq_len + kSdpaAttnWgTileM - 1) / kSdpaAttnWgTileM
              << " num_tiles_n="
              << (c.seq_len + kSdpaAttnWgTileN - 1) / kSdpaAttnWgTileN
              << " tiles=" << r.tiles << " all_masked=" << r.all_masked
              << " all_visible=" << r.all_visible << " diagonal=" << r.diagonal
              << " overlap=" << r.both << "\n";
    total.tiles += r.tiles;
    total.all_masked += r.all_masked;
    total.all_visible += r.all_visible;
    total.diagonal += r.diagonal;
    total.both += r.both;
    boundaries_ok =
        sdpa_check_region_boundaries(c.seq_len, c.seq_len, 0, c.name) &&
        boundaries_ok;
  }
  const bool exhaustive =
      total.all_masked + total.all_visible + total.diagonal == total.tiles;
  const bool covered =
      total.all_masked > 0 && total.all_visible > 0 && total.diagonal > 0;
  std::cout << "[sdpa-regions] TOTAL tiles=" << total.tiles
            << " all_masked=" << total.all_masked
            << " all_visible=" << total.all_visible
            << " diagonal=" << total.diagonal << " overlap=" << total.both
            << " exhaustive=" << (exhaustive ? "yes" : "NO")
            << " all_three_covered=" << (covered ? "yes" : "NO") << "\n";
  if (!covered) {
    std::cout << "[sdpa-regions] UNCOVERED:";
    if (total.all_masked == 0) {
      std::cout << " all_masked";
    }
    if (total.all_visible == 0) {
      std::cout << " all_visible";
    }
    if (total.diagonal == 0) {
      std::cout << " diagonal";
    }
    std::cout << " -- these shader paths are NOT covered by this gate\n";
  }
  std::cout << "[sdpa-regions] boundaries_ok=" << (boundaries_ok ? "yes" : "NO")
            << "\n";
  return total.both == 0 && exhaustive && covered && boundaries_ok;
}

// Causal, GQA-aware fp32 CPU reference. q is [S, Q_H, D], k/v are
// [S, KV_H, D] (row-major, batch=1 squeezed). kv_h = q_h / (Q_H / KV_H),
// matching sdpa_compute_attn_weights_coopmat.glsl's GQA head mapping exactly
// (see that file's header comment). Causal: query s attends to context
// c <= s (input_pos=0).
std::vector<float> sdpa_reference(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    int64_t S,
    int64_t D,
    int64_t Q_H,
    int64_t KV_H) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const int64_t group = Q_H / KV_H;
  std::vector<float> out(static_cast<size_t>(S * Q_H * D), 0.0f);
  std::vector<float> scores(static_cast<size_t>(S));
  for (int64_t h = 0; h < Q_H; ++h) {
    const int64_t kv_h = h / group;
    for (int64_t s = 0; s < S; ++s) {
      float max_score = -std::numeric_limits<float>::infinity();
      for (int64_t c = 0; c <= s; ++c) {
        float acc = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
          acc += q[(s * Q_H + h) * D + d] * k[(c * KV_H + kv_h) * D + d];
        }
        acc *= scale;
        scores[c] = acc;
        max_score = std::max(max_score, acc);
      }
      float denom = 0.0f;
      for (int64_t c = 0; c <= s; ++c) {
        scores[c] = std::exp(scores[c] - max_score);
        denom += scores[c];
      }
      for (int64_t d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int64_t c = 0; c <= s; ++c) {
          acc += (scores[c] / denom) * v[(c * KV_H + kv_h) * D + d];
        }
        out[(s * Q_H + h) * D + d] = acc;
      }
    }
  }
  return out;
}

// Builds+runs one coopmat SDPA case via direct ComputeGraph construction
// (mirrors sdpa_run_case's graph shape), reads Q/K/V/out host-side in
// float/half, computes the CPU reference, and compares. Returns true iff
// BOTH the QK^T and attn*V coopmat shaders actually dispatched (not a
// silent tiled fallback -- tiled is also numerically correct, so a pure
// value comparison alone cannot tell the two apart) AND every output
// element is within tolerance.
// --sdpa-force-fallback: run the SDPA correctness cases with coopmat DISABLED,
// so the tiled shaders serve the same shapes. Two uses:
//  1. it proves the gate's dispatch assertion is load-bearing -- the tiled path
//     is also numerically correct, so a pass here with qk_coopmat=NO must still
//     be reported FAILED, otherwise a silent fallback would look like a pass;
//  2. it is the control that separates "the coopmat shader is wrong" from "the
//     harness/reference/softmax is wrong" when a case fails intermittently.
bool g_sdpa_force_fallback = false;

bool sdpa_correctness_case(const SdpaCorrectnessCase& c) {
  if (g_sdpa_force_fallback) {
    setenv("ET_VK_DISABLE_COOPMAT", "1", 1);
  } else {
    unsetenv("ET_VK_DISABLE_COOPMAT");
  }

  GraphConfig config;
  config.enable_querypool = true;
  api::context()->initialize_querypool();
  ComputeGraph graph(config);

  const int64_t B = 1;
  const std::vector<int64_t> q_sizes = {B, c.seq_len, c.num_heads, c.head_dim};
  const std::vector<int64_t> kv_sizes = {
      B, c.seq_len, c.num_kv_heads, c.head_dim};
  // context_len == seq_len: input_pos=0, so the cache is exactly this step's
  // freshly-written K/V (see file comment above).
  const std::vector<int64_t> cache_sizes = kv_sizes;

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

  const int64_t q_numel = c.seq_len * c.num_heads * c.head_dim;
  const int64_t kv_numel = c.seq_len * c.num_kv_heads * c.head_dim;

  std::vector<float> qf(q_numel), kf(kv_numel), vf(kv_numel);
  std::vector<uint16_t> qh(q_numel), kh(kv_numel), vh(kv_numel);
  for (int64_t i = 0; i < q_numel; ++i) {
    qf[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f - 1.0f;
    qh[i] = float_to_half(qf[i]);
  }
  for (int64_t i = 0; i < kv_numel; ++i) {
    kf[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f - 1.0f;
    kh[i] = float_to_half(kf[i]);
    vf[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f - 1.0f;
    vh[i] = float_to_half(vf[i]);
  }
  graph.maybe_cast_and_copy_into_staging(
      r_q.staging, qh.data(), static_cast<size_t>(q_numel), vkapi::kHalf);
  graph.maybe_cast_and_copy_into_staging(
      r_k.staging, kh.data(), static_cast<size_t>(kv_numel), vkapi::kHalf);
  graph.maybe_cast_and_copy_into_staging(
      r_v.staging, vh.data(), static_cast<size_t>(kv_numel), vkapi::kHalf);

  graph.execute();

  graph.context()->querypool().extract_results();
  const auto shader_results =
      graph.context()->querypool().get_shader_timestamp_data();
  std::vector<std::string> dispatched;
  for (const auto& r : shader_results) {
    dispatched.push_back(r.kernel_name);
  }
  const bool qk_fired =
      has_kernel_containing(dispatched, "sdpa_compute_attn_weights_coopmat");
  const bool av_fired =
      has_kernel_containing(dispatched, "sdpa_compute_out_coopmat");

  std::vector<uint16_t> outh(q_numel);
  graph.maybe_cast_and_copy_from_staging(
      graph.outputs()[0].staging,
      outh.data(),
      static_cast<size_t>(q_numel),
      vkapi::kHalf);
  std::vector<float> outf(q_numel);
  for (int64_t i = 0; i < q_numel; ++i) {
    outf[i] = half_to_float(outh[i]);
  }

  const std::vector<float> ref = sdpa_reference(
      qf, kf, vf, c.seq_len, c.head_dim, c.num_heads, c.num_kv_heads);

  int64_t mismatches = 0;
  int64_t first_mismatch = -1;
  const float abs_tol = 0.03f;
  const float rel_tol = 0.05f;
  for (int64_t i = 0; i < q_numel; ++i) {
    const float diff = std::fabs(outf[i] - ref[i]);
    const float thresh = abs_tol + rel_tol * std::fabs(ref[i]);
    if (diff > thresh) {
      ++mismatches;
      if (first_mismatch < 0) {
        first_mismatch = i;
      }
    }
  }

  const bool numeric_ok = mismatches == 0;
  const bool fired_ok = qk_fired && av_fired;
  std::cout << "[sdpa-correctness] " << c.name << " S=" << c.seq_len
            << " D=" << c.head_dim << " Q_H=" << c.num_heads
            << " KV_H=" << c.num_kv_heads
            << " qk_coopmat=" << (qk_fired ? "yes" : "NO")
            << " av_coopmat=" << (av_fired ? "yes" : "NO")
            << " mismatches=" << mismatches << "/" << q_numel;
  if (!numeric_ok) {
    std::cout << " (first at " << first_mismatch << ": got=" << std::fixed
              << std::setprecision(4) << outf[first_mismatch]
              << " ref=" << ref[first_mismatch] << ")";
  }
  std::cout << (numeric_ok && fired_ok ? " PASSED" : " FAILED") << "\n";
  unsetenv("ET_VK_DISABLE_COOPMAT"); // restore the tree's default-on state
  return numeric_ok && fired_ok;
}

// Runs every case in kSdpaCorrectnessCases once and reports pass/fail.
// Callers that want the repeated-pass discipline this coopmat bug class
// requires (see file comment above) should invoke this function itself
// multiple times in a loop -- kept a single pass per call, like
// run_linear_correctness, so a driver script controls the rep count and can
// distinguish "which specific repeat failed."
bool run_sdpa_correctness(const char* tier = "all") {
  bool all_ok = true;
  // Report which QK^T mask regions these shapes actually reach before running
  // them, so a pass is never mistaken for coverage of a path no case produces
  // a tile for. Non-verbose: --sdpa-regions-only prints the per-tile detail.
  const bool regions_covered = sdpa_report_qk_regions(/*verbose=*/false, tier);
  if (!regions_covered) {
    std::cout << "[sdpa-correctness] WARNING: tier=" << tier
              << " does not cover every QK^T mask-region path (see "
                 "[sdpa-regions] above)\n";
  }
  int64_t ran = 0;
  for (const auto& c : kSdpaCorrectnessCases) {
    if (std::string(tier) != "all" && std::string(c.tier) != tier) {
      continue;
    }
    ++ran;
    all_ok = sdpa_correctness_case(c) && all_ok;
  }
  std::cout << "[sdpa-correctness] tier=" << tier << " cases_run=" << ran
            << "\n";
  if (ran == 0) {
    std::cout << "[sdpa-correctness] no case matched tier '" << tier
              << "' -- treating as failure rather than a silent pass\n";
    return false;
  }
  return all_ok;
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
// --- additive machine-readable output (--json) ----------------------------
//
// The text report is for a human reading a terminal; this is for the L2 stage
// of an automated ladder, which needs per-case median, CoV, GFLOP/s, the
// kernel that actually dispatched, and the correctness verdict, all keyed so a
// candidate can be matched to the variant it was selected as.
//
// Deliberately additive: it prints nothing unless --json is given, changes no
// existing line, and computes nothing the text path did not already compute.

std::string json_escape(const std::string& in) {
  std::string out;
  for (char c : in) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

// JSON has no NaN or Infinity, and the -1 sentinels mean "not applicable"
// rather than "minus one"; both become null so a consumer cannot mistake
// either for a measurement.
std::string json_num(float v) {
  if (std::isnan(v) || std::isinf(v) || v < 0.0f) {
    return "null";
  }
  std::ostringstream o;
  o << v;
  return o.str();
}

void print_json_report(std::ostream& out) {
  out << "{\n  \"schema\": \"test_llama_microbench.v1\"";
  {
    const auto* adapter = api::context()->adapter_ptr();
    out << ",\n  \"device\": \"" << json_escape(adapter->device_name()) << "\""
        << ",\n  \"subgroup_size\": " << adapter->subgroup_size()
        << ",\n  \"timestamp_period_ns\": " << adapter->timestamp_period()
        << ",\n  \"cooperative_matrix\": "
        << (adapter->supports_cooperative_matrix() ? "true" : "false");
  }
  out << ",\n  \"warmup_runs\": " << kWarmupRuns
      << ",\n  \"timed_runs\": " << kTimedRuns
      << ",\n  \"group_size\": " << g_group << ",\n  \"cases\": [\n";
  for (size_t i = 0; i < g_records.size(); ++i) {
    const Record& r = g_records[i];
    out << "    {" << "\"suite\": \"" << json_escape(r.suite) << "\""
        << ", \"model\": \"" << json_escape(r.model) << "\""
        << ", \"scheme\": \"" << json_escape(r.scheme) << "\""
        << ", \"regime\": \"" << json_escape(r.regime) << "\"" << ", \"op\": \""
        << json_escape(r.op) << "\"" << ", \"storage\": \""
        << json_escape(r.storage) << "\"" << ", \"variant\": \""
        << json_escape(r.variant) << "\"" << ", \"kernel\": \""
        << json_escape(r.kernel) << "\"" << ", \"M\": " << r.M
        << ", \"K\": " << r.K << ", \"N\": " << r.N
        << ", \"kv_heads\": " << r.kv
        << ", \"op_mean_us\": " << json_num(r.mean_us)
        << ", \"op_stdev_us\": " << json_num(r.stdev_us)
        << ", \"kernel_mean_us\": " << json_num(r.kernel_us)
        << ", \"kernel_median_us\": " << json_num(r.kernel_median_us)
        << ", \"kernel_cov\": " << json_num(r.kernel_cov)
        << ", \"gflops\": " << json_num(r.gflops) << ", \"dispatch\": \""
        << json_escape(r.dispatch) << "\"" << ", \"correctness\": \""
        << json_escape(r.correctness) << "\"" << ", \"detail\": \""
        << json_escape(r.detail_note) << "\""
        << ", \"ok\": " << (r.ok ? "true" : "false") << "}"
        << (i + 1 < g_records.size() ? "," : "") << "\n";
  }
  out << "  ]\n}\n";
}

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
         "  --scheme=<4w|8da4w>  only this quantization scheme\n"
         "  --storage=<buffer|texture3d>  only this storage type\n"
         "  --regime=<prefill|decode>     only this regime\n"
         "  --group-size=<N>     linear quant group size (default 128,\n"
         "                       must match the pte; tile_k must divide it)\n"
         "                       (--scheme/--storage also narrow the\n"
         "                        correctness gate; a tile-variant token\n"
         "                        only affects one scheme+buffer cell)\n"
         "  --correctness-only   run just the linear correctness matrix\n"
         "  --sdpa-correctness-only  run just the SDPA coopmat correctness "
         "cases\n"
         "  --sdpa-regions-only  enumerate the QK^T mask-region tile grid "
         "(no GPU)\n"
         "  --sdpa-tier=<fast|regions|all>  which SDPA correctness tier to "
         "run (default all)\n"
         "  --sdpa-force-fallback  run SDPA correctness with coopmat "
         "DISABLED (control)\n"
         "  --skip-correctness   skip the correctness gate before perf\n"
         "  --list               print every case with its sizes, no GPU\n"
         "  --help               this message\n";
}

void list_cases(
    bool linear,
    bool baseline,
    bool sdpa,
    const CaseFilter& filter) {
  const std::string& model_filter = filter.model;
  int n = 0;
  for (const char* suite : {"linear", "baseline"}) {
    if ((std::string(suite) == "linear" && !linear) ||
        (std::string(suite) == "baseline" && !baseline)) {
      continue;
    }
    for (const auto& pc : generate_linear_perf_cases(filter)) {
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
  bool correctness_only = false, sdpa_correctness_only = false;
  bool sdpa_regions_only = false;
  std::string sdpa_tier = "all";
  bool skip_correctness = false, list_only = false;
  bool production_diff = false;
  // Additive machine-readable output. Absent, every existing line is byte
  // for byte what it was.
  bool json_out = false;
  std::string json_path;
  CaseFilter filter;
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
    } else if (arg == "--sdpa-correctness-only") {
      sdpa_correctness_only = true;
    } else if (arg == "--sdpa-regions-only") {
      sdpa_regions_only = true;
    } else if (arg.rfind("--sdpa-tier=", 0) == 0) {
      sdpa_tier = arg.substr(std::string("--sdpa-tier=").size());
    } else if (arg == "--sdpa-force-fallback") {
      g_sdpa_force_fallback = true;
    } else if (arg == "--production-diff") {
      production_diff = true;
    } else if (arg == "--skip-correctness") {
      skip_correctness = true;
    } else if (arg == "--list") {
      list_only = true;
    } else if (arg == "--json") {
      json_out = true;
    } else if (arg.rfind("--json-out=", 0) == 0) {
      json_out = true;
      json_path = arg.substr(11);
    } else if (arg.rfind("--model=", 0) == 0) {
      filter.model = arg.substr(8);
    } else if (arg.rfind("--scheme=", 0) == 0) {
      filter.scheme = arg.substr(9);
      if (filter.scheme != "4w" && filter.scheme != "8da4w") {
        std::cerr << "--scheme must be 4w or 8da4w, got: " << filter.scheme
                  << "\n";
        return 2;
      }
    } else if (arg.rfind("--storage=", 0) == 0) {
      filter.storage = arg.substr(10);
      if (filter.storage != "buffer" && filter.storage != "texture3d") {
        std::cerr << "--storage must be buffer or texture3d, got: "
                  << filter.storage << "\n";
        return 2;
      }
    } else if (arg.rfind("--group-size=", 0) == 0) {
      g_group = std::stoll(arg.substr(13));
    } else if (arg.rfind("--regime=", 0) == 0) {
      filter.regime = arg.substr(9);
      if (filter.regime != "prefill" && filter.regime != "decode") {
        std::cerr << "--regime must be prefill or decode, got: "
                  << filter.regime << "\n";
        return 2;
      }
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
    list_cases(linear, baseline, sdpa, filter);
    return 0;
  }

  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "Llama microbench (3.1 8B / 3.2 3B / 3.2 1B real e2e shapes; "
               "prefill 2048 / decode 1 @ ctx3072; linear group_size="
            << g_group << "; " << kWarmupRuns << " warmup + " << kTimedRuns
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
              << ",max_shared_mem_bytes="
              << adapter->max_compute_shared_memory_size() << "\n";
  }
  print_separator();

  std::srand(0);
  bool ok = true;

  // Correctness gate: validates the tiled and coopmat linear kernels
  // (including the rank-3 dispatch check) before any perf time is spent.
  auto finish_correctness = [&](bool ok) {
    if (json_out) {
      if (json_path.empty()) {
        print_json_report(std::cout);
      } else {
        std::ofstream jf(json_path);
        if (jf) {
          print_json_report(jf);
        }
      }
    }
    return ok ? 0 : 1;
  };
  if (correctness_only) {
    return finish_correctness(run_linear_correctness(filter));
  }
  if (sdpa_regions_only) {
    return finish_correctness(
        sdpa_report_qk_regions(/*verbose=*/true, sdpa_tier.c_str()));
  }
  if (sdpa_correctness_only) {
    return finish_correctness(run_sdpa_correctness(sdpa_tier.c_str()));
  }
  if (production_diff) {
    return finish_correctness(run_production_diff());
  }
  if ((linear || baseline) && !skip_correctness) {
    if (!run_linear_correctness(filter)) {
      std::cout << "correctness gate FAILED -- not running the perf sweep\n";
      // Emit the JSON before returning. Without this the consumer sees no
      // document at all and cannot distinguish a correctness failure from a
      // crash -- which are different verdicts with different handling: one
      // rejects the candidate, the other quarantines it and attempts device
      // recovery.
      if (json_out) {
        if (json_path.empty()) {
          print_json_report(std::cout);
        } else {
          std::ofstream jf(json_path);
          if (jf) {
            print_json_report(jf);
          }
        }
      }
      return 1;
    }
  }

  if (linear) {
    run_linear_suite("linear", filter);
  }
  if (baseline) {
    run_linear_suite("baseline", filter);
  }
  bool sdpa_confirmed = true;
  if (sdpa) {
    sdpa_confirmed = run_sdpa_suite(filter.model);
  }

  print_report(baseline);

  if (json_out) {
    if (json_path.empty()) {
      print_json_report(std::cout);
    } else {
      std::ofstream jf(json_path);
      if (!jf) {
        std::cerr << "could not open " << json_path << " for --json-out\n";
        return 2;
      }
      print_json_report(jf);
    }
  }

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
