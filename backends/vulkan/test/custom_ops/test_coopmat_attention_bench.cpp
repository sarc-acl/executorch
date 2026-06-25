// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// fp16 attention-matmul TILE SWEEP microbenchmark at Llama prefill shapes
// (M=seq=2048). The two SDPA prefill matmuls are plain fp16 GEMMs:
//   QK*K^T : per head, out[s,c] = sum_d Q[s,d]*K[c,d]  -> M=seq, K=head_dim, N=seq
//   attn*V : per head, out[s,d] = sum_c P[s,c]*V[c,d]  -> M=seq, K=seq, N=head_dim
// Feeding mat2 as row-major [K,N] already is K^T's layout for QK*K^T, so the
// A*B the harness runs equals Q*K^T and the fp32 reference matches.
//
// Sweep: for each of the 4 attention shapes, run the matmul_coopmat shader at
// several tile geometries (coopmat_mm.yaml variants) and report GFLOP/s. The
// "tiled" matmul_vec proxy is included as a baseline column. A (shape,tile)
// combo is skipped (N/A) when the shape doesn't divide the tile (the coopmat
// shader has no partial-tile handling):
//   M % WG_TILE_M == 0, N % WG_TILE_N == 0, K % WG_TILE_K == 0.
// The narrow attn*V N=64 shapes therefore skip all WG_TILE_N=128 tiles.
//
// Each tile dispatches via test_etvk.test_mm with impl_selector "coopmat:<tile>"
// -> add_matmul_coopmat_node(tile_variant), which launches the per-variant
// workgroup size (= SG_GRID_X*SG_GRID_Y*SUBGROUP_SIZE). Correctness is validated
// on small aligned shapes against an fp32 reference before trusting the perf
// numbers (a wrong WG-size -> grid-stride staging overrun -> garbage output).
//
// M defaults to 2048 (real prefill batch) and is overridable with
// COOPMAT_BENCH_M.

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "utils.h"

using namespace executorch::vulkan::prototyping;
using namespace vkcompute;

struct GemmConfig {
  int64_t M;
  int64_t K;
  int64_t N;
};

// Tile geometries swept (must match coopmat_mm.yaml variant names + dims).
// suffix "" is the bare 64x64x32 baseline (matmul_coopmat); the rest pass a
// tile variant string through impl_selector "coopmat:<suffix>".
struct Tile {
  const char* label; // column header
  const char* suffix; // "" baseline, else "tNxMxK" variant suffix
  int64_t wg_m;
  int64_t wg_n;
  int64_t wg_k;
};
static const std::vector<Tile> kTiles = {
    {"t64x64x32", "t64x64x32", 64, 64, 32},
    {"t128x64x32", "t128x64x32", 128, 64, 32},
    {"t64x128x32", "t64x128x32", 64, 128, 32},
    {"t128x128x32", "t128x128x32", 128, 128, 32},
    {"t128x64x16", "t128x64x16", 128, 64, 16},
};

static bool tile_fits(const Tile& t, int64_t M, int64_t K, int64_t N) {
  return M % t.wg_m == 0 && N % t.wg_n == 0 && K % t.wg_k == 0;
}

// One TestCase. impl is "tiled" (baseline column) or "coopmat:<suffix>" / a bare
// tile suffix that gets prefixed with "coopmat:". Empty suffix -> bare coopmat.
static TestCase make_case(
    const GemmConfig& cfg,
    const std::string& impl_selector,
    const std::string& name_tag) {
  const vkapi::ScalarType dt = vkapi::kHalf;
  const bool is_tiled = (impl_selector == "tiled");
  const utils::StorageType storage =
      is_tiled ? utils::kTexture3D : utils::kBuffer;
  const std::string storage_str = is_tiled ? "Texture3D" : "Buffer";

  TestCase tc;
  tc.set_name(
      "attn_mm_" + name_tag + "_M" + std::to_string(cfg.M) + "_K" +
      std::to_string(cfg.K) + "_N" + std::to_string(cfg.N) + "_" + storage_str);

  ValueSpec mat1({cfg.M, cfg.K}, dt, storage, utils::kWidthPacked,
                 DataGenType::RANDOM);
  ValueSpec mat2({cfg.K, cfg.N}, dt, storage, utils::kWidthPacked,
                 DataGenType::RANDOM);
  ValueSpec output({cfg.M, cfg.N}, dt, storage, utils::kWidthPacked,
                   DataGenType::ZEROS);

  tc.set_operator_name("test_etvk.test_mm.default");
  tc.add_input_spec(mat1);
  tc.add_input_spec(mat2);
  tc.add_input_spec(ValueSpec::make_string(impl_selector));
  tc.add_output_spec(output);

  // tiled accumulates in fp16 (error grows with K); coopmat accumulates in fp32,
  // bounded by fp16 input/output rounding only.
  if (is_tiled) {
    tc.set_abs_tolerance(1.0f);
    tc.set_rel_tolerance(1e-1f);
  } else {
    tc.set_abs_tolerance(0.5f);
    tc.set_rel_tolerance(5e-2f);
  }
  tc.set_shader_filter({"nchw_to", "to_nchw", "view_copy"});
  return tc;
}

// CPU fp32 reference from the fp16 inputs; oversized (perf) shapes throw and
// the framework marks them SKIPPED.
static void bench_reference(TestCase& tc) {
  const ValueSpec& a = tc.inputs()[0];
  const ValueSpec& b = tc.inputs()[1];
  ValueSpec& out = tc.outputs()[0];
  const auto as = a.get_tensor_sizes();
  const int64_t M = as[0], K = as[1];
  const int64_t N = out.get_tensor_sizes()[1];
  if (M > 256 || K > 256 || N > 256) {
    throw std::invalid_argument("ref: too big");
  }
  const auto& ah = a.get_half_data();
  const auto& bh = b.get_half_data();
  auto& ref = out.get_ref_float_data();
  ref.resize(M * N);
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        acc += half_to_float(ah[m * K + k]) * half_to_float(bh[k * N + n]);
      }
      ref[m * N + n] = acc;
    }
  }
}

// The four distinct per-head SDPA prefill matmul shapes across the 3 models.
// 8B and 3B share per-head shapes (head_dim=128); 1B uses head_dim=64.
struct AttnShape {
  const char* label;
  int64_t K;
  int64_t N;
};
static const std::vector<AttnShape> kShapes = {
    {"QK*K^T  D=128 [8B,3B]", 128, 2048},
    {"QK*K^T  D=64  [1B]", 64, 2048},
    {"attn*V  D=128 [8B,3B]", 2048, 128},
    {"attn*V  D=64  [1B]", 2048, 64},
};

static int64_t bench_M() {
  if (const char* e = std::getenv("COOPMAT_BENCH_M")) {
    const int64_t v = std::atoll(e);
    if (v > 0) {
      return v;
    }
  }
  return 2048; // real prefill seq length
}

// Correctness shapes: small + aligned to every tile we sweep. Each must divide
// some tile's M/N/K; we run all tiles that fit per shape and verify against the
// fp32 reference. 128x64 covers M=128, N=64, K-step 16/32. 256x128 covers the
// 128x128 / 128xN tiles and a 2x2 workgroup grid.
static const std::vector<GemmConfig> kCorrectnessShapes = {
    {128, 64, 128},
    {256, 128, 256},
};

std::vector<TestCase> generate_cases() {
  std::vector<TestCase> cases;
  const int64_t M = bench_M();
  const bool correctness_only =
      std::getenv("COOPMAT_BENCH_CORRECTNESS_ONLY") != nullptr;

  // Perf sweep: per shape, the "tiled" baseline then each fitting tile.
  if (!correctness_only) {
    for (const auto& s : kShapes) {
      cases.push_back(make_case({M, s.K, s.N}, "tiled", "tiled"));
      for (const auto& t : kTiles) {
        if (!tile_fits(t, M, s.K, s.N)) {
          continue; // N/A — shape doesn't divide this tile
        }
        cases.push_back(make_case(
            {M, s.K, s.N}, std::string("coopmat:") + t.suffix, t.label));
      }
    }
  }

  // Correctness: every fitting tile per small aligned shape (+ tiled baseline).
  for (const auto& cfg : kCorrectnessShapes) {
    cases.push_back(make_case(cfg, "tiled", "tiled"));
    for (const auto& t : kTiles) {
      if (!tile_fits(t, cfg.M, cfg.K, cfg.N)) {
        continue;
      }
      cases.push_back(
          make_case(cfg, std::string("coopmat:") + t.suffix, t.label));
    }
  }
  return cases;
}

int64_t flop_calc(const TestCase& tc) {
  const auto& in = tc.inputs()[0].get_tensor_sizes();
  const auto& out = tc.outputs()[0].get_tensor_sizes();
  return 2 * in[0] * in[1] * out[1];
}

int main() {
  set_debugging(false);
  set_print_output(false);
  set_print_latencies(false);
  set_use_gpu_timestamps(true);

  const int64_t M = bench_M();
  print_performance_header();
  std::cout << "fp16 attention-matmul TILE SWEEP: matmul_coopmat tile geometries "
               "vs tiled baseline (Llama 8B/3B/1B per-head shapes, M=seq="
            << M << ")" << std::endl;
  print_separator();

  auto results = execute_test_cases(
      generate_cases, flop_calc, "AttnTileSweep",
      /*warmup=*/3, /*runs=*/5, /*reference=*/bench_reference);

  // Count expected perf cases (per shape: 1 tiled + the tiles that fit).
  size_t expected_perf = 0;
  for (const auto& s : kShapes) {
    expected_perf += 1; // tiled
    for (const auto& t : kTiles) {
      if (tile_fits(t, M, s.K, s.N)) {
        expected_perf++;
      }
    }
  }
  if (results.size() < expected_perf) {
    return 0; // correctness-only run
  }

  auto gflops = [](float time_us, int64_t M, int64_t K, int64_t N) -> float {
    return time_us > 0 ? (2.0f * M * N * K) / (time_us * 1e3f) : 0.0f;
  };

  // ---- Table: rows = shapes, columns = tiled + each tile geometry ----
  std::cout << "\n========== SUMMARY: fp16 attn-matmul TILE SWEEP GFLOP/s (M="
            << M << ") ==========\n";
  std::cout << "(N/A = shape does not divide the tile; the coopmat shader has "
               "no partial-tile handling)\n\n";

  std::cout << std::left << std::setw(24) << "shape (K,N)" << std::right
            << std::setw(9) << "tiled";
  for (const auto& t : kTiles) {
    std::cout << std::setw(12) << t.label;
  }
  std::cout << "   best-tile\n";

  size_t idx = 0;
  for (const auto& s : kShapes) {
    const float tiled = gflops(results[idx++].get_avg_time_us(), M, s.K, s.N);
    // Read each fitting tile in kTiles order; N/A for skipped ones.
    std::vector<float> tile_gf(kTiles.size(), -1.0f);
    for (size_t ti = 0; ti < kTiles.size(); ++ti) {
      if (tile_fits(kTiles[ti], M, s.K, s.N)) {
        tile_gf[ti] = gflops(results[idx++].get_avg_time_us(), M, s.K, s.N);
      }
    }
    // best tile
    float best = -1.0f;
    std::string best_label = "-";
    for (size_t ti = 0; ti < kTiles.size(); ++ti) {
      if (tile_gf[ti] > best) {
        best = tile_gf[ti];
        best_label = kTiles[ti].label;
      }
    }
    std::cout << std::left << std::setw(24)
              << (std::string(s.label) + " (" + std::to_string(s.K) + "," +
                  std::to_string(s.N) + ")")
              << std::right << std::fixed << std::setprecision(1)
              << std::setw(9) << tiled;
    for (size_t ti = 0; ti < kTiles.size(); ++ti) {
      if (tile_gf[ti] < 0.0f) {
        std::cout << std::setw(12) << "N/A";
      } else {
        std::cout << std::setw(12) << tile_gf[ti];
      }
    }
    std::cout << "   " << best_label << " (" << std::setprecision(1) << best
              << ")\n";
  }
  std::cout << "\nbest-tile column = highest-GFLOP/s coopmat tile for that "
               "shape (coopmat vs the tiled baseline shown in col 1).\n";
  return 0;
}
