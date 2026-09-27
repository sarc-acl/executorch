/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SARC selection rows for NVIDIA GPUs. First matching row wins, so each
// device lists its preferred tile first and the fallback tile after. Status
// kVerified requires sarc/tools/verify.sh evidence on that device.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

namespace vkcompute {
namespace sarc {
namespace {

// {m, n, k, sg_grid_x, sg_grid_y, subgroup, mma_m, csh_in_ash}
constexpr TileDims kT256x128k16g42 = {256, 128, 16, 4, 2, 32, 16, false};
constexpr TileDims kT128x128k16g24 = {128, 128, 16, 2, 4, 32, 16, false};
constexpr TileDims kT128x256k16g42 = {128, 256, 16, 4, 2, 32, 16, false};
constexpr TileDims kT128x128k16g42 = {128, 128, 16, 4, 2, 32, 16, false};
constexpr TileDims kT128x128k32g42 = {128, 128, 32, 4, 2, 32, 16, false};
constexpr TileDims kT256x128k16g22 = {256, 128, 16, 2, 2, 32, 16, false};
constexpr TileDims kT128x128k16g22 = {128, 128, 16, 2, 2, 32, 16, false};
// dq8ca texture3d variant drains through Ash_int8 (CSH_IN_ASH).
constexpr TileDims kT128x128k64g44 = {128, 128, 64, 4, 4, 32, 16, true};

// RTX 4070 Ti SUPER: the large tiles only pay off beyond N = 512.
bool wide_n(const ShapeInfo& s) {
  return s.N > 512;
}

// Jetson Orin: only the 2048-token prefill projections of Llama 3.2 1B/3B and
// 3.1 8B at group size 128 were measured (tools/jetson-study); everything
// else keeps the stock kernels.
bool orin_measured(const ShapeInfo& s) {
  const int64_t K = s.K;
  const int64_t N = s.N;
  const bool projection = (K == 2048 && (N == 512 || N == 2048 || N == 8192)) ||
      (K == 3072 && (N == 1024 || N == 3072 || N == 8192)) ||
      (K == 4096 && (N == 1024 || N == 4096 || N == 14336)) ||
      (K == 8192 && (N == 2048 || N == 3072)) || (K == 14336 && N == 4096);
  return projection && s.group_size == 128 && s.M <= 2048;
}
// fp16 accumulation loses accuracy beyond K = 8192: use fp32 there.
bool orin_large_k(const ShapeInfo& s) {
  return orin_measured(s) && s.K > 8192;
}
bool orin_256(const ShapeInfo& s) {
  return orin_measured(s) && s.K <= 8192 && (s.M != 256 || s.N % 2048 == 0);
}
bool orin_default(const ShapeInfo& s) {
  return orin_measured(s) && s.K <= 8192;
}

const Row kNvidiaRows[] = {
    // RTX 4070 Ti SUPER (Ada). fp16 MMA per quantization group plus an fp32
    // total ('ga'). Verified on release 1.5 (sarc-1.5-4w-port): prefill
    // 1B/3B/8B 19692/8790/4491 vs stock 1.5 6850/2557/1111 tok/s.
    {"4070 ti super", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t256x128k16g42s32ga", kT256x128k16g42,
     kTex3dTex2d, wide_n, Status::kVerified},
    {"4070 ti super", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x128k16g24s32ga", kT128x128k16g24,
     kTex3dTex2d, nullptr, Status::kVerified},
    {"4070 ti super", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x256k16g42s32ga", kT128x256k16g42,
     kBufTex2d, wide_n, Status::kVerified},
    {"4070 ti super", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x128k16g42s32ga", kT128x128k16g42,
     kBufTex2d | kBufBuf, nullptr, Status::kVerified},

    // Jetson Orin (Ampere iGPU), texture3d only. Verified on release 1.5
    // (sarc-1.5-4w-port): prefill 1B/3B/8B 890/361/190 vs stock 1.5
    // 229/83/35 tok/s. Buffer IO keeps the upstream path (no rows).
    {"tegra orin", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x128k32g42s32f32", kT128x128k32g42,
     kTex3dTex2d, orin_large_k, Status::kVerified},
    {"tegra orin", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t256x128k16g22s32", kT256x128k16g22,
     kTex3dTex2d, orin_256, Status::kVerified},
    {"tegra orin", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x128k16g22s32", kT128x128k16g22,
     kTex3dTex2d, orin_default, Status::kVerified},

    // 8da4w: zpgtr (row-major activations), MMA 16x16x32, K 64, raw A
    // staging, paired B. 1.4 branches -4070ti/-jetson: 1B prefill vs tiled
    // 4070 Ti SUPER 6942 -> 21558 tok/s.
    {"4070 ti super", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_t128x128k64g44s32mk32ra", kT128x128k64g44,
     kTex3dTex2d | kBufTex2d, nullptr, Status::kUnverified,
     /*rowmajor_a=*/true},
    {"tegra orin", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_t128x128k64g44s32mk32ra", kT128x128k64g44,
     kTex3dTex2d, orin_measured, Status::kUnverified,
     /*rowmajor_a=*/true},
};

struct Registrar {
  Registrar() {
    register_rows(kNvidiaRows, sizeof(kNvidiaRows) / sizeof(kNvidiaRows[0]));
  }
} registrar;

} // namespace
} // namespace sarc
} // namespace vkcompute
