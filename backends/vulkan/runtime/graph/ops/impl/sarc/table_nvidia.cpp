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
    // total ('ga'). 1.4 study: 1B 4w prefill vs tiled 5626 -> 20078 tok/s.
    {"4070 ti super", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t256x128k16g42s32ga", kT256x128k16g42,
     kTex3dTex2d, wide_n, Status::kUnverified},
    {"4070 ti super", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x128k16g24s32ga", kT128x128k16g24,
     kTex3dTex2d, nullptr, Status::kUnverified},
    {"4070 ti super", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x256k16g42s32ga", kT128x256k16g42,
     kBufTex2d, wide_n, Status::kUnverified},
    {"4070 ti super", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x128k16g42s32ga", kT128x128k16g42,
     kBufTex2d | kBufBuf, nullptr, Status::kUnverified},

    // Jetson Orin (Ampere iGPU), texture3d only. Study:
    // igpu-roofline docs/JETSON-WMMA-LESSONS.md.
    {"tegra orin", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x128k32g42s32f32", kT128x128k32g42,
     kTex3dTex2d, orin_large_k, Status::kUnverified},
    {"tegra orin", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t256x128k16g22s32", kT256x128k16g22,
     kTex3dTex2d, orin_256, Status::kUnverified},
    {"tegra orin", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x128k16g22s32", kT128x128k16g22,
     kTex3dTex2d, orin_default, Status::kUnverified},
};

struct Registrar {
  Registrar() {
    register_rows(kNvidiaRows, sizeof(kNvidiaRows) / sizeof(kNvidiaRows[0]));
  }
} registrar;

} // namespace
} // namespace sarc
} // namespace vkcompute
