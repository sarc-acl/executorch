/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

namespace vkcompute {
namespace sarc {
namespace {

const Row kArmRows[] = {
    // Mali-G1-Ultra MC12 (vivo V2502A, MT6993, driver r54p1): subgroup 16 fixed,
    // fp16 MMA 16x32x32, 32 KiB LDS. 8B 4w prefill shapes (microbench,
    // 2026-09-28): faster than the tiled kernel and within the production-diff
    // tolerance (the tiled kernel's fp16 accumulation is not). No 8da4w row:
    // the only int8 shape is 4x16x16 and there is no 4x16 fp32 accumulator.
    {"mali-g1",
     nullptr,
     Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t64x128k32g44s16m16x32x32gahb",
     {64, 128, 32, 4, 4, 16, 16, false},
     kTex3dTex2d | kBufTex2d | kBufBuf,
     nullptr,
     Status::kUnverified},
};

struct Registrar {
  Registrar() {
    register_rows(kArmRows, sizeof(kArmRows) / sizeof(kArmRows[0]));
  }
} registrar;

} // namespace
} // namespace sarc
} // namespace vkcompute
