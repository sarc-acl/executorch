/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SARC selection rows for Qualcomm Adreno GPUs. Status kVerified requires
// sarc/tools/verify.sh evidence on the device (Android mode).

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

namespace vkcompute {
namespace sarc {
namespace {

const Row kQualcommRows[] = {
    // Adreno 840 (S26): fp16 MMA 64x32x16 (Adreno exposes fp16 only at
    // M = 64), f16vec4 LDS staging. 1.4 branch -qualcomm: 1B 4w prefill
    // ~757 tok/s vs ~712 for the upstream GEMM. The 1.4 int8 (8da4w) kernel
    // is not ported (wrong output, then DEVICE_LOST on that branch).
    {"adreno",
     nullptr,
     Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t64x64k32g21s64m64x32x16",
     {64, 64, 32, 2, 1, 64, 64, false},
     kTex3dTex2d | kBufTex2d | kBufBuf,
     nullptr,
     Status::kUnverified},
};

struct Registrar {
  Registrar() {
    register_rows(
        kQualcommRows, sizeof(kQualcommRows) / sizeof(kQualcommRows[0]));
  }
} registrar;

} // namespace
} // namespace sarc
} // namespace vkcompute
