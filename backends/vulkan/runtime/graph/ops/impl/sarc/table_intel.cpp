/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SARC selection rows for Intel GPUs. First matching row wins. Status
// kVerified requires sarc/tools/verify.sh evidence on that device.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

namespace vkcompute {
namespace sarc {
namespace {

// t128x128k16g44s16, MMA 8x16x16 (the only fp16 coopmat shape Xe2 exposes;
// a 16x16x16 pipeline is created without error and then miscomputes),
// fragment-contiguous LDS, imageLoad A.
constexpr TileDims kXe2Dims = {128, 128, 16, 4, 4, 16, 8, false};

// Battlemage (Xe2). 1.4 study (release/1.4, igpu-roofline
// docs/XE2-WMMA-LESSONS.md): 1B 4w prefill vs tiled B580 2563 -> 8292,
// B70 3690 -> 11636 tok/s. Not yet verified on release 1.5.
const Row kIntelRows[] = {
    {"bmg g21", // Arc B580
     nullptr,
     Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x128k16g44s16m8fli",
     kXe2Dims,
     kTex3dTex2d | kBufTex2d,
     nullptr,
     Status::kUnverified},
    {"bmg g31", // Arc Pro B70
     nullptr,
     Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x128k16g44s16m8fli",
     kXe2Dims,
     kTex3dTex2d | kBufTex2d,
     nullptr,
     Status::kUnverified},
};

struct Registrar {
  Registrar() {
    register_rows(kIntelRows, sizeof(kIntelRows) / sizeof(kIntelRows[0]));
  }
} registrar;

} // namespace
} // namespace sarc
} // namespace vkcompute
