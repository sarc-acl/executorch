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

// Battlemage (Xe2), verified on release 1.5 (openspec/changes/
// sarc-1.5-4w-port): 2048-token prefill 1B/3B/8B vs stock 1.5
// B580 8498/3352/1672 vs 3185/1149/524, B70 11636/4842/2421 vs
// 4592/1708/780 tok/s; kernel times 1.004x / 0.999x of the 1.4 study.
const Row kIntelRows[] = {
    {"bmg g21", // Arc B580
     nullptr,
     Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x128k16g44s16m8fli",
     kXe2Dims,
     kTex3dTex2d | kBufTex2d,
     nullptr,
     Status::kVerified},
    {"bmg g31", // Arc Pro B70
     nullptr,
     Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x128k16g44s16m8fli",
     kXe2Dims,
     kTex3dTex2d | kBufTex2d,
     nullptr,
     Status::kVerified},
};

struct Registrar {
  Registrar() {
    register_rows(kIntelRows, sizeof(kIntelRows) / sizeof(kIntelRows[0]));
  }
} registrar;

} // namespace
} // namespace sarc
} // namespace vkcompute
