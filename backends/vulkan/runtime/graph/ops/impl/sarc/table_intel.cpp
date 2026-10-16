/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SARC selection rows for Intel GPUs. First matching row wins. Status
// kVerified requires sarc/tools/verify.sh evidence on every member device.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

namespace vkcompute {
namespace sarc {
namespace {

bool xe2_weight_heavy(const ShapeInfo& s) {
  return s.K * s.N >= 24000000;
}

// Intel Xe2 family (Family::kXe2), openspec/changes/sarc-1.5-wmma-families:
// one kernel set for every member, accepted on the Arc B580 (BMG G21) and the
// Arc Pro B70 (BMG G31). Six-cell prefill geomean vs each card's tuned profile
// (linear kernels only): B580 1.0068, B70 0.9964, every cell in the noise band.
// MMA 8x16x16 / 8x16x32 (the only fp16 / int8 coopmat shapes Xe2 exposes; a
// 16x16x16 pipeline is created without error and then miscomputes),
// subgroup 16, fragment-contiguous LDS, imageLoad A.
const Row kIntelRows[] = {
    // 4w, texture3d: banded drain ('b'), 8 x 2 subgroups.
    {"", in_xe2_family, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x128k16g82s16m8flib",
     {128, 128, 16, 8, 2, 16, 8, false}, kTex3dTex2d, nullptr,
     Status::kVerified},
    // 4w, buffer IO: the release-1.5 Xe2 tile (sarc-1.5-4w-port).
    {"", in_xe2_family, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x128k16g44s16m8fli",
     {128, 128, 16, 4, 4, 16, 8, false}, kBufTex2d, nullptr,
     Status::kVerified},
    // 8da4w: zpg with texel-wise B staging and a full A thread map. On the
    // weight-heavy projections (K * N >= 24e6) workgroups run in blocks of at
    // most 13 N tiles, all M tiles of a block first ('n13'); elsewhere in
    // dispatch order ('i').
    {"", in_xe2_family, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8n13",
     {128, 128, 64, 8, 4, 16, 8, false}, kTex3dTex2d | kBufTex2d,
     xe2_weight_heavy, Status::kVerified},
    {"", in_xe2_family, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8i",
     {128, 128, 64, 8, 4, 16, 8, false}, kTex3dTex2d | kBufTex2d, nullptr,
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
