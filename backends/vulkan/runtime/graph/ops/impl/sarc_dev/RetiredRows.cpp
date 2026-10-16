/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SARC development zone: release rows retired by the family rows of
// openspec/changes/sarc-1.5-wmma-families (2026-10-16). Their variants are
// still shipped; as candidates they stay selectable by name, so the profiles
// that name them (the campaign controls) keep resolving.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

namespace vkcompute {
namespace sarc {
namespace {

const Row kRetiredRows[] = {
    // Arc B580 / Arc Pro B70, release 1.5 (sarc-1.5-4w-port, sarc-1.5-8da4w-port).
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x128k16g44s16m8fli",
     {128, 128, 16, 4, 4, 16, 8, false}, kTex3dTex2d | kBufTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpg_t256x64k32g48s16m8",
     {256, 64, 32, 4, 8, 16, 8, false}, kTex3dTex2d | kBufTex2d, nullptr,
     Status::kUnverified},
    // Jetson Orin, release 1.5.
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t128x128k32g42s32f32",
     {128, 128, 32, 4, 2, 32, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_linear_q4gsw_coopmat_t256x128k16g22s32",
     {256, 128, 16, 2, 2, 32, 16, false}, kTex3dTex2d, nullptr,
     Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_linear_dq8ca_coopmat_zpgtr_t128x128k64g44s32mk32ra",
     {128, 128, 64, 4, 4, 32, 16, true}, kTex3dTex2d, nullptr,
     Status::kUnverified, /*rowmajor_a=*/true},
};

struct Registrar {
  Registrar() {
    register_candidates(
        kRetiredRows, sizeof(kRetiredRows) / sizeof(kRetiredRows[0]));
  }
} registrar;

} // namespace
} // namespace sarc
} // namespace vkcompute
