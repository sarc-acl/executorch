/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// EXPERIMENT order-2/b70: the incumbent 4w kernel (sarc_linear_q4gsw_coopmat_sweep_t128x128k16g82s16m8flib) with a
// remapped (blocked) workgroup order (glsl/sarc_dev/sarc_exp_order2_linear_q4gsw_coopmat.yaml), one row per block
// width. Candidate rows: never chosen by a table or a profile; they fit only while ET_VK_SARC_EXP starts with
// "exp-order2", and are then selected by name with
// ET_VK_SARC_Q4GSW_VARIANT=<ident|n43|n28|n21|n11>_t128x128k16g82s16m8flib (needs ET_VK_SARC_UNVERIFIED=1).

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

#include <cstdlib>
#include <cstring>

namespace vkcompute {
namespace sarc {
namespace {

bool exp_order2_requested(const ShapeInfo&) {
  const char* e = std::getenv("ET_VK_SARC_EXP");
  return e != nullptr && std::strncmp(e, "exp-order2", 10) == 0;
}

const Row kExpOrder2LinearCandidates[] = {
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_exp_order2_linear_q4gsw_coopmat_ident_t128x128k16g82s16m8flib",
     {128, 128, 16, 8, 2, 16, 8, false}, kTex3dTex2d, exp_order2_requested, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_exp_order2_linear_q4gsw_coopmat_n43_t128x128k16g82s16m8flib",
     {128, 128, 16, 8, 2, 16, 8, false}, kTex3dTex2d, exp_order2_requested, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_exp_order2_linear_q4gsw_coopmat_n28_t128x128k16g82s16m8flib",
     {128, 128, 16, 8, 2, 16, 8, false}, kTex3dTex2d, exp_order2_requested, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_exp_order2_linear_q4gsw_coopmat_n21_t128x128k16g82s16m8flib",
     {128, 128, 16, 8, 2, 16, 8, false}, kTex3dTex2d, exp_order2_requested, Status::kUnverified},
    {"", nullptr, Op::kQ4gswLinear,
     "sarc_exp_order2_linear_q4gsw_coopmat_n11_t128x128k16g82s16m8flib",
     {128, 128, 16, 8, 2, 16, 8, false}, kTex3dTex2d, exp_order2_requested, Status::kUnverified},
};

struct Registrar {
  Registrar() {
    register_candidates(
        kExpOrder2LinearCandidates,
        sizeof(kExpOrder2LinearCandidates) / sizeof(kExpOrder2LinearCandidates[0]));
  }
} registrar;

} // namespace
} // namespace sarc
} // namespace vkcompute
