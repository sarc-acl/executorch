/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// EXPERIMENT order-1/b580: the incumbent 8da4w kernel with a remapped (blocked) workgroup order
// (glsl/sarc_dev/sarc_exp_order1_linear_dq8ca_coopmat_zpg_xe2bt.yaml), one row per block size. Candidate rows:
// never chosen by a table or a profile; they fit only while ET_VK_SARC_EXP starts with "exp-order1", and are then
// selected by name with ET_VK_SARC_DQ8CA_VARIANT=<nb1|b4m|b6m|b8m|b12m|ident>_t128x128k64g84s16m8 (needs
// ET_VK_SARC_UNVERIFIED=1).

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

#include <cstdlib>
#include <cstring>

namespace vkcompute {
namespace sarc {
namespace {

bool exp_order1_requested(const ShapeInfo&) {
  const char* e = std::getenv("ET_VK_SARC_EXP");
  return e != nullptr && std::strncmp(e, "exp-order1", 10) == 0;
}

const Row kExpOrder1LinearCandidates[] = {
    {"", nullptr, Op::kDq8caLinear,
     "sarc_exp_order1_linear_dq8ca_coopmat_zpg_xe2bt_nb1_t128x128k64g84s16m8",
     {128, 128, 64, 8, 4, 16, 8, false}, kTex3dTex2d, exp_order1_requested, Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_exp_order1_linear_dq8ca_coopmat_zpg_xe2bt_b4m_t128x128k64g84s16m8",
     {128, 128, 64, 8, 4, 16, 8, false}, kTex3dTex2d, exp_order1_requested, Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_exp_order1_linear_dq8ca_coopmat_zpg_xe2bt_b6m_t128x128k64g84s16m8",
     {128, 128, 64, 8, 4, 16, 8, false}, kTex3dTex2d, exp_order1_requested, Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_exp_order1_linear_dq8ca_coopmat_zpg_xe2bt_b8m_t128x128k64g84s16m8",
     {128, 128, 64, 8, 4, 16, 8, false}, kTex3dTex2d, exp_order1_requested, Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_exp_order1_linear_dq8ca_coopmat_zpg_xe2bt_b12m_t128x128k64g84s16m8",
     {128, 128, 64, 8, 4, 16, 8, false}, kTex3dTex2d, exp_order1_requested, Status::kUnverified},
    {"", nullptr, Op::kDq8caLinear,
     "sarc_exp_order1_linear_dq8ca_coopmat_zpg_xe2bt_ident_t128x128k64g84s16m8",
     {128, 128, 64, 8, 4, 16, 8, false}, kTex3dTex2d, exp_order1_requested, Status::kUnverified},
};

struct Registrar {
  Registrar() {
    register_candidates(
        kExpOrder1LinearCandidates,
        sizeof(kExpOrder1LinearCandidates) / sizeof(kExpOrder1LinearCandidates[0]));
  }
} registrar;

} // namespace
} // namespace sarc
} // namespace vkcompute
