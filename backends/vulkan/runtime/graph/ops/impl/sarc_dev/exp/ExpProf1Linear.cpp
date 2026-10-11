/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// EXPERIMENT prof-1/b580 (profiling task): one timing-only ablation of the incumbent 8da4w kernel
// (glsl/sarc_dev/sarc_exp_prof1_linear_dq8ca_coopmat_zpg_xe2bt_nobfetch.yaml), used to calibrate stall-sample
// shares against time. INCORRECT BY CONSTRUCTION: the in-loop weight texel fetch is removed. Candidate row:
// never chosen by a table or a profile; it fits only while ET_VK_SARC_EXP starts with "exp-prof1", and is then
// selected by name with ET_VK_SARC_DQ8CA_VARIANT=nobfetch_t128x128k64g84s16m8 (needs ET_VK_SARC_UNVERIFIED=1).

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

#include <cstdlib>
#include <cstring>

namespace vkcompute {
namespace sarc {
namespace {

bool exp_prof1_requested(const ShapeInfo&) {
  const char* e = std::getenv("ET_VK_SARC_EXP");
  return e != nullptr && std::strncmp(e, "exp-prof1", 9) == 0;
}

const Row kExpProf1LinearCandidates[] = {
    {"", nullptr, Op::kDq8caLinear,
     "sarc_exp_prof1_linear_dq8ca_coopmat_zpg_xe2bt_nobfetch_t128x128k64g84s16m8",
     {128, 128, 64, 8, 4, 16, 8, false}, kTex3dTex2d, exp_prof1_requested, Status::kUnverified},
};

struct Registrar {
  Registrar() {
    register_candidates(
        kExpProf1LinearCandidates,
        sizeof(kExpProf1LinearCandidates) / sizeof(kExpProf1LinearCandidates[0]));
  }
} registrar;

} // namespace
} // namespace sarc
} // namespace vkcompute
