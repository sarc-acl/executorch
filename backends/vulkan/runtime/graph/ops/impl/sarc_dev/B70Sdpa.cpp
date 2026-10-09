/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SARC development zone, Intel Arc Pro B70 (device tag b70): SDPA prefill base
// rows for the device string "bmg g31" (openspec/changes/
// sarc-1.5-b70-fused-port). Not part of a release.
//
// Same mechanism and the same two rows as Xe2Sdpa.cpp, whose rows match
// xe2-* profile names only: kUnverified rows (ET_VK_SARC_UNVERIFIED=1) that
// match only while ET_VK_SARC_DEV_PROFILE names a b70-* profile, so every
// other configuration selects exactly what it selected before. The b70-*
// profiles (Overrides.cpp, b70-fused block) then pick the kernels by name.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

#include <cstdlib>
#include <cstring>

namespace vkcompute {
namespace sarc {
namespace {

bool b70_profile_requested(const DeviceInfo&) {
  const char* e = std::getenv("ET_VK_SARC_DEV_PROFILE");
  return e != nullptr && std::strncmp(e, "b70-", 4) == 0;
}

const Row kB70SdpaRows[] = {
    {"bmg g31", b70_profile_requested, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_sweep_t128x64k32g44s16m8nf",
     {128, 64, 32, 4, 4, 16, 8, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"bmg g31", b70_profile_requested, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_sweep_t64x64k32g44s16m8",
     {64, 64, 32, 4, 4, 16, 8, false}, kBufBuf, nullptr,
     Status::kUnverified},
};

struct Registrar {
  Registrar() {
    register_rows(
        kB70SdpaRows, sizeof(kB70SdpaRows) / sizeof(kB70SdpaRows[0]));
  }
} registrar;

} // namespace
} // namespace sarc
} // namespace vkcompute
