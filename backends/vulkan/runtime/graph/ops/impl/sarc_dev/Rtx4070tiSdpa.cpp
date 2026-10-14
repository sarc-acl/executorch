/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SARC development zone, RTX 4070 Ti SUPER (device tag 4070ti): SDPA prefill
// base rows (openspec/changes/sarc-1.5-4070ti-prefill-refine). Not part of a
// release.
//
// The release tables have no NVIDIA SDPA row, and impl/sarc/SdpaCoopmat.cpp
// builds the SARC SDPA path (spec constants, truncated softmax) only on a
// device with an active SDPA row. The rows below are that row, from the dev
// zone: kUnverified (so they need ET_VK_SARC_UNVERIFIED=1) and matching only
// while ET_VK_SARC_DEV_PROFILE names a 4070ti-* profile, so every other
// configuration of a dev build selects exactly what the release tables select.
// Their kernels are candidates of impl/sarc_dev/Overrides.cpp; WG_TILE_K is
// 32, which the spec constants assume. The profile picks the kernel per shape.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h>

#include <cstdlib>
#include <cstring>

namespace vkcompute {
namespace sarc {
namespace {

bool profile_4070ti_requested(const DeviceInfo&) {
  const char* e = std::getenv("ET_VK_SARC_DEV_PROFILE");
  return e != nullptr && std::strncmp(e, "4070ti-", 7) == 0;
}

const Row k4070tiSdpaRows[] = {
    {"4070 ti super", profile_4070ti_requested, Op::kSdpaQk,
     "sarc_sdpa_qk_coopmat_4070ti_t128x64k32g42s32",
     {128, 64, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
    {"4070 ti super", profile_4070ti_requested, Op::kSdpaAv,
     "sarc_sdpa_av_coopmat_4070ti_t64x64k32g42s32",
     {64, 64, 32, 4, 2, 32, 16, false}, kBufBuf, nullptr,
     Status::kUnverified},
};

struct Registrar {
  Registrar() {
    register_rows(
        k4070tiSdpaRows, sizeof(k4070tiSdpaRows) / sizeof(k4070tiSdpaRows[0]));
  }
} registrar;

} // namespace
} // namespace sarc
} // namespace vkcompute
